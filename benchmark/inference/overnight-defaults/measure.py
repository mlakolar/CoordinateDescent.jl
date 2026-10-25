"""Bound one process tree and collect Linux child-resource usage."""
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

path, *cmd = sys.argv[1:]
started = time.monotonic()
p = subprocess.Popen(cmd, start_new_session=True)
try:
    rc = p.wait(timeout=10000)
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    rc = 124
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
Path(path).write_text(f"process_seconds={time.monotonic()-started}\n"
                      f"max_rss_kib={usage.ru_maxrss}\n"
                      f"cpu_seconds={usage.ru_utime+usage.ru_stime}\n"
                      f"process_exit={rc}\n")
sys.exit(rc if rc >= 0 else 128-rc)
