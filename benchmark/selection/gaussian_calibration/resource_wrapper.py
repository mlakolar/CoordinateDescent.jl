"""One worker/process tree per Slurm step; preserve scheduler-independent exit/RSS."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

destination=Path(sys.argv[1]);start=time.monotonic()
returncode=subprocess.call(sys.argv[2:])
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
destination.write_text(json.dumps(dict(returncode=returncode,wall_seconds=time.monotonic()-start,
    user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,max_rss_bytes=usage.ru_maxrss*1024,
    scope='whole child process tree including imports/JIT/warmup/timers/profiles; Linux maxrss KiB'),indent=2)+'\n')
sys.exit(returncode)
