"""Linux child-process peak RSS, including interpreter/JIT/imports; KiB units."""
import json, resource, subprocess, sys, time
started=time.monotonic()
result=subprocess.run(sys.argv[2:], check=False)
r=resource.getrusage(resource.RUSAGE_CHILDREN)
with open(sys.argv[1], 'w') as f:
    json.dump({'exit_code':result.returncode,'peak_rss_kib':r.ru_maxrss,
               'elapsed_seconds':time.monotonic()-started,'command':sys.argv[2:]},f,indent=2)
sys.exit(result.returncode)
