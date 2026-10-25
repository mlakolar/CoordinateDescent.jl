import subprocess, resource, time, json, sys, pathlib
destination=pathlib.Path(sys.argv[1])
start=time.monotonic()
process=subprocess.run(sys.argv[2:])
r=resource.getrusage(resource.RUSAGE_CHILDREN)
destination.write_text(json.dumps(dict(exit_code=process.returncode,elapsed_seconds=time.monotonic()-start,
    peak_rss_bytes=r.ru_maxrss*1024,user_seconds=r.ru_utime,system_seconds=r.ru_stime),indent=2))
sys.exit(process.returncode)
