"""Fail clearly on absent replication coverage, independently of Slurm exits."""
import csv
import json
import sys
from pathlib import Path
root=Path(sys.argv[1])
rows=list(csv.DictReader(open(root/"summary/all-timings.csv")))
keys={(r["case"],r["family"],r["solver"],r["mode"],r["frac"],r["rep"],r["round"]) for r in rows}
profiles=list((root/"main").glob("*-*/*/profile-*.txt"))
memory=list(csv.DictReader(open(root/"summary/memory.csv")))
result=dict(timing_rows=len(rows),unique_timing_keys=len(keys),expected_timing_rows=5250,
    profile_reports=len(profiles),expected_profiles=255,
    memory_observations=len(memory),expected_memory_observations=170,
    distinct_reps=sorted({int(r["rep"]) for r in rows}))
(root/"summary/coverage.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
assert len(rows)==len(keys)==5250,result
assert len(profiles)==255,result
assert len(memory)==170,result
assert result["distinct_reps"]==[1,2,3,4,5],result
