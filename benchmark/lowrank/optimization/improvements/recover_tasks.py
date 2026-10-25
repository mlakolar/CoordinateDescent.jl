"""Select missing/failed steps without refitting completed measurements."""
import csv
import re
import shutil
from pathlib import Path

complete=set()
with open("results/main/joblog.tsv") as f:
    for r in csv.DictReader(f,delimiter="\t"):
        m=re.search(r"worker\.sh (\S+) (\S+)",r["Command"])
        if m and int(r["Exitval"])==0:complete.add(m.groups())
tasks=[]
for line in Path("main-tasks.txt").read_text().splitlines():
    case,language=line.split()
    before=language=="before"
    folder=Path("results/before" if before else "results/main")/case/("julia" if before else language)
    filename="julia.csv" if language in ("julia","before") else "python.csv" if language=="python" else "r.csv"
    count=150 if language=="julia" else 27 if before else 33 if language=="python" else 15 if case.startswith("responses-") else 24
    file=folder/filename
    valid=(case,language) in complete and file.is_file()
    if valid:
        with file.open() as f:valid=len(list(csv.DictReader(f)))==count
    if not valid:
        if file.is_file():
            target=Path("results/recovery/partial")/(case+"-"+language)
            shutil.copytree(folder,target,dirs_exist_ok=True)
        tasks.append(line)
Path("recovery-tasks.txt").write_text("\n".join(tasks)+"\n")
print(f"Reusing {len(complete)} successful receipts; running {len(tasks)} missing/failed ordinary steps")
