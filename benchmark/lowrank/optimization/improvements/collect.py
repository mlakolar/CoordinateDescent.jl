"""Package reporting tables and many short profiles; retain matrices remotely."""
import shutil
import tarfile
from pathlib import Path

dest=Path("delivery");dest.mkdir(exist_ok=True)
def copy(src,relative):
    target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,target)
for p in Path("results/summary").iterdir():
    if p.is_file():copy(p,p.name)
for version in ("1.13.2","1.10.12"):
    for name in ("Project","Manifest"):
        copy(f"env-{version}/{name}.toml",f"provenance/Julia-{version}-{name}.toml")
    log=Path(f"results/test-v2/julia-{version}.log")
    if not log.is_file():log=Path(f"results/prep/julia-{version}.log")
    copy(log,f"provenance/tests-{version}.log")
for name in ("source-sha256.txt","correctness.log","platform.txt"):
    copy(Path("results")/name,f"provenance/{name}")
copy("/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/results/python-freeze.txt","provenance/python-freeze.txt")
copy("/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp/results/input-sha256.txt","provenance/input-sha256.txt")
for name in ("source-sha256.txt","platform.txt","selection.log","joblog.tsv","recovery-v5-sha256.txt","recovery-v7-sha256.txt","recovery-v8-sha256.txt","r-component-joblog.tsv"):
    src=Path("results/recovery")/name
    if src.is_file():copy(src,f"provenance/recovery-{name}")
for name in ("source-sha256.txt","joblog.tsv"):
    src=Path("results/core-v2")/name
    if src.is_file():copy(src,f"provenance/core-v2-{name}")
for name in ("source-sha256.txt","joblog.tsv"):
    src=Path("results/focused-memory")/name
    if src.is_file():copy(src,f"provenance/focused-memory-{name}")
for name in ("source-sha256.txt","joblog.tsv","large-joblog.tsv","reused.json","platform.txt"):
    src=Path("results/memory-pass")/name
    if src.is_file():copy(src,f"provenance/memory-pass-{name}")
profiles=[];diagnostics=[]
for base in ("main","before","adaptive"):
    profiles+=sorted(Path("results",base).glob("*-*/*/profile-*"))
for pattern in ("inputs/*/meta.json","results/main/*/python/python-environment.json",
    "results/main/*/r/r-environment.txt","results/main/*/*/trace-*.csv","results/main/*/*/setup.csv"):
    diagnostics+=sorted(Path().glob(pattern))
for name,files in (("profiles",profiles),("diagnostics",diagnostics)):
    with tarfile.open(dest/f"{name}.tar.gz","w:gz") as t:
        for p in files:t.add(p,arcname=str(p),recursive=False)
print(f"Collected {len(profiles)} profile files and {len(diagnostics)} diagnostic files")
