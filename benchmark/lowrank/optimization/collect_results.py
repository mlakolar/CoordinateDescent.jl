"""Collect compact reporting artifacts; leave matrices in the run directory."""
import shutil
import tarfile
from pathlib import Path

dest = Path("delivery")
dest.mkdir(exist_ok=True)

def copy(src, relative):
    src = Path(src)
    target = dest / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, target)

for file in Path("results/summary").iterdir():
    if file.is_file(): copy(file, "full-table.md" if file.name == "README.md" else file.name)
for name in ("adaptive", "rank-core", "rank-matched"):
    for filename in ("all-timings.csv", "independent-certificates.csv", "summary.json", "aggregate.csv"):
        file = Path("results") / name / filename
        if file.is_file(): copy(file, f"{name}/{filename}")
copy("results/main/independent-certificates.csv", "independent-certificates.csv")
copy("results/main/validation-summary.json", "validation-summary.json")
for file in ("python-freeze.txt", "python-config.txt", "platform.txt", "source-sha256.txt",
             "input-sha256.txt", "r-configuration.txt"):
    copy(Path("results") / file, f"provenance/{file}")
copy("results/r-precision/source-sha256.txt", "provenance/r-source-sha256.txt")
copy("results/finish/source-sha256.txt", "provenance/supplement-source-sha256.txt")
copy("results/rank-matched/source-sha256.txt", "provenance/rank-target-source-sha256.txt")
copy("env/Project.toml", "provenance/JuliaProject.toml")
copy("env/Manifest.toml", "JuliaManifest.toml")
copy("results/pilot/cvxpy-reference.json", "cvxpy-reference.json")
diagnostics = []
for pattern in ("inputs/*/meta.json", "results/main/*/python/python-environment.json",
                "results/main/*/r/r-environment.txt", "results/main/*/*/trace-*.csv",
                "results/main/*/*/setup.csv"):
    diagnostics += sorted(Path().glob(pattern))
with tarfile.open(dest / "diagnostics.tar.gz", "w:gz") as archive:
    for file in diagnostics: archive.add(file, arcname=str(file), recursive=False)

# Hundreds of short text/raw profiles are kept together for practical transfer.
profiles = []
for base in ("main", "adaptive", "rank-core", "rank-matched"):
    profiles += sorted(Path(f"results/{base}").glob("*-*/*/profile-*"))
with tarfile.open(dest / "profiles.tar.gz", "w:gz") as archive:
    for file in profiles: archive.add(file, arcname=str(file), recursive=False)
print(f"Collected {len(list(dest.rglob('*')))} reporting paths and {len(profiles)} profile files")
