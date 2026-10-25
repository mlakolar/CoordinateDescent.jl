"""Select missing memory observations; retain completed rows and partial files."""
import csv
import json
import shutil
from pathlib import Path

methods = {
    "julia": ["cd_rank", "cd_nuclear", "cd_panoc", "ps_fista", "ps_admm", "pa_fista",
              "pa_panoc", "cd_sparse", "block_fista", "cd_sparse_api", "cd_sparse_fista",
              "cd_component", "component_fista"],
    "before": ["before_rank", "before_nuclear", "before_sparse"],
    "python": ["scipy_rrr", "py_fista", "py_admm"],
    "r": ["r_rrr", "r_rrr_qr", "r_srrr", "r_component"],
}
ordinary, large, reused = [], [], []
for language, solvers in methods.items():
    for solver in solvers:
        for case in ("responses", "wide"):
            for rep in range(1, 6):
                case_id = f"{case}-{rep}"
                folder = Path("results/memory") / case_id / solver
                file = folder / ("memory.json" if language == "python" else "memory.csv")
                valid = False
                try:
                    rows = [json.loads(file.read_text())] if language == "python" else list(csv.DictReader(file.open()))
                    if len(rows) == 1:
                        r = rows[0]
                        valid = (r["case"] == case and int(r["rep"]) == rep and r["solver"] == solver
                                 and r["status"] == "passed" and float(r["rss_peak_kb"]) > 0)
                except (OSError, ValueError, KeyError):
                    pass
                task = f"{case_id} {language} {solver}"
                if valid:
                    reused.append(task)
                    continue
                if file.is_file():
                    target = Path("results/memory-pass/partial") / case_id / solver / file.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(file, target)
                (large if case == "responses" and solver == "r_component" else ordinary).append(task)
Path("memory-tasks.txt").write_text("\n".join(ordinary) + "\n")
Path("memory-large-tasks.txt").write_text("\n".join(large) + "\n")
Path("results/memory-pass/reused.json").write_text(json.dumps(reused, indent=2) + "\n")
print(f"Reusing {len(reused)} completed memory observations; running {len(ordinary)} ordinary and {len(large)} large steps")
