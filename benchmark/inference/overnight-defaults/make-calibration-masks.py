"""Compute-node format conversion of saved, independently checked recession masks."""
import csv,json
from pathlib import Path
for path in sorted(Path('outputs/calibration-recession').glob('*/*/checks.json')):
 rows=json.loads(path.read_text())
 with path.with_name('mask.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['scope','fold','level','unbounded_witness'])
  for r in rows:w.writerow([r['scope'],r['fold'],r['level'],'TRUE' if r.get('unbounded_witness') else 'FALSE'])
