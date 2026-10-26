import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
from study_check import cases, fits, certificate

root=Path(os.environ['HDM_STUDY']);sc=int(sys.argv[1])
cs={c['seed']:c for c in cases(root/'inputs'/f'sc{sc}-shard0.bin')}
meta=list(csv.DictReader((root/'stopping-probe-v1'/f'sc{sc}.csv').open()))
rows=[]
for m,(seed,W,B) in zip(meta,fits(root/'stopping-probe-v1'/f'sc{sc}.bin'),strict=True):
    assert seed==int(m['seed']) and B.shape[1]==20
    certs=[certificate(cs[seed],W[:,j],B[:,j]) for j in range(20)]
    rows.append(dict(**m,certificates=20,failed=sum(not c['valid'] for c in certs),max_independent_residual=max(c['residual'] for c in certs)))
result=dict(case=sc,inputs=3,certificates=sum(r['certificates'] for r in rows),failed=sum(r['failed'] for r in rows),
 max_residual=max(r['max_independent_residual'] for r in rows),reported_nonconvergence=sum(r['converged']!='true' for r in rows),rows=rows,
 scope='tol1e-8/max100k diagnostic on existing rep0 inputs; one cold call/workload including JIT, excluded from runtime rankings; no changed accuracy threshold')
(root/'stopping-probe-v1'/f'sc{sc}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
