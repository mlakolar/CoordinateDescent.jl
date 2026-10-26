"""Check every timed runtime output; no refitting. Slurm execution only."""
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
from study_check import cases, fits, certificate, weights

ROOT = Path(os.environ['HDM_STUDY'])

def check(sc, rep):
    cs = list(cases(ROOT/'inputs'/f'sc{sc}-shard{rep}.bin'))
    assert len(cs) == 3
    by_seed = {c['seed']: c for c in cs}
    for c in cs:
        np.testing.assert_allclose(np.mean(c['X']**2, axis=0), 1, rtol=1e-12, atol=1e-12)
        for j, rule in enumerate(('bh','paper','julia')):
            np.testing.assert_allclose(c['W'][:,j], weights(c['p'],c['n'],1,False,rule),rtol=2e-12,atol=1e-14)
    allrows, saved = [], {}
    for engine in ('native','julia','R'):
        stem = ROOT/'fits'/f'{engine}-sc{sc}-shard{rep}'
        meta = list(csv.DictReader(Path(str(stem)+'.records.csv').open()))
        it = fits(Path(str(stem)+'.bin'))
        for m in meta:
            seed, W, B = next(it)
            assert seed == int(m['seed']) and B.shape[1] == int(m['ncolumns'])
            c = by_seed[seed]
            base = c['W'][:,('bh','paper','julia').index(m['method'])]
            amax = np.max(np.cumsum(np.sort(np.abs(c['X'].T@c['y']/c['n']))[::-1])/np.cumsum(base))
            factors = np.ones(1) if m['payload']=='fit' else np.geomspace(amax,.2*amax,20)
            np.testing.assert_allclose(W, base[:,None]*factors,rtol=2e-12,atol=1e-14)
            for j in range(B.shape[1]):
                cert = certificate(c, W[:,j], B[:,j])
                allrows.append(dict(scenario=sc, rep=rep, engine=engine, **m, point=j, **cert))
            key = seed, m['method'], m['payload'], m['setting'], int(m['round']), engine
            saved[key] = B
        assert next(it, None) is None
    for seed in by_seed:
        for method in ('bh','paper','julia'):
            for payload in ('fit','path'):
                for r in (1,2,3):
                    np.testing.assert_array_equal(saved[seed,method,payload,'public',r,'native'],saved[seed,method,payload,'prepared',r,'native'])
    witness = list(csv.DictReader((ROOT/'fits'/f'native-sc{sc}-shard{rep}.witness.csv').open()))
    assert len(witness) == 18 and all(w['complete_output_equal']=='true' for w in witness)
    assert all(int(w['allocation_difference']) > 0 for w in witness)
    out = ROOT/'checks'/f'sc{sc}-shard{rep}'
    out.parent.mkdir(exist_ok=True)
    with Path(str(out)+'.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    counts = {s:sum(r['setting']==s for r in allrows) for s in ('public','prepared','default','unscreened')}
    receipt = dict(case=sc,rep=rep,inputs=3,certificates=len(allrows),failed=sum(not r['valid'] for r in allrows),
        max_residual=max(r['residual'] for r in allrows),settings=counts,public_prepared_bitwise=True,positive_allocation_witnesses=len(witness))
    Path(str(out)+'.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    if os.environ.get('HDM_GATE')=='pilot':
        assert receipt['failed']==0

if __name__=='__main__':check(int(sys.argv[1]),int(sys.argv[2]))
