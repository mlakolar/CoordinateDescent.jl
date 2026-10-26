"""Independent original-design certificates; execute under Slurm only."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.special import ndtri

ROOT = Path(os.environ['HDM_STUDY'])

def read_array(f, dtype, count):
    a = np.fromfile(f, dtype=dtype, count=count)
    if len(a) != count:
        raise ValueError('truncated record')
    return a

def cases(path):
    with path.open('rb') as f:
        while True:
            h = np.fromfile(f, dtype='<i8', count=8)
            if len(h) == 0:
                return
            if len(h) != 8:
                raise ValueError('truncated header')
            n, p, k, sc, seed, intercept, dataset, nw = map(int, h)
            sigma = read_array(f, '<f8', 1)[0]
            X = read_array(f, '<f8', n*p).reshape((n, p), order='F')
            y = read_array(f, '<f8', n)
            beta = read_array(f, '<f8', p)
            W = read_array(f, '<f8', p*nw).reshape((p, nw), order='F')
            yield dict(n=n, p=p, k=k, sc=sc, seed=seed, intercept=bool(intercept),
                       dataset=dataset, sigma=sigma, X=X, y=y, beta=beta, W=W)

def fits(path):
    with path.open('rb') as f:
        while True:
            h = np.fromfile(f, dtype='<i8', count=3)
            if len(h) == 0:
                return
            seed, p, nm = map(int, h)
            W = read_array(f, '<f8', p*nm).reshape((p, nm), order='F')
            B = read_array(f, '<f8', (p+1)*nm).reshape((p+1, nm), order='F')
            yield seed, W, B

def weights(p, n, sigma, intercept, rule):
    # Independent SciPy quantiles and recurrence, with one-based indexing.
    a = -ndtri(.1*np.arange(1, p+1)/(2*p))
    if rule != 'bh':
        m = n-int(intercept)
        total = 0.
        for j in range(1, p):
            total += a[j-1]**2
            denom = m-(j+1)
            if rule == 'paper' and denom <= 0:
                a[j:] = a[j-1]
                break
            denom = max(1, denom-(rule == 'julia'))
            nxt = a[j]*np.sqrt(1+total/denom)
            if rule == 'paper' and nxt >= a[j-1]:
                a[j:] = a[j-1]
                break
            a[j] = nxt
        if rule in ('julia', 'R-native'):
            a = np.minimum.accumulate(a)
    return sigma/np.sqrt(n)*a

def prox(v, w):
    order = np.argsort(-np.abs(v), kind='stable')
    x = np.abs(v[order])-w
    values, sizes = [], []
    for value in x:
        values.append(float(value)); sizes.append(1)
        while len(values) > 1 and values[-2] < values[-1]:
            size = sizes[-2]+sizes[-1]
            value = (sizes[-2]*values[-2]+sizes[-1]*values[-1])/size
            values[-2:] = [value]; sizes[-2:] = [size]
    pooled = np.repeat(np.maximum(values, 0), sizes)
    result = np.empty_like(v)
    result[order] = np.sign(v[order])*pooled
    return result

def certificate(c, w, b):
    X, y, n, p = c['X'], c['y'], c['n'], c['p']
    if not np.isfinite(b).all():
        return dict(valid=False, residual=float('inf'), objective=float('nan'),
                    selected=0, false=0, fdp=float('nan'), power=float('nan'), selected_ids='')
    beta, b0 = b[:p], b[p]
    r = X@beta+b0-y
    Z = X-X.mean(axis=0) if c['intercept'] else X
    v = y-y.mean() if c['intercept'] else y
    scale = max(1., np.linalg.norm(Z.T@v/n, np.inf))
    g = Z.T@r/n
    residual = np.linalg.norm(beta-prox(beta-g, w), np.inf)/scale
    if c['intercept']:
        residual = max(residual, abs(r.mean())/max(1., abs(y.mean())))
    else:
        residual = max(residual, abs(b0))
    obj = float(r@r/(2*n)+w@np.sort(np.abs(beta))[::-1])
    s = np.flatnonzero(np.abs(beta) > 1e-6)+1
    false = int(np.count_nonzero(s > c['k']))
    return dict(valid=bool(residual <= 1e-7), residual=residual, objective=obj,
                selected=len(s), false=false, fdp=false/max(1, len(s)),
                power=(len(s)-false)/c['k'] if c['k'] else float('nan'),
                selected_ids=';'.join(map(str, s)))

def check(sc, shard):
    input_path = ROOT/'inputs'/f'sc{sc}-shard{shard}.bin'
    expected_hash = Path(str(input_path)+'.sha256').read_text().split()[0]
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == expected_hash
    engine_iters = {e: fits(ROOT/'fits'/f'{e}-sc{sc}-shard{shard}.bin') for e in ('native','julia','R')}
    output = ROOT/'checks'/f'sc{sc}-shard{shard}'
    output.parent.mkdir(exist_ok=True)
    rows, pairs = [], []
    auto_equal = True
    for c in cases(input_path):
        for j, rule in enumerate(('bh', 'paper', 'julia')):
            np.testing.assert_allclose(c['W'][:,j], weights(c['p'], c['n'], c['sigma'], c['intercept'], rule), rtol=2e-12, atol=1e-14)
        got = {}
        for engine, it in engine_iters.items():
            seed, W, B = next(it)
            assert seed == c['seed'] and B.shape[0] == c['p']+1
            methods = ['bh', 'paper', 'julia']+(['automatic'] if engine == 'native' else ['R-native'] if engine == 'R' else [])
            assert B.shape[1] == len(methods)
            for j, rule in enumerate(methods):
                actual_rule = 'paper' if rule == 'automatic' else rule
                np.testing.assert_allclose(W[:,j], weights(c['p'], c['n'], c['sigma'], c['intercept'], actual_rule), rtol=2e-12, atol=1e-14)
                cert = certificate(c, W[:,j], B[:,j])
                row = dict(scenario=sc, shard=shard, seed=seed, engine=engine, method=rule, **cert)
                rows.append(row)
                got[engine, rule] = (B[:,j], cert)
            if engine == 'native':
                auto_equal &= np.array_equal(B[:,1], B[:,3])
        for method in ('bh','paper','julia'):
            a, ca = got['native', method]
            for e in ('julia','R'):
                b, cb = got[e, method]
                sa, sb = set(ca['selected_ids'].split(';')), set(cb['selected_ids'].split(';'))
                pairs.append(dict(scenario=sc, shard=shard, seed=c['seed'], method=method, engine=e,
                    both_valid=ca['valid'] and cb['valid'],
                    coefficient_relative=np.max(np.abs(a-b))/max(1., np.max(np.abs(a))),
                    objective_relative=abs(ca['objective']-cb['objective'])/max(1.,abs(ca['objective'])),
                    selection_disagreement=len(sa.symmetric_difference(sb))))
    for it in engine_iters.values():
        assert next(it, None) is None, 'extra fit record'
    for name, values in (('.csv', rows), ('.agreement.csv', pairs)):
        with Path(str(output)+name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    receipt = dict(scenario=sc, shard=shard, certificates=len(rows), failed=sum(not r['valid'] for r in rows),
                   max_residual=max(r['residual'] for r in rows), automatic_bitwise_equal=auto_equal,
                   input_sha256=expected_hash, common_pairs=len(pairs),
                   max_common_objective_relative=max(r['objective_relative'] for r in pairs))
    Path(str(output)+'.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)
    # The pilot gate is strict; preserve every failed result before failing it.
    if os.environ.get('HDM_GATE') == 'pilot':
        assert receipt['failed'] == 0 and auto_equal and receipt['max_common_objective_relative'] <= 1e-7
    return receipt

if __name__ == '__main__':
    check(int(sys.argv[1]), int(sys.argv[2]))
