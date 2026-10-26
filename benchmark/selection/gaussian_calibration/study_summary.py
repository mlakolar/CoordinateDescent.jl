"""Monte Carlo reporting from saved certificates only; run under Slurm."""
import csv
from collections import defaultdict
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT=Path(os.environ['HDM_STUDY'])
SCENARIOS=['iid-alt','iid-same','ar04-alt','ar04-same','ar08-alt','ar08-same',
 'weak-ar04-alt','weak-ar04-same','wide-ar04-alt','wide-ar04-same','null-ar04',
 'orth-alt','orth-same','estimated-iid','estimated-ar04','block-alt']

def estimate(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if not len(a):return dict(n=0,mean=None,se=None,lo=None,hi=None)
    mean=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None
    margin=float(t.ppf(.975,len(a)-1)*se) if se is not None else None
    return dict(n=len(a),mean=mean,se=se,lo=mean-margin if margin is not None else None,hi=mean+margin if margin is not None else None)

def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    out=ROOT/'report';out.mkdir(exist_ok=True)
    check_dir=ROOT/os.environ.get('HDM_CHECK_DIR','checks')
    files=sorted(check_dir.glob('sc*-shard*.json'))
    receipts=[json.loads(p.read_text()) for p in files]
    rows=[];groups=defaultdict(list);paired=defaultdict(dict)
    for p in sorted(check_dir.glob('sc*-shard*.csv')):
        if '.agreement.' in p.name:continue
        for r in csv.DictReader(p.open()):
            rows.append(r);groups[int(r['scenario']),r['engine'],r['method']].append(r)
            paired[int(r['scenario']),r['engine'],int(r['seed'])][r['method']]=r
    summary=[]
    for (sc,engine,method),rs in sorted(groups.items()):
        valid=[r for r in rs if r['valid']=='True']
        d=dict(scenario=sc,name=SCENARIOS[sc-1],cohort='stress' if sc>=14 else 'main',engine=engine,method=method,
               declared=len(rs),valid=len(valid),failed=len(rs)-len(valid))
        for metric in ('fdp','power','selected','false'):
            for k,v in estimate([float(r[metric]) for r in valid]).items():d[f'{metric}_{k}']=v
        summary.append(d)
    differences=[]
    for sc in range(1,17):
        for engine in ('native','julia','R'):
            for method in ('paper','julia'):
                rs=[v for (s,e,_),v in paired.items() if (s,e)==(sc,engine)]
                valid=[v for v in rs if v['bh']['valid']=='True' and v[method]['valid']=='True']
                d=dict(scenario=sc,name=SCENARIOS[sc-1],engine=engine,method=method,reference='bh',paired_inputs=len(valid))
                for metric in ('fdp','power','selected','false'):
                    vals=[float(v[method][metric])-float(v['bh'][metric]) for v in valid]
                    for k,v in estimate(vals).items():d[f'{metric}_difference_{k}']=v
                differences.append(d)
    write_csv(out/'monte-carlo.csv',summary);write_csv(out/'paired-differences.csv',differences)
    agreements=[]
    for p in sorted(check_dir.glob('*.agreement.csv')):agreements.extend(csv.DictReader(p.open()))
    agreement=dict(pairs=len(agreements),invalid_pairs=sum(r['both_valid']!='True' for r in agreements),
       selection_disagreements=sum(int(r['selection_disagreement'])>0 for r in agreements),
       max_coefficient_relative=max(float(r['coefficient_relative']) for r in agreements),
       max_objective_relative=max(float(r['objective_relative']) for r in agreements))
    api=[]
    for p in sorted((ROOT/'fits').glob('*.csv')):api.extend(csv.DictReader(p.open()))
    result=dict(receipts=len(receipts),independent_inputs=len(paired)//3,certificates=len(rows),
      failed_certificates=sum(r['valid']!='True' for r in rows),
      automatic_all_bitwise_equal=all(r['automatic_bitwise_equal'] for r in receipts),
      max_residual=max(float(r['residual']) for r in rows),common_weight_agreement=agreement,
      native_reported_nonconvergence=sum(r['reported_converged']=='false' for r in api),
      api_errors=sum(bool(r['error']) for r in api),
      intervals='pointwise approximate Student-t Monte Carlo intervals over independent datasets; no multiple-comparison adjustment; null power undefined',
      interpretation='FDR=mean FDP. Do not use empirical FDR<=q as a correctness gate outside the known-noise orthogonal theorem. Invalid fits excluded with explicit counts; no substitution of zeros.')
    if os.environ.get('HDM_GATE')=='main':
        assert len(receipts)==112 and len(paired)==16*1008*3 and len(rows)==16*1008*11
        assert all(len(v)==1008 for v in groups.values())
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
