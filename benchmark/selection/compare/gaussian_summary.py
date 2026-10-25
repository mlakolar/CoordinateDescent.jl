import csv,json,math,os,statistics as st
from collections import defaultdict
from pathlib import Path
out=Path(os.environ['RUN'])/'results'
rows=[r for p in out.glob('gaussian-mc-*.csv') for r in csv.DictReader(p.open())]
groups=defaultdict(list)
lookup={}
for r in rows:
    key=int(r['scenario']),r['method'];seed=int(r['seed'])
    assert (key[0],key[1],seed) not in lookup
    groups[key].append(r);lookup[key[0],key[1],seed]=r
    ids=list(map(int,filter(None,r['selected_ids'].split(';'))))
    assert len(ids)==len(set(ids))==int(r['selected'])
    assert sum(j>int(r['k']) for j in ids)==int(r['false'])
    assert abs(float(r['fdp'])-int(r['false'])/max(1,len(ids)))<1e-12
def stats(v):
    v=[x for x in v if math.isfinite(x)]
    return (st.mean(v),st.stdev(v)/math.sqrt(len(v))) if v else (float('nan'),float('nan'))
summary=[];agreements=[]
names=['iid','AR(.4)','AR(.8)','weak AR(.4)','null AR(.4)','wide AR(.4)']
for sc in range(1,7):
    for method in ['gaussian-CD','gaussian-SLOPE.jl','gaussian-R-common','gaussian-R-native']:
        rs=groups[sc,method];assert len(rs)==140
        fdr,fdrse=stats([float(r['fdp']) for r in rs]);power,powerse=stats([float(r['power']) for r in rs])
        summary.append(dict(scenario=names[sc-1],method=method,n=140,mean_fdp=fdr,fdp_se=fdrse,power=power,power_se=powerse))
        if method!='gaussian-CD':
            mismatches=sum(set(filter(None,r['selected_ids'].split(';')))!=set(filter(None,lookup[sc,'gaussian-CD',int(r['seed'])]['selected_ids'].split(';'))) for r in rs)
            agreements.append(dict(scenario=names[sc-1],method=method,n=140,selection_mismatches=mismatches))
weight_rows=[r for p in out.glob('gaussian-weight-difference-*.csv') for r in csv.DictReader(p.open())]
for name,data in [('gaussian-simulation-summary.csv',summary),('gaussian-selection-agreement.csv',agreements)]:
    with (out/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
receipt=dict(datasets=840,rows=len(rows),common_weight_selection_mismatches=sum(r['selection_mismatches'] for r in agreements if r['method']!='gaussian-R-native'),native_R_selection_mismatches=sum(r['selection_mismatches'] for r in agreements if r['method']=='gaussian-R-native'),max_native_R_weight_difference=max(float(r['max_weight_difference']) for r in weight_rows))
(out/'gaussian-validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
