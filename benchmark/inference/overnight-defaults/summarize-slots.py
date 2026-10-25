"""Compute-node complete-dataset summaries and fixed target-role Monte Carlo errors.

In random-support designs a physical column changes truth across datasets.
Use requested target position, not pooled physical-column labels, for comparisons.
"""
import csv,json,math,statistics,sys
from pathlib import Path
from collections import defaultdict
stage,suffix,maxrep=sys.argv[1],sys.argv[2],int(sys.argv[3]);dest=Path('reports')/(stage+'-'+suffix)
cases={r['case']:r for r in csv.DictReader(open(sys.argv[4] if len(sys.argv)>4 else 'cases.csv'))}
groups=defaultdict(lambda:defaultdict(list))
for r in csv.DictReader((dest/'records.csv').open()):groups[(r['case'],r['method'])][int(r['rep'])].append(r)
def dataset_metrics(rr):
    return dict(coverage=statistics.mean(float(float(r['lower'])<=float(r['truth'])<=float(r['upper'])) for r in rr),rejection=statistics.mean(float(float(r['pvalue'])<.05) for r in rr),length=statistics.mean(float(r['upper'])-float(r['lower']) for r in rr),signed_z=statistics.mean((float(r['estimate'])-float(r['truth']))/float(r['se'])*(1 if float(r['truth'])>=0 else -1) for r in rr))
def aggregate(vals):
    out={}
    for key in vals[0]:
        v=[x[key] for x in vals];out[key]=statistics.mean(v);out[key+'_mcse']=statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else None
    return out
summary=[];slots=[];invalid=[];counts=[]
for (case,method),reps in sorted(groups.items()):
    c=cases[case];count=int(c.get('target_count') or (6 if c['design']=='belloni' else 7 if c['design']=='sihr' else 10));valid={}
    for rep,rr in reps.items():
        if len(rr)==count and len({r['target'] for r in rr})==count and all(r['converged'].lower()=='true' and float(r['se'])>0 and all(math.isfinite(float(r[k])) for k in ['estimate','se','lower','upper','pvalue']) and 0<=float(r['pvalue'])<=1 for r in rr):valid[rep]=rr
        else:invalid.append([case,method,rep,len(rr)])
    counts.append(dict(case=case,method=method,valid=len(valid),planned=maxrep,missing_or_invalid=maxrep-len(valid)))
    for cls in ['active','null']:
        vals=[dataset_metrics([r for r in rr if (float(r['truth'])!=0)==(cls=='active')]) for rr in valid.values() if any((float(r['truth'])!=0)==(cls=='active') for r in rr)]
        if vals:summary.append(dict(case=case,method=method,target_class=cls,valid_datasets=len(vals),planned_datasets=maxrep,**aggregate(vals)))
    for slot in range(count):
        rr=[rows[slot] for rows in valid.values()]
        if not rr:continue
        truths={r['truth'] for r in rr};columns={r['target'] for r in rr}
        # Every role must have the same true coefficient in this frozen DGP.
        if len(truths)!=1:raise RuntimeError(f'inconsistent target ordering/truth: {case},{method},{slot+1},{truths}')
        slots.append(dict(case=case,method=method,target_slot=slot+1,target_column=next(iter(columns)) if len(columns)==1 else 'varies',truth=next(iter(truths)),valid_datasets=len(rr),planned_datasets=maxrep,**aggregate([dataset_metrics([r]) for r in rr])))
for name,content in [('strict-summary',summary),('per-slot',slots),('strict-counts',counts),('strict-checks',dict(incomplete_or_nonconverged=invalid,target_role_truth_consistent=True,method_case_groups=len(groups)))]: (dest/(name+'.json')).write_text(json.dumps(content,indent=2)+'\n')
lines=['# Complete-dataset inference summaries','','EMPIRICAL. Every requested target must be finite, unique and converged before a dataset contributes. Missing/failed fits count against reliability. Per-slot results use fixed requested target roles; physical columns vary in random-support designs. The older per-target-by-column file mixes roles in those designs and is superseded. All Monte Carlo errors use independent datasets.','','|Case|Method|Class|Valid/planned|Coverage ± MCSE|Rejection ± MCSE|Length|','|---|---|---|---|---|---|---|']
for r in summary:lines.append(f"|{r['case']}|{r['method']}|{r['target_class']}|{r['valid_datasets']}/{r['planned_datasets']}|{r['coverage']:.4f} ± {r['coverage_mcse']}|{r['rejection']:.4f} ± {r['rejection_mcse']}|{r['length']:.4f}|")
(dest/'STRICT_RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(method_case_groups=len(groups),slot_groups=len(slots),incomplete_dataset_groups=len(invalid))))
