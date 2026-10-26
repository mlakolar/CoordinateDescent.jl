import csv,json,runpy,sys
from pathlib import Path
stage,n=sys.argv[1],int(sys.argv[2]);cases=[c['case'] for c in csv.DictReader(open('../gaussian-hc3-v1/confirmation-cases.csv')) if c['family']=='gaussian']
rows=[];errors=[];missing=[];warnings=[]
for c in cases:
 if stage=='canary' and c not in ['gaussian-moderate','gaussian-correlated']:continue
 root=Path('outputs')/stage/c
 for rep in range(1,n+1):
  if not (root/(str(rep)+'.qphc3.complete')).exists():missing.append([c,rep])
  p=root/(str(rep)+'.qphc3.csv')
  if p.exists():rows.extend(dict(r,case=c,rep=rep) for r in csv.DictReader(p.open()))
 for p in root.glob('*.qphc3.error'):errors.append(dict(case=c,file=p.name,error=p.read_text()))
 for p in root.glob('*.qphc3.search-warning'):warnings.append(dict(case=c,file=p.name,message=p.read_text()))
D=Path('reports')/(stage+'-qphc3');D.mkdir(exist_ok=True,parents=True)
if rows:
 with (D/'records.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for name,obj in [('checks',dict(records=len(rows),unique_records=len({(r['case'],r['rep'],r['method'],r['target']) for r in rows}),missing=missing,errors=len(errors),search_warnings=len(warnings))),('errors',errors)]: (D/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
assert not missing and rows
sys.argv=['summarize-slots.py',stage,'qphc3',str(n),'../gaussian-hc3-v1/confirmation-cases.csv'];runpy.run_path('../gaussian-campaign-v1/summarize-slots.py',run_name='__main__')
if stage=='canary':assert not errors and len(rows)==80

if stage=='qphc3-confirmation':
 summary=json.loads((D/'strict-summary.json').read_text());counts=json.loads((D/'strict-counts.json').read_text());gates=[]
 for c in cases:
  method='partialled_adaptive_cvmin_score_hc3'
  a=next(r for r in summary if r['case']==c and r['method']==method and r['target_class']=='active');z=next(r for r in summary if r['case']==c and r['method']==method and r['target_class']=='null');ct=next(r for r in counts if r['case']==c and r['method']==method)
  g=dict(case=c,valid=ct['valid'],planned=n,active_lower95_mc=a['coverage']-1.959963984540054*a['coverage_mcse'],null_upper95_mc=z['rejection']+1.959963984540054*z['rejection_mcse'])
  g['passed']=bool(ct['valid']/n>=.99 and g['active_lower95_mc']>=.925 and g['null_upper95_mc']<=.075);gates.append(g)
 (D/'prospective-gates.json').write_text(json.dumps(gates,indent=2)+'\n')
 print('Primary pooled gates:',sum(g['passed'] for g in gates),'/10')
