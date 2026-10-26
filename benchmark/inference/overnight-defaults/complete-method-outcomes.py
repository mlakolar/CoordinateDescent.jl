"""Explicit native-study denominator grid, including methods with zero outputs."""
import csv,json,sys
from pathlib import Path
stage,maxrep=sys.argv[1],int(sys.argv[2]);dest=Path('reports')/(stage+'-hc1')
counts={(r['case'],r['method']):r for r in json.loads((dest/'strict-counts.json').read_text())};out=[]
for c in csv.DictReader(open('cases.csv')):
 gaussian=c['family']=='gaussian';lowdim=int(c['n'])>int(c['p'])+5
 base=['rldpe','rldpe_refit']+(['qp','qp_refit','qp_half','qp_half_refit'] if gaussian else [])
 methods=base+([m+'_hc1' for m in base] if gaussian else [])+['hdm_post','hdm_no_post','cvds_min','cvds_1se','SIHR']
 if gaussian:methods+=['hdm_ds','hdm_ds_single']
 else:methods+=['SIHR_alter']
 if lowdim:methods+=['full_mle']+([] if gaussian else ['xia_refined','xia_refined_1se'])
 for method in methods:
  r=counts.get((c['case'],method),dict(case=c['case'],method=method,valid=0,planned=maxrep,missing_or_invalid=maxrep))
  out.append(dict(**r,appears_in_successful_records=(c['case'],method) in counts))
(dest/'complete-method-outcomes.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# All planned native-study methods','','Zero-output methods remain in this denominator grid. Coverage is undefined when no complete dataset succeeds; it must not be reported as zero or silently omitted. Raw errors and warnings remain separate receipts.','','|Case|Method|Complete/planned|','|---|---|---|']
for r in out:lines.append(f"|{r['case']}|{r['method']}|{r['valid']}/{r['planned']}|")
(dest/'ALL_METHOD_OUTCOMES.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(planned_method_case_groups=len(out),zero_success_groups=[r for r in out if r['valid']==0])))
