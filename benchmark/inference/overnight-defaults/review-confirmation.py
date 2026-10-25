"""Prespecified target-role review on saved fresh validation, on compute only."""
import json,math
from pathlib import Path
p=Path('reports/confirmation-frozen');roles=json.loads((p/'per-slot.json').read_text());out=[]
for r in roles:
 active=float(r['truth'])!=0
 row=dict(case=r['case'],slot=r['target_slot'],truth=float(r['truth']),coverage=r['coverage'],coverage_mcse=r['coverage_mcse'],lower95=r['coverage']-1.959963984540054*r['coverage_mcse'],rejection=r['rejection'],rejection_mcse=r['rejection_mcse'],upper95_rejection=r['rejection']+1.959963984540054*r['rejection_mcse'],length=r['length'])
 out.append(row)
flags=[r for r in out if (r['truth']!=0 and r['lower95']<.925) or (r['truth']==0 and r['upper95_rejection']>.075)]
summary=dict(active_coverage_range=[min(r['coverage'] for r in out if r['truth']!=0),max(r['coverage'] for r in out if r['truth']!=0)],null_rejection_range=[min(r['rejection'] for r in out if r['truth']==0),max(r['rejection'] for r in out if r['truth']==0)],role_flags=flags,note='Role review uses unadjusted marginal normal MC intervals; exploratory diagnostics, no multiplicity-adjusted simultaneous claim. Pooled prospective gate is separate. No retuning on these data.')
(p/'role-review.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Fresh confirmation: requested roles','','|Case|Role|Truth|Coverage ± MCSE|Power/type I ± MCSE|Length|','|---|---|---|---|---|---|']
for r in out:lines.append(f"|{r['case']}|{r['slot']}|{r['truth']}|{r['coverage']:.4f} ± {r['coverage_mcse']:.4f}|{r['rejection']:.4f} ± {r['rejection_mcse']:.4f}|{r['length']:.4f}|")
(p/'ROLE_REVIEW.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary))
