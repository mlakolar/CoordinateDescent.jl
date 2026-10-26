import csv,json,math,statistics,collections
from pathlib import Path
N=Path('reports/gaussian-native-full1000');O=Path('reports/seedfix-campaign-gaussian');D=Path('reports/gaussian-core-comparison1000');D.mkdir(exist_ok=True,parents=True)
fits=json.loads((N/'fits.json').read_text());ns=json.loads((N/'summary.json').read_text());os=json.loads((O/'strict-summary.json').read_text());rr=json.loads((N/'resource-records.json').read_text())
assert len(fits)==15000 and len(rr)==15000 and all(f['complete'] and f['interval_available'] for f in fits)
checks=json.loads((N/'checks.json').read_text());(D/'native-original-checks-v1.json').write_text(json.dumps(checks,indent=2)+'\n');checks['planned_fits']=3*5*1000;checks['denominator_correction']='V1 retained old7timing-task multiplier; native5methods require15000planned. No missing methods/fits or changed scientific output.'
(D/'native-checks-corrected.json').write_text(json.dumps(checks,indent=2)+'\n')
oldhash={}
for r in csv.DictReader(Path('reports/campaign-gaussian/diagnostics.csv').open()):
 if r['case'] in ['gaussian-moderate','gaussian-correlated','gaussian-benign']:oldhash[(r['case'],int(r['rep']))]=(r['input_x_sha256'],r['input_y_sha256'])
newhash={}
for r in csv.DictReader((O/'diagnostics.csv').open()):newhash[(r['case'],int(r['rep']))]=(r['input_x_sha256'],r['input_y_sha256'])
assert len(newhash)==3000 and newhash==oldhash
resources=[]
for p in Path('resources').glob('gaussian-core-current1000-*.txt'):
 r=dict(l.split('=',1) for l in p.read_text().splitlines() if '=' in l);assert float(r['process_exit'])==0;resources.append({k:float(v) for k,v in r.items()})
assert len(resources)==96
resource=dict(native_worker_cpu_hours=sum(r['cpu_seconds'] for r in rr)/3600,native_max_worker_rss_gib=max(r['max_rss_kib'] for r in rr)/1024**2,current_own_worker_cpu_hours=sum(r['cpu_seconds'] for r in resources)/3600,current_own_max_worker_rss_gib=max(r['max_rss_kib'] for r in resources)/1024**2,note='Native resources include reused first20/case;failed observer attempts/recoveries accounted separately. Individual-worker RSS,not simultaneous total. Native job13452003 elapsed7708s64CPU128GiB;current-source own13455533 separate.')
(D/'resource-summary.json').write_text(json.dumps(resource,indent=2)+'\n')
summary=[]
for r in os:summary.append(dict(case=r['case'],engine='Julia_current',role=r['target_class'],planned=1000,interval_available=r['valid_datasets'],strict_valid=r['valid_datasets'],**{k:r[k] for k in ['coverage','coverage_mcse','rejection','rejection_mcse','length','length_mcse']}))
summary+=ns;(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(D/'additional-checks.json').write_text(json.dumps(dict(matched_dataset_hashes=3000,own_source='package-hc3-v2/env-hc3-v2;canonicalphysicaltargetseeds',native_source='gaussian-native-full-v1 frozen56c8add',datasets='same92300000inputs;notadditionalindependentdata',planned_methods=6),indent=2)+'\n')
lines=['# Current Gaussian policy versus native alternatives:1000datasets per core design','','EMPIRICAL. Same3000response datasets and ten primary statistical targets per dataset. Julia uses the current canonical-target Gaussian HC3 policy. Native hdi/EconML/sslasso compute all slopes internally;SIHR/hdm request ten targets. Their native statistical recipes differ. This is not matched-penalty engine timing.','','|Design|Method|Finite/strict valid per1000|Active coverage ± MCSE (%)|TypeI ± MCSE (%)|Power (%)|Active length|','|---|---|---:|---:|---:|---:|---:|']
for c in ['gaussian-moderate','gaussian-correlated','gaussian-benign']:
 for e in ['Julia_current','hdi','EconML','sslasso','SIHR','hdm']:
  a=next(r for r in summary if r['case']==c and r['engine']==e and r['role']=='active');n=next(r for r in summary if r['case']==c and r['engine']==e and r['role']=='null')
  lines.append(f"|{c}|{e}|{a['interval_available']}/{a['strict_valid']}|{100*a['coverage']:.2f} ±{100*a['coverage_mcse']:.2f}|{100*n['rejection']:.2f} ±{100*n['rejection_mcse']:.2f}|{100*a['rejection']:.2f}|{a['length']:.4f}|")
lines+=['','Native coverage is conditional on finite returned intervals, including warned/uncertified fits. Strict availability separately rejects exposed convergence warnings and requires native sslassoQPcertificates. Warnings can concern unselectedCVlevels; strict rejection is conservative,not a final-model failure theorem. Native outputs/formulas were not repaired. A failed certificate cannot be ignored because coverage looks favorable. All target-role/dataset records remain retained.','','The first analyzer had a planned-fit denominator typo from an earlier seven-task timing protocol. Native-only five methods x3designs x1000 =15000,all present. The original checks are retained beside corrected checks. No method was dropped.','','Actual native workerCPU '+f"{resource['native_worker_cpu_hours']:.2f}"+'hours and maxindividualworkerRSS '+f"{resource['native_max_worker_rss_gib']:.3f}"+'GiB. Current-source own refreshCPU '+f"{resource['current_own_worker_cpu_hours']:.2f}"+'hours and maxworkerRSS '+f"{resource['current_own_max_worker_rss_gib']:.3f}"+'GiB. Warm all-slope20dataset timings and initialCV costs are reported separately;ten-target own times are not all-slope speedups.','','This completes the native comparison in three core Gaussian designs. Seven stress-design native arms and refreshed matched-statistical-recipe campaigns are not complete. The ten-design disjoint500validation remains the current automatic-policy calibration evidence. No logistic/QPproductiondefault is promoted.','']
(D/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps(resource))
