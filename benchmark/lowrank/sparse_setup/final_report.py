import csv,json,pathlib,statistics,sys
root=pathlib.Path(sys.argv[1]);main=root/'results'/'main';follow=root/'results'/'main-r-tol-v1'
a=json.loads((main/'summary.json').read_text());b=json.loads((follow/'summary.json').read_text());v=json.loads((follow/'validation.json').read_text());post=json.loads((main/'poststudy.json').read_text())
r=[x for x in v['certificates'] if x['variant'].startswith('r-')];assert len(r)==760 and all(x['passed'] for x in r)
rm=[x for x in b['resources'] if x['experiment'].startswith('r-')];assert len(rm)==240
for x in rm:
 md=x['metadata'];assert '7513' in md['cpu'] and md['rrpack']=='0.1.14' and all(md[k]=='1' for k in ['openblas_threads','omp_threads','mkl_threads'])
wide=[x for x in v['pairs'] if x['case']=='wide' and x['config']=='cd-0.5-unit' and x['comparator']=='r-public']
assert len(wide)==20
sensitivity=[]
for case in ['small','medium','responses','wide','ill','deficient']:
 for mode in ['public','common']:
  # The same Julia numerator cancels: ratio(strict Julia/R)/ratio(loose Julia/R).
  for tag in sorted({x['config'] for x in b['table'] if x['case']==case and x['mode']==mode and x['comparator']=='r-'+mode}):
   ra=next(x for x in a['table'] if x['case']==case and x['mode']==mode and x['config']==tag and x['comparator']=='r-'+mode)
   rb=next(x for x in b['table'] if x['case']==case and x['mode']==mode and x['config']==tag and x['comparator']=='r-'+mode)
   sensitivity.append(dict(case=case,mode=mode,config=tag,strict_R_ms=ra['comparator_ms'],historical_R_ms=rb['comparator_ms'],strict_matched=ra['inputs'],historical_matched=rb['inputs']))
result=dict(new_R_processes=240,new_R_certificates=760,max_R_kkt=max(x['kkt'] for x in r),max_R_response=max(x['response'] for x in r),
 R_setting_decision='Use global historical R outer1e-8/inner1e-9 for the main alternative ranking: all760newfitcertificatespassedunchanged1e-6thresholds;strictR1e-10/1e-12isretainedassensitivityandnotusedtoinflateprimaryspeedclaims.',
 strong_wide_R_unmatched=sum(not x['matched_objective'] for x in wide),strong_wide_R_max_objective_difference=max(x['objective_relative_difference'] for x in wide),
 primary_metadata=post['verified_process_metadata'],native_profiles=post['native_cpu_profiles'],R_profiles_primary=post['R_cpu_memory_profiles'],R_profiles_followup=sum(x['experiment'].startswith('r-') for x in b['profiles']),
 source_and_Julia_results_unchanged=True,sensitivity=sensitivity)
(follow/'final-report.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:x for k,x in result.items() if k!='sensitivity'},indent=2))
