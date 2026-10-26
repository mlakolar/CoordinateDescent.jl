import pathlib,csv,json,tomllib,statistics
root=pathlib.Path('results');primary=root/'pilot-v1';recovery=root/'pilot-public-recovery-v2'
s=json.loads((primary/'summary.json').read_text());table=s['table'];new=[]
for case in ['small','medium','responses','wide','ill','deficient']:
 for config in ['native','apg']:
  f=recovery/case/config;assert json.loads((f/'resource.json').read_text())['exit_code']==0
  assert tomllib.loads((f/'metadata.toml').read_text())['matched_original_B']
  ts=list(csv.DictReader((f/'timings.csv').open()));assert len(ts)==3
  row=dict(case=case,config=config,mode='public',seconds=statistics.median(float(r['seconds']) for r in ts),bytes=statistics.median(int(r['bytes']) for r in ts),source='public-recovery-v2')
  table=[r for r in table if (r['case'],r['config'],r['mode'])!=(case,config,'public')];table.append(row);new.append(row)
lookup={(r['case'],r['config'],r['mode']):r for r in table};ratios=[]
for case in ['small','medium','responses','wide','ill','deficient']:
 for config in sorted({r['config'] for r in table}):
  r=lookup[case,config,'public'];b=lookup[case,'native','public'];c=lookup[case,'pb95','public']
  ratios.append(dict(case=case,config=config,seconds=r['seconds'],ratio_native=r['seconds']/b['seconds'],ratio_controlled_pb95=r['seconds']/c['seconds'],allocated=r['bytes']))
report=dict(phase='calibration_only',independent_inputs_per_case=1,reused_certificates=len(s['certificates']),certificate_failures=sum(not r['passed'] for r in s['certificates']),public_recovery=new,table=table,ratios=ratios,retained_original='pilot-v1/summary.json;native/APG public setup was duplicated, invalid as public-API timing;cached/controlled/path results reused unchanged')
(recovery/'summary.json').write_text(json.dumps(report,indent=2))
for label,rows in [('corrected-table',table),('ratios',ratios)]:
 with (recovery/(label+'.csv')).open('w') as f:
  fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)
print(json.dumps(dict(corrected_public=len(new),reused_certificates=report['reused_certificates'],certificate_failures=report['certificate_failures'],target_ratios=[r for r in ratios if r['config'] in ['combined','gram','shared','finalreuse','quadtrait','po95']]),indent=2))
