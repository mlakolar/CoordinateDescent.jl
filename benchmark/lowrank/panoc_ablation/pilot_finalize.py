import pathlib,json,tomllib,csv,statistics,tarfile,re
root=pathlib.Path('.');out=root/'results/pilot-final-report-v1';out.mkdir(exist_ok=True)
s=json.loads((root/'results/pilot-cached-recovery-v3/summary.json').read_text());table=s['table'];lookup={(r['case'],r['config'],r['mode']):r for r in table}
paths=[];counts=[];profiles=[]
for case in ['small','medium','responses','wide','ill','deficient']:
 f=root/'results/pilot-v1'/case
 for config in ['native','apg','combined']:
  rs=list(csv.DictReader((f/(config+'-path-timings.csv')).open()));assert len(rs)==3
  paths.append(dict(case=case,config=config,seconds=statistics.median(float(r['seconds']) for r in rs),allocated=statistics.median(int(r['bytes']) for r in rs)))
 for f2 in sorted(f.glob('*-counts.toml')):counts.append(dict(case=case,config=f2.name.removesuffix('-counts.toml'),**tomllib.loads(f2.read_text())))
 for config in sorted({r['config'] for r in table}):
  original=f/(config+'-cpu.txt');corrected=root/'results/pilot-public-recovery-v2'/case/config/'cpu.txt'
  path=corrected if config in ['native','apg'] else original
  content=path.read_text();words=int(re.search(r'raw_stack_words=(\d+)',content).group(1));profiles.append(dict(case=case,config=config,source=str(path),stack_words=words))
ratio=[]
for case in ['small','medium','responses','wide','ill','deficient']:
 for config in sorted({r['config'] for r in table}):
  for mode in ['public','cached']:
   r=lookup[case,config,mode];b=lookup[case,'native',mode];c=lookup[case,'pb95',mode]
   ratio.append(dict(case=case,config=config,mode=mode,seconds=r['seconds'],ratio_native=r['seconds']/b['seconds'],ratio_pb95=r['seconds']/c['seconds'],allocated=r['bytes']))
for name,rows in [('ratios',ratio),('paths',paths),('counts',counts),('profiles',profiles)]:
 with (out/(name+'.csv')).open('w') as f:
  fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)
report=dict(phase='calibration_only',inputs_per_case=1,certificates_reused=336,new_certificates=0,certificate_failures=s['certificate_failures'],paths=paths,ratios=ratio,counts=counts,profiles=profiles,empty_profiles=[r for r in profiles if not r['stack_words']],timing_corrections='12 public native/APG controls duplicated setup;96 cached labels called public due named closure overwrite; all108 corrected with byte-equal B;84 other public controls and all paths reused',cold_fraction=.1,path_fractions='8geometriclevels.5to.02',source_baseline='295fbf5420d6384a1ca261023367ee86d0a40b8c')
(out/'summary.json').write_text(json.dumps(report,indent=2))
lines=['# Nuclear PANOC controlled calibration','', 'One independent new dataset per design, rep0 excluded from a later20-input holdout. Three warmed rounds per workload, fixed EPYC7513/single-thread stack. All336 original SciPy primal-dual gap and proximal-gradient certificates pass1e-6; corrected timing outputs are byte-identical to those certified arrays. No production optimization or solver-default change has been adopted.','','Initial pilot13450595 completed3m00s. Two harness defects were found before drawing speed conclusions: native/APG public controls prepared data twice (12 affected controls), and repeated local named fit definitions made cached labels call the public closure (96 controls). Targeted corrections13450643(39s) and13450661(42s) reran only those108 controls on the same inputs. Preserve initial/v2/v3 tables, and do not count corrections as new inputs/certificates. The84 other public controls and all paths remain valid.','','| Case | Native public ms | Controlled PB95/native | PO95/PB95 | Gram/PB95 | Shared-residual/PB95 | Final-reuse/PB95 | Combined/native |','|---|---:|---:|---:|---:|---:|---:|---:|']
for case in ['small','medium','responses','wide','ill','deficient']:
 rs={(r['config'],r['mode']):r for r in ratio if r['case']==case}
 values=[case,f"{rs['native','public']['seconds']*1000:.4f}",f"{rs['pb95','public']['ratio_native']:.3f}",f"{rs['po95','public']['ratio_pb95']:.3f}",f"{rs['gram','public']['ratio_pb95']:.3f}",f"{rs['shared','public']['ratio_pb95']:.3f}",f"{rs['finalreuse','public']['ratio_pb95']:.3f}",f"{rs['combined','public']['ratio_native']:.3f}"]
 lines.append('| '+' | '.join(values)+' |')
lines+=['','Ratios from one input are diagnostic signals, not repeatable improvement evidence. The compiled controlled PB95 wrapper can differ from the native extension despite equivalent mathematics/outputs; use controlled one-factor ratios for attribution and actual before/after production APIs for adoption. The many-response native/control difference remains visible rather than being assigned entirely to prox/diagnostics. Cached controls now explicitly exclude centering/gradient setup, with an allocation witness and exact B equality.','','The Gram/shared-residual proposal merits a guarded production prototype and20-input test. A quadratic Gram loss value can cancel near a tiny residual loss, so the calibration expression alone is insufficient for adoption; examine stable direct-residual fallback and Float32/64 near-fit behavior. Stable shared residual work helps wide designs but is adverse on tall designs, so a universal residual-gradient switch is unsupported. Final residual/singular-value reuse gives smaller effects; preserve rank threshold/support/intercept/DF/objective semantics. Quadratic-trait call counts, cadence sensitivity, unsafe boundary-step controls and all profiles are retained. Keep certificate tolerance1e-8/cadence10/standard0.95/L and APG default until broader evidence justifies a change.','','Every8-level calibration path point is independently certified, including paths with nonuniform timing results. Complete20-input cold strong/middle/weak and path evidence, independent certificates, regression controls, all-target fresh-process memory/profiles, and optional-extension/Float32/64/zero/iteration-limit/CV tests remain required by issue103. No inference or universal solver ranking is established.','',f"Available corrected public CPU profiles: {len(profiles)}; empty: {len(report['empty_profiles'])}. Short calibration profiles can be empty; future20-input target profiles must use bounded repeated sampling outside timers. Raw coefficient matrices and binary inputs remain on Discovery. This reporting delivery contains original/corrected tables, certificates, call counts, profiles, logs and frozen manifests."]
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
with tarfile.open('delivery-pilot-report-v1.tar.gz','w:gz') as tar:
 for folder in ['results/pilot-v1','results/pilot-public-recovery-v2','results/pilot-cached-recovery-v3','results/pilot-final-report-v1']:
  for f in sorted(pathlib.Path(folder).rglob('*')):
   if f.is_file() and f.suffix!='.bin':tar.add(f,arcname=str(f))
 for f in pathlib.Path('inputs/pilot-v1').glob('*/meta.json'):tar.add(f,arcname=str(f))
 for name in ['Project.toml','Manifest.toml']:tar.add(pathlib.Path('env-baseline-1.13')/name,arcname='provenance/'+name)
print(json.dumps(dict(report=str(out/'RESULTS.md'),reused_certificates=336,certificate_failures=s['certificate_failures'],empty_profiles=report['empty_profiles'],paths=paths),indent=2))
