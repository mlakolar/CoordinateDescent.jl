"""Select reporting inputs for transfer; generated matrices remain on Discovery."""
import glob
import hashlib
import json
import os
import tarfile
from pathlib import Path

root=Path(os.environ['RUN'])
patterns=['timing-*.csv','mc-*.csv','accuracy-*.csv','validity-*.csv','fixed-validity-*.csv',
          'cpss-agreement-*.csv','profile-*.txt','profile-*.csv','environment-*.txt','R-session-*.txt',
          '*summary.csv','paired-*.csv','selection-agreement.csv','constructor-validity.csv',
          'summary.md','validation.json','gaussian-*.csv','gaussian-validation.json']
files={p:'results/'+p.name for pattern in patterns for p in (root/'results').glob(pattern)}
for name in ['Project.toml','Manifest.toml']:
    files[root/'env'/name]='metadata/'+name
for pattern in ['*-joblog.tsv','*-*.out','setup.log','accounting.tsv','benchmark-*.log','mc-*.log','gaussian-*.log','generate.log']:
    for p in root.glob(pattern): files[p]='metadata/'+p.name
for folder in ['pilot','pilot-v3','pilot-v4']:
    if (root/folder/'worker.log').exists(): files[root/folder/'worker.log']='pilots/'+folder+'/worker.log'
    for pattern in ['timing-*.csv','accuracy-*.csv','validity-*.csv','constructor-validity.csv','validation.json']:
        for p in (root/folder/'results').glob(pattern): files[p]='pilots/'+folder+'/'+p.name
for p in (root/'depot'/'packages'/'Knockoffs').glob('*/src/modelX.jl'):
    files[p]='metadata/Knockoffs-modelX-installed.jl'
for p in (root/'depot'/'packages'/'Knockoffs').glob('*/LICENSE'):
    files[p]='metadata/Knockoffs-LICENSE'
old_manifest=Path('/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/source/benchmark/selection/Manifest.toml')
if old_manifest.is_file(): files[old_manifest]='metadata/baseline-dependency-Manifest.toml'
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.tar.gz') if p.name!='evidence.tar.gz'}
(root/'source-snapshots.json').write_text(json.dumps(hashes,indent=2)+'\n')
files[root/'source-snapshots.json']='metadata/source-snapshots.json'
with tarfile.open(root/'evidence.tar.gz','w:gz') as tar:
    for p,name in sorted(files.items(),key=lambda x:x[1]):
        if p.is_file(): tar.add(p,arcname=name,recursive=False)
print(json.dumps({'files':len(files),'bytes':(root/'evidence.tar.gz').stat().st_size}))
