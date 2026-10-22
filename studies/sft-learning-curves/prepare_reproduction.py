"""Prepare a distinct execution copy without loading a model or running training."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('study',choices=['bounded','closure']);p.add_argument('model',type=Path);p.add_argument('destination',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parent/a.study/'source';model=a.model.resolve()
manifest=json.loads((root/'vendor/model_manifest.json').read_text())
for name,info in manifest['files'].items():
 b=(model/name).read_bytes()
 if len(b)!=info['bytes'] or hashlib.sha256(b).hexdigest()!=info['sha256']:raise SystemExit('Model mismatch: '+name)
if a.destination.exists():raise SystemExit('Destination must not exist')
shutil.copytree(root,a.destination)
cfg=json.loads((a.destination/'protocol.json').read_text());cfg['model_dir']=str(model)
(a.destination/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
files={str(f.relative_to(a.destination)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.destination.rglob('*')) if f.is_file() and f.name!='FREEZE.json' and '__pycache__' not in f.parts}
(a.destination/'FREEZE.json').write_text(json.dumps(files,indent=2)+'\n')
print('Prepared source copy only. No training or inference was run.')
