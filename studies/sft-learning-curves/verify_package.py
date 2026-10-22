"""Verify public package bytes and installed pinned distributions; no model execution."""
import hashlib,json,importlib.metadata
from pathlib import Path
root=Path(__file__).resolve().parent
m=json.loads((root/'MANIFEST.json').read_text())
for name,expected in m.items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise SystemExit('Hash mismatch: '+name)
errors=[]
for line in (root/'bounded/source/vendor/requirements.lock.txt').read_text().splitlines():
 name,version=line.split('==')
 try:actual=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:actual='not installed'
 if actual!=version:errors.append(f'{name}: expected {version}, found {actual}')
if errors:raise SystemExit('\n'.join(errors))
print(f'Verified {len(m)} package hashes and every pinned distribution version.')
