"""Post-run tooling: fetch only manifest-pinned public data; never execute model code."""
import argparse,hashlib,json,os,re,tempfile,urllib.request
from pathlib import Path


def validate(manifest):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',manifest['model_id']):
        raise ValueError('Expected a public owner/model identifier')
    if not re.fullmatch(r'[a-f0-9]{40}',manifest['revision']):raise ValueError('Immutable commit required')
    for name,info in manifest['files'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+\.(json|txt|md|safetensors)',name):raise ValueError('Unsafe data filename')
        if not re.fullmatch(r'[a-f0-9]{64}',info['sha256']):raise ValueError('Invalid SHA256')
        if type(info['bytes']) is not int or info['bytes']<0:raise ValueError('Invalid byte count')


def matches(path,info):
    if path.is_symlink() or not path.is_file() or path.stat().st_size!=info['bytes']:return False
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()==info['sha256']


def public_open(url):
    # No token, account, cookies or remote-code loader. Standard HTTPS certificate validation.
    return urllib.request.urlopen(url,timeout=30)


def fetch(manifest,destination,verify_only=False,opener=public_open):
    validate(manifest);destination=Path(destination)
    if destination.is_symlink():raise ValueError('Symlink destination refused')
    if verify_only and not destination.exists():raise FileNotFoundError(destination)
    if not verify_only:destination.mkdir(parents=True,exist_ok=True)
    for name,info in manifest['files'].items():
        target=destination/name
        if target.exists() or target.is_symlink():
            if not matches(target,info):raise ValueError('Existing file mismatch: '+name)
            continue
        if verify_only:raise FileNotFoundError(target)
        url=f"https://huggingface.co/{manifest['model_id']}/resolve/{manifest['revision']}/{name}"
        fd,tmp=tempfile.mkstemp(prefix='.fetch-',dir=destination)
        try:
            with os.fdopen(fd,'wb') as output,opener(url) as response:
                count=0
                while block:=response.read(1024*1024):
                    count+=len(block)
                    if count>info['bytes']:raise ValueError('Download exceeds manifest size: '+name)
                    output.write(block)
            if not matches(Path(tmp),info):raise ValueError('Downloaded file mismatch: '+name)
            # Atomic publication without replacing a concurrently created file.
            os.link(tmp,target)
        finally:
            Path(tmp).unlink(missing_ok=True)
    return len(manifest['files'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    manifest=json.loads(Path(__file__).with_name('model_manifest.json').read_text())
    print('Verified',fetch(manifest,a.destination,a.verify_only),'manifest-pinned files')
