"""Post-run aggregate tests. NumPy-only by default; pretrained tests are explicit."""
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def suites(pretrained):
    selected=[('tests','src'),('tooling/tests','pretrained:tooling')]
    if pretrained:selected.append(('pretrained/tests','pretrained'))
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pretrained',action='store_true')
    p.add_argument('--model-dir',type=Path);a=p.parse_args()
    if a.pretrained and not a.model_dir:p.error('--pretrained requires --model-dir; see pretrained/README.md for optional dependencies and model fetch')
    env=dict(os.environ);env.setdefault('OPENBLAS_NUM_THREADS','1')
    if a.model_dir:env['MODEL_DIR']=str(a.model_dir.resolve())
    if a.pretrained:
        import importlib.util
        missing=[m for m in ['torch','transformers','peft','safetensors'] if importlib.util.find_spec(m) is None]
        if missing:p.error('Missing optional dependencies: '+', '.join(missing)+'; see pretrained/README.md')
        sys.path.insert(0,str(ROOT/'pretrained'));import fetch_model,json
        fetch_model.fetch(json.loads((ROOT/'pretrained/model_manifest.json').read_text()),env['MODEL_DIR'],verify_only=True)
    for folder,paths in suites(a.pretrained):
        env['PYTHONPATH']=os.pathsep.join(str(ROOT/q) for q in paths.split(':'))
        subprocess.run([sys.executable,'-m','unittest','discover','-s',folder,'-v'],cwd=ROOT,env=env,check=True)
    if not a.pretrained:print('Pretrained suite not requested. Use --pretrained --model-dir PATH for all suites.')


if __name__=='__main__':main()
