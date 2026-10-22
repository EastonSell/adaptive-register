"""Reload saved local adapters and compare all held-out greedy generations."""
import argparse,gc,json,time
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
import pilot


def replay(model_dir,run_dir,adapter_dir,out):
    model_dir,run_dir,adapter_dir,out=map(Path,(model_dir,run_dir,adapter_dir,out))
    # Reserve the receipt before loading; never overwrite an earlier replay.
    with out.open('x') as f:json.dump({'status':'started'},f)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    manifest=pilot.verify_model(model_dir)
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    groups=json.loads((run_dir/'splits.json').read_text())['test']
    result={'status':'running','scope':'saved-adapter reload and all 48 held-out generations per trained arm; not retraining',
            'model_revision':manifest['revision'],'torch':torch.__version__,'arms':{}}
    for arm in ('sft','auxiliary','auxiliary_consistency'):
        folder=adapter_dir/arm
        if not (folder/'adapter_model.safetensors').is_file():raise ValueError('Local safetensors adapter required')
        base=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,
            use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager')
        model=PeftModel.from_pretrained(base,folder,local_files_only=True,is_trainable=False)
        rows=pilot.evaluate(model,tokenizer,groups,time.monotonic()+600)
        expected=json.loads((run_dir/arm/'generations.json').read_text())
        result['arms'][arm]={'examples':len(rows),'exact_row_matches':sum(a==b for a,b in zip(rows,expected)),
                             'all_rows_identical':rows==expected,'adapter_sha256':pilot.digest(folder/'adapter_model.safetensors'),
                             'metrics':pilot.summarize(rows)}
        out.write_text(json.dumps(result,indent=2)+'\n')
        print(arm,result['arms'][arm]['all_rows_identical'],flush=True)
        del model,base;gc.collect()
    result['status']='passed' if all(v['all_rows_identical'] for v in result['arms'].values()) else 'mismatch'
    out.write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='passed':raise RuntimeError('Saved adapter generation mismatch')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--adapter-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    replay(a.model_dir,a.run_dir,a.adapter_dir,a.out)
