"""Offline SFT-only development curve. Invoke via supervise.py, never directly."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, json, random, sys, time, platform, math
from contextlib import contextmanager
from pathlib import Path
from evidence import atomic, append, digest, check_coverage, summarize, verify_freeze
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import data



@contextmanager
def preserve_evaluation_state(model):
    import torch
    cpu_rng=torch.get_rng_state().clone(); python_rng=random.getstate()
    modes=[(module,module.training) for module in model.modules()]
    try: yield
    finally:
        torch.set_rng_state(cpu_rng); random.setstate(python_rng)
        for module,mode in modes: module.training=mode


def compare_original(checkpoint,out):
    import torch
    from safetensors.torch import load_file
    current=load_file(str(checkpoint/'adapter_model.safetensors'))
    original=load_file(str(ROOT/'vendor/original-sft24.safetensors'))
    same_keys=set(current)==set(original)
    comparisons={key:{'equal':bool(torch.equal(current[key],original[key])),
        'max_abs_difference':float((current[key]-original[key]).abs().max())}
        for key in sorted(set(current)&set(original))}
    atomic(out/'step24-original-comparison.json',{'same_keys':same_keys,
        'all_tensors_identical':same_keys and all(v['equal'] for v in comparisons.values()),
        'tensor_comparisons':comparisons,'original_sha256':digest(ROOT/'vendor/original-sft24.safetensors'),
        'interpretation':'Tensor control only; any mismatch requires investigation before claiming exact pilot continuation. No test scoring.'})


def execute(out):
    import torch, transformers, peft, safetensors
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import training
    cfg=json.loads((ROOT/'protocol.json').read_text())
    frozen=verify_freeze(ROOT)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(cfg['seed']); torch.use_deterministic_algorithms(True)
    manifest=json.loads((ROOT/'vendor/model_manifest.json').read_text())
    model_dir=Path(cfg['model_dir'])
    for name,info in manifest['files'].items():
        if digest(model_dir/name)!=info['sha256']: raise ValueError('Model hash mismatch: '+name)
    # Test records are deliberately excluded from the saved case definitions and all evaluation.
    generated=data.build(cfg['data_seed']); splits={k:generated[k] for k in ('train','validation')}
    if splits!=json.loads((ROOT/'cases-development.json').read_text()): raise ValueError('Frozen case mismatch')
    atomic(out/'cases.json',splits)
    atomic(out/'identity.json',{'freeze':frozen,'model':manifest,'python':platform.python_version(),
        'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
        'safetensors':safetensors.__version__,'scope':'development validation only; no test evaluation'})
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    train=[training.encode(tokenizer,data.examples(g)) for g in splits['train']]
    validation=[training.encode(tokenizer,data.examples(g)) for g in splits['validation']]
    model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,
        use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager')
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=cfg['rank'],lora_alpha=cfg['alpha'],lora_dropout=cfg['dropout'],
        target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',bias='none'))
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    order=list(range(len(train))); random.Random(cfg['seed']).shuffle(order)
    atomic(out/'order.json',{'record_indices':order,'trainable_parameters':sum(p.numel() for p in params)})
    for step in range(cfg['max_updates']+1):
        if step:
            wall=time.monotonic(); cpu=time.process_time()
            model.train(); optimizer.zero_grad(set_to_none=True)
            batch=train[order[(step-1)%len(order)]]
            loss,parts=training.loss(model,None,batch,0.,0.)
            value=float(loss.detach())
            if not math.isfinite(value):
                append(out/'training.jsonl',{'step':step,'status':'nonfinite','loss_repr':repr(value)})
                raise ValueError('Nonfinite loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(params,cfg['clip_norm'],error_if_nonfinite=True)
            optimizer.step()
            append(out/'training.jsonl',{'step':step,'group_id':splits['train'][order[(step-1)%len(order)]]['id'],
                'lm_loss':value,'gradient_norm':float(norm),'wall_seconds':time.monotonic()-wall,'cpu_seconds':time.process_time()-cpu})
        if step in cfg['checkpoints']:
            checkpoint=out/f'checkpoint-{step:04d}'; checkpoint.mkdir()
            model.save_pretrained(checkpoint,safe_serialization=True)
            for file in checkpoint.iterdir():
                if file.is_file():
                    with file.open('rb') as handle: os.fsync(handle.fileno())
            atomic(checkpoint/'adapter_manifest.json',{p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()})
            if step==24: compare_original(checkpoint,out)
            with preserve_evaluation_state(model):
                evaluate(model,tokenizer,validation,splits['validation'],out,step,cfg)
            append(out/'exposure.jsonl',{'step':step,'source_record_presentations':step,
                'record_register_presentations':3*step,'unique_source_records':min(step,64),
                'unique_record_register_pairs':3*min(step,64),'record_pass_equivalents':step/64,
                'record_register_pass_equivalents':3*step/192})
    atomic(out/'status.json',{'status':'complete','checkpoints':cfg['checkpoints'],'no_test_scoring':True})


def evaluate(model,tokenizer,batches,groups,out,step,cfg):
    import torch
    model.eval(); start=time.monotonic(); cpu=time.process_time(); total=0.; count=0
    with torch.no_grad():
        for group,b in zip(groups,batches):
            result=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],labels=b['labels'],use_cache=False)
            n=int((b['labels'][:,1:]!=-100).sum()); score=float(result.loss)
            append(out/'validation_lm.jsonl',{'step':step,'group_id':group['id'],'lm_loss':score,'target_tokens':n})
            total+=score*n; count+=n
        lm_wall=time.monotonic()-start; lm_cpu=time.process_time()-cpu
        start=time.monotonic(); cpu=time.process_time(); rows=[]
        for group in groups:
            for e in data.examples(group):
                ids=tokenizer.apply_chat_template(e['messages'],tokenize=True,add_generation_prompt=True,return_tensors='pt')
                output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=cfg['max_new_tokens'],
                    do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id,temperature=None,top_p=None)
                tokens=output[0,ids.shape[1]:].tolist()
                text=tokenizer.decode(tokens,skip_special_tokens=True)
                row={'step':step,'group_id':e['group_id'],'register':data.REGISTERS[e['register']],
                    'answer':e['answer'],'target':e['target'],'messages':e['messages'],'input_ids':ids[0].tolist(),
                    'output_token_ids':tokens,'generated':text,'prediction':data.parse_answer(text.strip()),
                    'hit_token_cap':len(tokens)==cfg['max_new_tokens'],'ended_eos':bool(tokens and tokens[-1]==tokenizer.eos_token_id)}
                append(out/'generations.jsonl',row); rows.append(row)
        # Coverage is checked only after all available evidence is already durable.
        expected=[(g['id'],r) for g in groups for r in data.REGISTERS]
        check_coverage(rows,expected)
        append(out/'curves.jsonl',{'step':step,'validation_lm_loss':total/count,
            'lm_wall_seconds':lm_wall,'lm_cpu_seconds':lm_cpu,'generation_wall_seconds':time.monotonic()-start,
            'generation_cpu_seconds':time.process_time()-cpu,**summarize(rows)})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if os.environ.get('ADAPTIVE_SUPERVISED')!='1': raise SystemExit('Use supervise.py to enforce resource caps')
    try: execute(a.out)
    except BaseException as exc:
        atomic(a.out/'status.json',{'status':'failed','error':repr(exc)})
        raise
