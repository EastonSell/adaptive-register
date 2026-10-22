"""Bounded CPU LoRA pilot. Public pinned local safetensors; no network at run time."""
import argparse,copy,gc,hashlib,json,platform,random,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import LoraConfig,get_peft_model
from safetensors.torch import save_file
import data,training

ARMS={'base':(0.,0.),'sft':(0.,0.),'auxiliary':(.1,0.),'auxiliary_consistency':(.1,1.)}
ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()

def verify_model(folder):
    manifest=json.loads((ROOT/'model_manifest.json').read_text())
    for name,info in manifest['files'].items():
        if digest(folder/name)!=info['sha256']:raise ValueError(f'Model file hash mismatch: {name}')
    return manifest

def summarize(rows):
    groups={}
    for row in rows:groups.setdefault(row['group_id'],[]).append(row)
    valid=[g for g in groups.values() if all(r['prediction'] is not None for r in g)]
    return {'answer_accuracy':sum(r['prediction']==r['answer'] for r in rows)/len(rows),
            'answer_parse_coverage':sum(r['prediction'] is not None for r in rows)/len(rows),
            'exact_target_template_match':sum(r['exact_template_match'] for r in rows)/len(rows),
            'groups_with_invalid_generation':len(groups)-len(valid),
            'all_registers_correct':sum(all(r['prediction']==r['answer'] for r in g) for g in groups.values())/len(groups),
            'valid_group_answer_change_rate':sum(len({r['prediction'] for r in g})>1 for g in valid)/len(valid) if valid else None,
            'answer_or_invalid_status_change_rate':sum(len({r['prediction'] for r in g})>1 for g in groups.values())/len(groups),
            'valid_groups':len(valid),'examples':len(rows),'max_new_tokens':24}

def evaluate(model,tokenizer,groups,deadline):
    model.eval();rows=[]
    with torch.no_grad():
        for group in groups:
            if time.monotonic()>deadline:raise TimeoutError('30-minute pilot budget reached')
            for e in data.examples(group):
                ids=tokenizer.apply_chat_template(e['messages'],tokenize=True,add_generation_prompt=True,return_tensors='pt')
                generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=24,
                    do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id,temperature=None,top_p=None)
                text=tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True).strip()
                rows.append(dict(group_id=e['group_id'],register=data.REGISTERS[e['register']],answer=e['answer'],
                    target=e['target'],generated=text,prediction=data.parse_answer(text),exact_template_match=text==e['target']))
    return rows

def validation(model,batches,deadline):
    model.eval();total=0.;count=0
    with torch.no_grad():
        for b in batches:
            if time.monotonic()>deadline:raise TimeoutError('30-minute pilot budget reached')
            out=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],labels=b['labels'],use_cache=False)
            n=int((b['labels'][:,1:]!=-100).sum());total+=float(out.loss)*n;count+=n
    return total/count

def run(model_dir,out,arms=tuple(ARMS)):
    torch.set_num_threads(2);torch.manual_seed(11);torch.use_deterministic_algorithms(True)
    model_dir=Path(model_dir);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest=verify_model(model_dir);deadline=time.monotonic()+1800
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    splits=data.build();(out/'splits.json').write_text(json.dumps(splits,indent=2)+'\n')
    train=[training.encode(tokenizer,data.examples(g)) for g in splits['train']]
    val=[training.encode(tokenizer,data.examples(g)) for g in splits['validation']]
    summary={'status':'executed_pretrained_lora_pilot','model_id':manifest['model_id'],'revision':manifest['revision'],
             'protocol_sha256':digest(ROOT/'PROTOCOL.md'),'python':platform.python_version(),'torch':torch.__version__,'arms':{}}
    for arm in arms:
        start=time.monotonic();torch.manual_seed(11)
        model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,
            use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager')
        model.config.use_cache=False;head=None;trace=[];best_step=0
        destination=out/arm;destination.mkdir(exist_ok=True)
        if arm!='base':
            model=get_peft_model(model,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',bias='none'))
            aux,cons=ARMS[arm]
            # Initialize equally across arms; optimize head only when its loss is used.
            head=torch.nn.Linear(model.config.hidden_size,3)
            params=[p for p in model.parameters() if p.requires_grad]+(list(head.parameters()) if aux else [])
            optimizer=torch.optim.AdamW(params,lr=.0005,weight_decay=0.)
            best=validation(model,val,deadline)
            best_state=[p.detach().clone() for p in params]
            trace.append({'step':0,'validation_lm_loss':best})
            order=list(range(len(train)));random.Random(11).shuffle(order)
            for step in range(1,25):
                if time.monotonic()>deadline:raise TimeoutError('30-minute pilot budget reached')
                model.train();head.train();optimizer.zero_grad(set_to_none=True)
                loss,parts=training.loss(model,head,train[order[(step-1)%len(order)]],aux,cons)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
                row={'step':step,'training_total':float(loss.detach()),**parts}
                if step%8==0:
                    score=validation(model,val,deadline);row['validation_lm_loss']=score
                    if score<best:
                        best=score;best_step=step;best_state=[p.detach().clone() for p in params]
                trace.append(row)
                (destination/'training.json').write_text(json.dumps(trace,indent=2)+'\n')
                print(arm,step,row,flush=True)
            with torch.no_grad():
                for p,value in zip(params,best_state):p.copy_(value)
            model.save_pretrained(destination,safe_serialization=True)
            if aux:save_file({k:v.detach().contiguous() for k,v in head.state_dict().items()},destination/'register_head.safetensors')
            count=sum(p.numel() for p in params)
        else:count=0
        rows=evaluate(model,tokenizer,splits['test'],deadline)
        (destination/'generations.json').write_text(json.dumps(rows,indent=2)+'\n')
        result=dict(selected_step=best_step,trainable_parameters=count,seconds=time.monotonic()-start,**summarize(rows))
        summary['arms'][arm]=result
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('RESULT',arm,result,flush=True)
        del model,head
        if arm!='base':del optimizer,params,best_state,loss
        gc.collect()
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=tuple(ARMS));a=p.parse_args()
    run(a.model_dir,a.out,(a.arm,) if a.arm else tuple(ARMS))
