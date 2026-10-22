"""Exploratory inference only: base repeat and an unseen instruction label."""
import argparse,copy,gc,hashlib,json,sys,time,platform,os
from importlib.metadata import version
from pathlib import Path
HERE=Path(__file__).resolve().parent
PRETRAINED=HERE.parent
ROOT=PRETRAINED.parent
sys.path.insert(0,str(PRETRAINED))
import data
REGISTERS=(*data.REGISTERS,'concise')
ARMS=('base','sft','auxiliary','auxiliary_consistency')


def fourth(group):
    e=copy.deepcopy(data.examples(group)[1])
    e['messages'][1]['content']=e['messages'][1]['content'].replace('Requested register: working.','Requested register: concise.')
    e['register']='concise';e.pop('target')
    return e


def prompt_plan(groups):
    return {'original':[e for g in groups for e in data.examples(g)],'fourth':[fourth(g) for g in groups]}


def metrics(rows,registers):
    groups={};correct=covered=0
    for row in rows:
        group=groups.setdefault(row['group_id'],{})
        if row['register'] in group:raise ValueError('Duplicate group/register')
        prediction=data.parse_answer(row['generated'])
        group[row['register']]=(prediction,row['answer'])
        covered+=prediction is not None;correct+=prediction==row['answer']
    if not groups or any(set(g)!=set(registers) for g in groups.values()):raise ValueError('Missing or unexpected registers')
    valid=[g for g in groups.values() if all(p is not None for p,a in g.values())]
    changed=sum(len({p for p,a in g.values()})>1 for g in valid)
    allcorrect=sum(all(p==a for p,a in g.values()) for g in groups.values())
    return dict(record_count=len(groups),row_count=len(rows),strict_correct_count=correct,
        format_coverage_count=covered,format_coverage=covered/len(rows),all_registers_correct_count=allcorrect,
        all_registers_correct=allcorrect/len(groups),valid_group_count=len(valid),invalid_group_count=len(groups)-len(valid),
        answer_change_count=changed,valid_group_answer_change_rate=changed/len(valid) if valid else None,
        answer_or_invalid_status_change_count=sum(len({p for p,a in g.values()})>1 for g in groups.values()))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(model,tokenizer,examples,deadline):
    import torch
    model.eval()
    with torch.no_grad():
        for e in examples:
            if time.monotonic()>deadline:raise TimeoutError('Cooperative 30-minute inference budget exceeded')
            ids=tokenizer.apply_chat_template(e['messages'],tokenize=True,add_generation_prompt=True,return_tensors='pt')
            output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=24,
                do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id,temperature=None,top_p=None)
            text=tokenizer.decode(output[0,ids.shape[1]:],skip_special_tokens=True).strip()
            label=data.REGISTERS[e['register']] if isinstance(e['register'],int) else e['register']
            yield dict(group_id=e['group_id'],register=label,answer=e['answer'],generated=text,prediction=data.parse_answer(text))


def retain(generated,path):
    rows=[]
    with Path(path).open('x') as f:
        for row in generated:
            f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
            rows.append(row)
    return rows


def run(model_dir,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((HERE/'inputs.json').read_text())
    for path,sha in manifest.items():
        if digest(ROOT/path)!=sha:raise ValueError('Frozen input mismatch: '+path)
    import torch,pilot
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    runtime={name:version(name) for name in ('torch','transformers','peft','safetensors')}
    if runtime!={'torch':'2.5.1+cpu','transformers':'4.46.3','peft':'0.13.2','safetensors':'0.4.5'}:raise ValueError('Pinned runtime mismatch')
    model_dir=Path(model_dir);pilot.verify_model(model_dir)
    torch.set_num_threads(2);torch.manual_seed(11);torch.use_deterministic_algorithms(True)
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    groups=json.loads((PRETRAINED/'results/splits.json').read_text())['test']
    plan=json.loads((HERE/'prompts.json').read_text())
    if plan!=prompt_plan(groups):raise ValueError('Frozen prompts mismatch')
    deadline=time.monotonic()+1800
    result={'status':'running','scope':'exploratory reused facts, unseen instruction label; no retraining',
            'python':platform.python_version(),'runtime':runtime,
            'inputs_sha256':digest(HERE/'inputs.json'),'protocol_sha256':digest(HERE/'PROTOCOL.md'),'arms':{}}
    for arm in ARMS:
        model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,
            use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager')
        model.config.use_cache=False
        if arm!='base':model=PeftModel.from_pretrained(model,PRETRAINED/'adapters'/arm,local_files_only=True,is_trainable=False)
        old=json.loads((PRETRAINED/'results'/arm/'generations.json').read_text())
        current=retain(generate(model,tokenizer,plan['original'],deadline),out/'base-repeat-original.jsonl') if arm=='base' else old
        if arm=='base':
            (out/'base-repeat-original.json').write_text(json.dumps(current,indent=2)+'\n')
            if [(r['group_id'],r['register']) for r in current]!=[(r['group_id'],r['register']) for r in old]:raise ValueError('Base row alignment mismatch')
            result['base_repeat']={'generated_text_matches':sum(a['generated']==b['generated'] for a,b in zip(current,old)),
                'total':48,'original_three_metrics':metrics(current,data.REGISTERS)}
        fresh=retain(generate(model,tokenizer,plan['fourth'],deadline),out/(arm+'-fourth.jsonl'))
        (out/(arm+'-fourth.json')).write_text(json.dumps(fresh,indent=2)+'\n')
        result['arms'][arm]={'original_three_source':'fresh repeat' if arm=='base' else 'archived selected-step outputs',
            'original_three':metrics(current,data.REGISTERS),'four_register':metrics(current+fresh,REGISTERS),
            'fourth_strict_correct':sum(r['prediction']==r['answer'] for r in fresh),
            'fourth_format_coverage':sum(r['prediction'] is not None for r in fresh),'fourth_rows':16}
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(arm,'complete',flush=True)
        del model;gc.collect()
    result['status']='complete';(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.model_dir,a.out)
