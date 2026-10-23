"""Fixed one-permutation cyclic training schedule; no test exporter."""
import random


def need(ok,message):
    if not ok:raise ValueError(message)


def make_schedule(source_ids):
    ids=list(source_ids)
    need(ids and all(isinstance(x,str) and x for x in ids) and len(ids)==len(set(ids)),'unique nonempty training source IDs required')
    ids.sort();random.Random(11).shuffle(ids)
    return [ids[i%len(ids)] for i in range(128)]


def check_inputs(document):
    need(isinstance(document,dict) and set(document)=={'train','dev'},'only train/capacity-dev inputs allowed')
    train,dev=document['train'],document['dev']
    need(len(train)==96 and len(dev)==12,'exactly 96 train and 12 capacity-dev sources required')
    seen=set()
    for split,rows in document.items():
        for row in rows:
            sid=row['source_id'];need(sid not in seen,'duplicate source');seen.add(sid)
            need(row.get('split')==split,'source split mismatch; test not permitted')
            if split=='dev':need(row.get('capacity_gate') is True,'only assigned capacity-dev allowed')
            examples=row['examples'];need(len(examples)==2,'two complementary profiles required')
            need(examples[0]['known_labels']==[1-x for x in examples[1]['known_labels']],'profiles are not complementary')
            need(examples[0]['profile_id']!=examples[1]['profile_id'],'duplicate profile')
            for example in examples:
                ids=example['input_ids'];n=example['prompt_length']
                need(example['source_id']==sid and 0<n<len(ids)<=256,'source/token sequence boundary mismatch')
                need(example['assistant_mask']==[0]*n+[1]*(len(ids)-n),'assistant suffix mismatch')
    return make_schedule([r['source_id'] for r in train])
