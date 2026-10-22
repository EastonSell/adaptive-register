"""Assistant-only supervised loss plus explicit-register auxiliary objectives."""
import torch
import torch.nn.functional as F

def encode(tokenizer,examples):
    digits=[tokenizer.encode(str(i),add_special_tokens=False) for i in range(4)]
    if any(len(d)!=1 for d in digits) or len({d[0] for d in digits})!=4:raise ValueError('Need four distinct single-token digits.')
    digit_ids=[d[0] for d in digits];rows=[]
    for e in examples:
        prompt=tokenizer.apply_chat_template(e['messages'],tokenize=True,add_generation_prompt=True)
        target=tokenizer.encode(e['target'],add_special_tokens=False)
        matches=[i for i,t in enumerate(target) if t in digit_ids]
        if len(matches)!=1 or target[matches[0]]!=digit_ids[e['answer']]:raise ValueError('Ambiguous answer token.')
        rows.append((prompt+target+[tokenizer.eos_token_id],len(prompt),len(target),len(prompt)+matches[0]))
    length=max(len(r[0]) for r in rows);n=len(rows)
    ids=torch.full((n,length),tokenizer.eos_token_id,dtype=torch.long)
    mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100);aux=torch.zeros_like(ids,dtype=torch.float)
    for i,(tokens,start,count,_) in enumerate(rows):
        ids[i,:len(tokens)]=torch.tensor(tokens);mask[i,:len(tokens)]=1
        labels[i,start:len(tokens)]=torch.tensor(tokens[start:]);aux[i,start:start+count]=1
    return {'input_ids':ids,'attention_mask':mask,'labels':labels,'aux_mask':aux,
            'answer_positions':torch.tensor([r[3] for r in rows]),'prompt_lengths':[r[1] for r in rows],
            'digit_ids':digit_ids,'registers':torch.tensor([e['register'] for e in examples])}

def consistency(logits):
    p=logits.softmax(-1)
    return ((p-p.mean(0,keepdim=True))**2).mean()

def loss(model,head,batch,aux_weight,consistency_weight):
    out=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],labels=batch['labels'],
              output_hidden_states=bool(aux_weight),use_cache=False,return_dict=True)
    total=out.loss;parts={'lm':float(out.loss.detach())}
    if aux_weight:
        h=out.hidden_states[-1];mask=batch['aux_mask'].unsqueeze(-1)
        pooled=(h*mask).sum(1)/mask.sum(1)
        aux=F.cross_entropy(head(pooled),batch['registers']);total=total+aux_weight*aux
        parts['auxiliary']=float(aux.detach())
    if consistency_weight:
        logits=out.logits[torch.arange(len(batch['input_ids'])),batch['answer_positions']-1][:,batch['digit_ids']]
        cons=consistency(logits);total=total+consistency_weight*cons
        parts['consistency']=float(cons.detach())
    return total,parts
