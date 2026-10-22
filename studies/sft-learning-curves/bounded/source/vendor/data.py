"""Original synthetic record queries. No external dataset or inferred user traits."""
import itertools,random,re
REGISTERS=('introductory','working','technical')
SLOTS=('red','blue','green','yellow')

def build(seed=20261022):
    rng=random.Random(seed);records=list(itertools.product(range(4),repeat=4));rng.shuffle(records)
    groups=[{'id':i,'record':list(r),'query':rng.randrange(4)} for i,r in enumerate(records[:96])]
    return {'train':groups[:64],'validation':groups[64:80],'test':groups[80:]}

def examples(group):
    answer=group['record'][group['query']]
    result=[]
    for index,register in enumerate(REGISTERS):
        detail={'introductory':' This is the number stored in the requested slot.',
                'working':'','technical':' The lookup returns the stored scalar.'}[register]
        record=', '.join(f'{k}={v}' for k,v in zip(SLOTS,group['record']))
        messages=[{'role':'system','content':'Answer only from the supplied record. Honor the explicitly requested register. Begin with Value: followed by the answer number.'},
                  {'role':'user','content':f'Record: {record}. What is in the {SLOTS[group["query"]]} slot? Requested register: {register}.'}]
        result.append({'group_id':group['id'],'register':index,'answer':answer,'messages':messages,'target':f'Value: {answer}.'+detail})
    return result

def parse_answer(text):
    # Reject multiple integers, signs, decimals, unsupported values or absent prefix.
    numbers=re.findall(r'[-+]?\d+(?:\.\d+)?',text)
    match=re.search(r'\bValue:\s*([0-3])(?!\d|\.\d)',text)
    return int(match.group(1)) if match and numbers==[match.group(1)] else None
