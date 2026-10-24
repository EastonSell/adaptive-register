#!/usr/bin/env python3
"""Synthetic selection controls derived from PUBLIC development sources. No model calls."""
import itertools,json
from pathlib import Path
from analyze import analyze,DEFAULT,IDENTITY

def score(response, expected):
    # Fixed three-ASCII-character response, two distinct ascending one-digit IDs.
    if not isinstance(response,str) or len(response)!=3 or response[1]!=',' or response[0] not in '0123' or response[2] not in '0123' or response[0]>=response[2]:
        return {'valid':False,'unknown_recall':0.0,'selectivity':0.0,'exact_pair':False}
    selected={int(response[0]),int(response[2])}; target=set(expected)
    if len(target)!=2 or not target<=set(range(4)): raise ValueError('Malformed target')
    hits=len(selected&target)
    return {'valid':True,'unknown_recall':hits/2,'selectivity':hits-1,'exact_pair':selected==target}

def route_known_complement(model_input):
    """Deterministic selector using only allowed task input, never target IDs."""
    known=set(model_input['reader_knows'])
    candidates=model_input['candidates']
    terms={c['term'] for c in candidates}
    if len(candidates)!=4 or len(terms)!=4 or len(known)!=2 or not known<=terms:
        raise ValueError('Invalid four-term/two-known partition')
    if {c['id'] for c in candidates} != set(range(4)):
        raise ValueError('Invalid candidate IDs')
    return ','.join(str(i) for i in sorted(c['id'] for c in candidates if c['term'] not in known))

def build():
    audit, packets=analyze(DEFAULT,IDENTITY)
    prompts=[];targets=[];renders=[]
    for p in packets:
        for perm_index, order in enumerate(itertools.permutations(sorted(p['glossary']))):
            cid=f"{p['output_id']}-perm{perm_index:02}"
            candidates=[{'id':i,'term':term,'meaning':p['glossary'][term]['definition']} for i,term in enumerate(order)]
            known=sorted(p['profile']['known_concept_ids'])
            prompt={'case_id':cid,'kind':'synthetic_development_control_not_new_model_output','instruction':'Select exactly two meanings useful to a reader who already knows the listed terms. Return two distinct candidate IDs in ascending order, formatted as 0,1. Do not rewrite the source.','source':p['source'],'reader_knows':known,'candidates':candidates}
            target=[i for i,term in enumerate(order) if term in p['unknown_concept_ids']]
            prompts.append({'case_id':cid, 'kind':prompt.pop('kind'), 'model_input':{k:v for k,v in prompt.items() if k!='case_id'}});targets.append({'case_id':cid,'target_ids':target,'source_id':p['source_id'],'historical_output_id':p['output_id']})
            if perm_index==0:
                selected=[c for c in candidates if c['id'] in target]
                renders.append({'case_id':cid,'kind':'synthetic_deterministic_scaffold_not_model_output','source_prefix':p['source'],'glossary_appendix':'\n'.join(f"{c['term']}: {c['meaning']}." for c in selected),'claim':'Source bytes preserved, glossary definitions copied; not independently validated comprehension.'})
    summary={'kind':'synthetic_control_enumeration_not_model_performance','case_count':len(prompts),'source_scenario_count':12,'profiles_per_source':2,'candidate_permutations':24,'permutations_not_independent_evidence':True,'response_ascii_chars':3,'token_budget_status':'Same maximum response budget required prospectively; actual model tokenizer not run. Fixed ASCII length is NOT a claim of equal token count.','oracle_exact_pairs':0,'fixed_candidate_01_recall':0,'swapped_oracle_recall':0,'scaffold_count':len(renders)}
    for prompt,t in zip(prompts,targets):
        oracle=route_known_complement(prompt['model_input'])
        summary['oracle_exact_pairs']+=score(oracle,t['target_ids'])['exact_pair']
        summary['fixed_candidate_01_recall']+=score('0,1',t['target_ids'])['unknown_recall']
        swapped=','.join(str(i) for i in range(4) if i not in t['target_ids'])
        summary['swapped_oracle_recall']+=score(swapped,t['target_ids'])['unknown_recall']
    summary['fixed_candidate_01_recall']/=len(prompts)
    summary['swapped_oracle_recall']/=len(prompts)
    return prompts,targets,renders,summary

if __name__=='__main__':
    prompts,targets,renders,summary=build()
    for name,rows in [('selector-inputs.jsonl',prompts),('selector-gold.jsonl',targets),('synthetic-scaffolds.jsonl',renders)]:
        Path(name).write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows))
    Path('synthetic-results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
