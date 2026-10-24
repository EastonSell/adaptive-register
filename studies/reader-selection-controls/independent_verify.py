#!/usr/bin/env python3
"""Read-only independent checks; no model, API, or remote publication."""
import collections, hashlib, itertools, json, sys
from pathlib import Path
package=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
source=package.parent/'reader-capacity-gate'
identity=package/'SOURCE_IDENTITY.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
inv=load(identity)
source_checks=[{'path':r['path'],'sha256':sha(source/r['path']),'matches':sha(source/r['path'])==r['sha256']} for r in inv['files']]
assert all(x['matches'] for x in source_checks)
ratings=load(source/'adjudication/source-aware-final.json'); comparisons=load(source/'adjudication/output-only-ledger-comparison.json')
expected={(r['output_id'],p['proposition_id']):p['status'] for r in ratings for p in r['propositions']}
assert len(expected)==sum(len(r['propositions']) for r in ratings)
observed={(r['output_id'],r['proposition_id']):r['source_aware_proposition_status'] for r in comparisons}
assert len(observed)==len(comparisons) and expected==observed
cross=collections.Counter((r['source_aware_proposition_status'],r['phase1_answerability']) for r in comparisons)
prompts=[json.loads(l) for l in (package/'selector-inputs.jsonl').read_text().splitlines()]
gold={r['case_id']:r for r in map(json.loads,(package/'selector-gold.jsonl').read_text().splitlines())}
assert len(gold)==len(prompts)==576
pairs=collections.defaultdict(list);oracle_correct=0
for r in prompts:
 x=r['model_input'];g=gold[r['case_id']]; candidates=x['candidates'];known=set(x['reader_knows']);terms={c['term'] for c in candidates}
 assert len(candidates)==4 and len(terms)==4 and {c['id'] for c in candidates}==set(range(4))
 assert len(known)==2 and known<=terms
 prediction={c['id'] for c in candidates if c['term'] not in known}
 oracle_correct+=prediction==set(g['target_ids'])
 assert set(x)=={'instruction','source','reader_knows','candidates'}
 pairs[(g['source_id'],json.dumps(candidates,sort_keys=True))].append((x,g))
blind_checks=0;max_exact=0
for pair in pairs.values():
 assert len(pair)==2
 (a,ga),(b,gb)=pair
 assert {k:v for k,v in a.items() if k!='reader_knows'}=={k:v for k,v in b.items() if k!='reader_knows'}
 assert not set(a['reader_knows']) & set(b['reader_knows'])
 ta,tb=set(ga['target_ids']),set(gb['target_ids'])
 assert ta.isdisjoint(tb) and ta|tb==set(range(4))
 for chosen in map(set,itertools.combinations(range(4),2)):
  ha,hb=len(chosen&ta),len(chosen&tb)
  assert (ha+hb)/4==.5 and ((ha-1)+(hb-1))/2==0
  max_exact=max(max_exact,((chosen==ta)+(chosen==tb))/2)
  blind_checks+=1
assert oracle_correct==576 and len(pairs)==288 and max_exact==.5
result={'review_scope':'Retained file identities and model-free controls only; no remote HEAD verification or comprehension test.','source_inventory_sha256':sha(identity),'source_files_checked':len(source_checks),'source_checks':source_checks,'annotation_denominator':len(comparisons),'annotation_cross_tab':[{'truth_status':a,'answerability':b,'count':n} for (a,b),n in sorted(cross.items())],'failed_answerable':sum(n for (a,b),n in cross.items() if a!='preserved' and b=='answerable'),'failed_total':sum(n for (a,b),n in cross.items() if a!='preserved'),'input_only_oracle_correct':oracle_correct,'oracle_total':len(prompts),'paired_permutation_cells':len(pairs),'blind_pair_checks':blind_checks,'profile_blind_paired_recall':.5,'profile_blind_paired_selectivity':0,'profile_blind_max_paired_exact':max_exact,'uniform_random_exact_probability':1/6,'source_clusters':len({g['source_id'] for g in gold.values()})}
print(json.dumps(result,indent=2))
