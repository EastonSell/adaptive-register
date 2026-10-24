#!/usr/bin/env python3
"""Model-free recount of retained public evidence; no model or network access."""
import argparse, collections, hashlib, itertools, json
from pathlib import Path
DEFAULT = Path(__file__).resolve().parent.parent / 'reader-capacity-gate'
IDENTITY = Path(__file__).resolve().parent / 'SOURCE_IDENTITY.json'

def analyze(source, identity):
    inventory = json.loads(identity.read_text())
    anchors = {r['path']:r['sha256'] for r in inventory['files']}
    receipts = []
    def read(name):
        b = (source/name).read_bytes(); sha = hashlib.sha256(b).hexdigest()
        if sha != anchors[name]: raise ValueError(f'Hash mismatch: {name}')
        receipts.append({'path':name, 'sha256':sha})
        return json.loads(b)
    ratings = read('adjudication/source-aware-final.json')
    packets = read('evaluation/source-aware-packet.json')
    comparisons = read('adjudication/output-only-ledger-comparison.json')
    adjudication_keys={(r['output_id'],p['proposition_id']):p['status'] for r in ratings for p in r['propositions']}
    comparison_keys={(r['output_id'],r['proposition_id']):r['source_aware_proposition_status'] for r in comparisons}
    assert len(adjudication_keys)==sum(len(r['propositions']) for r in ratings)==74
    assert len(comparison_keys)==len(comparisons)==74
    assert adjudication_keys==comparison_keys
    for p in packets:
        known=set(p['profile']['known_concept_ids']);unknown=set(p['unknown_concept_ids']);terms=set(p['glossary'])
        assert len(known)==len(unknown)==2 and not known&unknown and known|unknown==terms
    outputs = {r['output_id']:read(f"outputs/{r['output_id']}.json") for r in ratings}
    rows=[]
    for r in ratings:
        truth = all(p['status']=='preserved' for p in r['propositions'])
        support = all(c['status']=='supported' for c in r['unknown_concepts'])
        safe = truth and not r['unsupported_assertions']
        rows.append({'output_id':r['output_id'],'source_id':r['source_id'], 'truth_complete':truth,'truth_safe':safe,'all_unknown_supported':support,'tokens_including_eos':len(outputs[r['output_id']]['token_ids']), 'unknown_statuses':dict(collections.Counter(c['status'] for c in r['unknown_concepts']))})
    groups=collections.defaultdict(list)
    for r in rows: groups[r['source_id']].append(r)
    pair_counts=collections.Counter()
    for source_id, pair in groups.items():
        assert len(pair)==2
        pair_counts[str(sum(r['truth_complete'] for r in pair))+'_truth_complete']=pair_counts[str(sum(r['truth_complete'] for r in pair))+'_truth_complete']+1
    # Full finite enumeration of the no-profile selector null, NOT a fitted/model result.
    synthetic=[]
    for p in packets:
        candidates=sorted(p['glossary'])
        assert len(candidates)==4 and len(p['unknown_concept_ids'])==2
        unknown=set(p['unknown_concept_ids'])
        for chosen in itertools.combinations(candidates,2):
            score=len(set(chosen)&unknown)/2
            synthetic.append({'output_id':p['output_id'],'selected':chosen,'unknown_recall':score,'unknown_minus_known_selection':2*score-1,'exact_unknown_pair':set(chosen)==unknown})
    failed=[x for x in comparisons if x['source_aware_proposition_status']!='preserved']
    result={'kind':'retained_public_annotations_recount_and_synthetic_null','not_a_model_experiment':True,'historical_gate':'failed; 14/24 fluent truth complete, 3/24 all unknown supported; no B/C/S/test admission','corrected005':'not run; original seals lost; gate NULL/unvalidated', 'source_identity':{'repository':inventory['repository'],'acquisition_commit':inventory['acquisition_commit'],'unchanged_at_commit':inventory['unchanged_at_commit'],'latest_requested_commit_not_independently_verified':'79b745b5374293560ba9a0c546ce0403837ffb80'}, 'consumed_sha256':receipts,'outputs':rows,'source_pair_truth_counts':dict(pair_counts), 'per_scenario':[{'source_id':sid,'outputs':len(pair),'truth_complete':sum(r['truth_complete'] for r in pair),'all_unknown_supported':sum(r['all_unknown_supported'] for r in pair)} for sid,pair in groups.items()], 'all_supported_outputs':[r['output_id'] for r in rows if r['all_unknown_supported']], 'all_supported_and_truth_complete':sum(r['all_unknown_supported'] and r['truth_complete'] for r in rows), 'proposition_judgment_count':len(adjudication_keys),'preserved_proposition_judgments':sum(x=='preserved' for x in adjudication_keys.values()),'failed_proposition_judgments':len(failed),'failed_propositions_answerability':dict(collections.Counter(x['phase1_answerability'] for x in failed)), 'failed_but_answerable_ids':[{'output_id':x['output_id'],'proposition_id':x['proposition_id'],'ledger_comparison':x['ledger_comparison']} for x in failed if x['phase1_answerability']=='answerable'], 'synthetic_uniform_two_of_four_null':{'enumerated_cells':len(synthetic),'mean_unknown_recall':sum(x['unknown_recall'] for x in synthetic)/len(synthetic),'mean_unknown_minus_known_selection':sum(x['unknown_minus_known_selection'] for x in synthetic)/len(synthetic),'exact_pair_rate':sum(x['exact_unknown_pair'] for x in synthetic)/len(synthetic)}}
    return result,packets

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,default=DEFAULT);parser.add_argument('--identity',type=Path,default=IDENTITY);parser.add_argument('--output',type=Path,default=Path('audit-results.json'));args=parser.parse_args()
    result,_=analyze(args.source,args.identity);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['outputs','consumed_sha256']},indent=2))
