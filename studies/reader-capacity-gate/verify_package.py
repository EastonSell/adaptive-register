"""New public saved-output audit. No model, package install, network, or archive mutation.

This verifies the released public files, semantic evidence mappings and count
arithmetic. It is not the executed strict engineering validator and cannot
reconstruct model derivatives, omitted training inputs or private custody.
"""
from pathlib import Path
import collections
import hashlib
import json
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent

def need(ok, message):
    if not ok:
        raise ValueError(message)

def read(name):
    return json.loads((ROOT / name).read_text())

def main():
    manifest = read('MANIFEST.json')
    listed = manifest['files']
    expected = {x['path'] for x in listed} | {'MANIFEST.json'}
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
    need(actual == expected, 'Missing or unlisted package files')
    for x in listed:
        p = ROOT / x['path']
        need(not p.is_symlink() and p.stat().st_size == x['bytes'], 'File identity/size mismatch: '+x['path'])
        need(hashlib.sha256(p.read_bytes()).hexdigest() == x['sha256'], 'SHA-256 mismatch: '+x['path'])
    cases = read('evaluation/evaluation-cases.json')
    source = {x['output_id']:x for x in read('evaluation/source-aware-packet.json')}
    outputs = {x['output_id']:read('outputs/'+x['output_id']+'.json') for x in cases}
    need([x['output_id'] for x in cases] == ['O%03d'%i for i in range(1,25)], 'Frozen output denominator/order changed')
    need(len(source) == 24 and len({x['source_id'] for x in source.values()}) == 12, 'Source/profile count changed')
    for c in cases:
        oid = c['output_id']; o = outputs[oid]; s = source[oid]
        need(o['status'] == 'returned', 'Output did not return: '+oid)
        for k in ('output_id','source_id','profile_id','proposition_ids','unknown_concept_ids'):
            need(c[k] == o[k] == s[k], 'Case mapping changed: '+oid+'/'+k)
        for p in s['propositions']:
            ev = p['source_evidence']
            need(s['source'][ev['start']:ev['end']] == ev['text'], 'Source evidence mismatch')
    snapshots = {}
    for name in ('source-aware-initial-A','source-aware-initial-B','source-aware-final'):
        rows = read('adjudication/'+name+'.json')
        need([r['output_id'] for r in rows] == [c['output_id'] for c in cases], 'Rating order changed')
        for r,c in zip(rows,cases):
            text = outputs[r['output_id']]['scoring_text']
            for key,idkey,wanted in (('propositions','proposition_id',c['proposition_ids']),('unknown_concepts','concept_id',c['unknown_concept_ids'])):
                need([x[idkey] for x in r[key]] == wanted, 'Rating mapping changed')
                for x in r[key]:
                    need(bool(x['rationale']) and (not x['span'] or x['span'] in text), 'Rating evidence mismatch')
                    if x['status'] in ('preserved','supported'):
                        need(bool(x['span']), 'Passing rating has no evidence')
            for key in ('unsupported_assertions','known_explanations'):
                for x in r[key]:
                    need(bool(x['span']) and x['span'] in text and bool(x['rationale']), 'Assertion/descriptor evidence mismatch')
            need(r['adjudicator_kind'] == 'AI', 'Evaluator kind changed')
            final = name == 'source-aware-final'
            need(r['adjudication_complete'] is final and r['disagreement_resolved'] is final, 'Phase flags changed')
        snapshots[name] = rows
    need(sum(len(r['propositions']) for r in rows) == 74, 'Proposition count changed')
    need(sum(len(r['unknown_concepts']) for r in rows) == 48, 'Unknown-concept count changed')
    from source.scoring import aggregate
    result = aggregate(cases, snapshots['source-aware-final'], outputs=outputs)
    need(result == read('results/aggregate.json'), 'Published aggregate differs from released final ratings')
    need(result['denominator'] == 24 and result['fluent_truth_complete'] == 14 and result['unknown_support'] == 3, 'Frozen result changed')
    need(result['capacity_gate_passed'] is False, 'Unexpected gate verdict')
    recon = read('adjudication/reconciliation.json')
    items = recon['all_170_compared_decisions']
    need(len(items) == 170 and len(recon['all_24_unsupported_assertion_reviews']) == 24, 'Incomplete comparison record')
    need(sum(bool(x['categorical_disagreement']) for x in items) == 5, 'Disagreement count changed')
    need(recon['unresolved_interrater_disagreements'] == 0, 'Unresolved rater disagreement')
    final_lookup = {r['output_id']:r for r in snapshots['source-aware-final']}
    for x in items:
        text = outputs[x['output_id']]['scoring_text']; s = source[x['output_id']]['source']
        need(x['resolution_status'] == 'resolved' and x['final_span'] in text, 'Invalid reconciliation evidence')
        need(s[x['source_start']:x['source_end']] == x['source_span'], 'Invalid reconciliation source span')
        r = final_lookup[x['output_id']]
        if x['category'] in ('fluency','task_following'):
            actual_status = r[x['category']]
        else:
            idkey = 'proposition_id' if x['category'] == 'propositions' else 'concept_id'
            actual_status = next(y['status'] for y in r[x['category']] if y[idkey] == x['item_id'])
        need(actual_status == x['final_status'], 'Reconciliation/final rating differs')
    # Initial flags withhold finalization. These diagnostics intentionally ignore
    # those flags and must not be confused with completed official gate scores.
    for label,name in [('initial_A','source-aware-initial-A'),('initial_B','source-aware-initial-B')]:
        diagnostic = dict(fluent_truth_complete=0,all_unknown_supported=0,truth_safe=0,truth_safe_and_supported=0)
        for r in snapshots[name]:
            truth = all(x['status']=='preserved' for x in r['propositions'])
            support = all(x['status']=='supported' for x in r['unknown_concepts'])
            safe = truth and not r['unsupported_assertions']
            diagnostic['fluent_truth_complete'] += int(truth and r['fluency']=='pass' and r['task_following']=='pass')
            diagnostic['all_unknown_supported'] += int(support)
            diagnostic['truth_safe'] += int(safe)
            diagnostic['truth_safe_and_supported'] += int(safe and support)
        need(diagnostic == recon['initial_and_final_component_counts'][label], 'Initial provenance count differs')
    phase = read('adjudication/output-only-ratings.json')
    packet = {x['output_id']:x for x in read('evaluation/output-only-packet.json')}
    cross = {x['output_id']:x['questions'] for x in read('evaluation/question-crosswalk.json')}
    need(len(phase) == len(packet) == 24, 'Incomplete output-only record')
    count = 0
    for r in phase:
        oid=r['output_id']; text=outputs[oid]['scoring_text']
        need(r['scoring_text'] == text == packet[oid]['scoring_text'], 'Phase-1 scoring text changed')
        for q in r['questions']:
            need(q['evidence_span'] in text and q['question_id'] in cross[oid], 'Phase-1 evidence/mapping changed')
            count += 1
    need(count == 74, 'Phase-1 question count changed')
    derivation = read('DERIVATION.json')
    for x in derivation['files']:
        need(hashlib.sha256((ROOT/x['public_path']).read_bytes()).hexdigest() == x['public_sha256'], 'Derivation/public identity differs')
    print(json.dumps({'public_package_verified':True,'scope':'Released saved-output evidence/mappings and arithmetic only; no model or complete engineering replay.','adjudicated_outputs':24,'fluent_truth_complete':result['fluent_truth_complete'],'required_fluent_truth_complete':18,'unknown_support':result['unknown_support'],'required_unknown_support':12,'capacity_gate_passed':result['capacity_gate_passed'],'all_expected_evidence_spans_checked':True,'human_comprehension_measured':False},indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
