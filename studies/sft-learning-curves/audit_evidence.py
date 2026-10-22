#!/usr/bin/env python3
"""Read-only, standard-library audit of two frozen SFT development studies.

No model execution is performed. JSONL generation text is independently parsed;
safetensors payloads are compared by key, dtype, shape, and raw bytes.
"""
import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def manifest(root, filename):
    entries = read(root / filename)
    for name, expected in entries.items():
        require(digest(root / name) == expected, f'Hash mismatch: {filename}: {name}')
    return {'entries_verified': len(entries), 'manifest_sha256': digest(root / filename)}


def tensors(path):
    raw = path.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8+size])
    payload = memoryview(raw)[8+size:]
    result = {}
    for key, spec in header.items():
        if key == '__metadata__':
            continue
        lo, hi = spec['data_offsets']
        require(0 <= lo <= hi <= len(payload), 'Invalid tensor offsets')
        result[key] = (spec['dtype'], spec['shape'], bytes(payload[lo:hi]))
    return result


def compare_tensors(a, b):
    first, second = tensors(a), tensors(b)
    require(first.keys() == second.keys(), 'Tensor key mismatch')
    for key in first:
        require(first[key] == second[key], f'Tensor mismatch: {key}')
    require(len(first) == 120, 'Expected 120 adapter tensors')
    return {'tensors_verified': len(first), 'comparison': 'key, dtype, shape, raw bytes',
            'first_sha256': digest(a), 'second_sha256': digest(b)}


def parse(text):
    numbers = re.findall(r'[-+]?\d+(?:\.\d+)?', text)
    match = re.search(r'\bValue:\s*([0-3])(?!\d|\.\d)', text)
    return int(match[1]) if match and numbers == [match[1]] else None


def audit_run(source, run, expected_steps, n_updates):
    result = {'source': manifest(source, 'FREEZE.json'),
              'run': manifest(run, 'RESULTS-MANIFEST.json')}
    cases = read(run / 'cases.json')
    require(cases == read(source / 'cases-development.json'), 'Case mismatch')
    require(set(cases) == {'train', 'validation'}, 'Unexpected data split')
    require(len(cases['train']) == 64 and len(cases['validation']) == 16, 'Case count mismatch')
    order = read(run / 'order.json')['record_indices']
    require(sorted(order) == list(range(64)), 'Training order not a permutation')
    train = rows(run / 'training.jsonl')
    require(len(train) == n_updates, 'Training row count mismatch')
    for i, row in enumerate(train):
        require(row['step'] == i+1 and row['group_id'] == cases['train'][order[i % 64]]['id'], 'Training continuity mismatch')
    result['training_updates'] = len(train)
    result['training_cpu_seconds'] = sum(x['cpu_seconds'] for x in train)
    result['training_wall_seconds'] = sum(x['wall_seconds'] for x in train)
    gens = rows(run / 'generations.jsonl')
    lms = rows(run / 'validation_lm.jsonl')
    curves = rows(run / 'curves.jsonl')
    require([c['step'] for c in curves] == expected_steps, 'Curve step mismatch')
    require(len(gens) == 48 * len(expected_steps) and len(lms) == 16 * len(expected_steps), 'Raw evaluation count mismatch')
    registers = {'introductory': ' This is the number stored in the requested slot.', 'working': '', 'technical': ' The lookup returns the stored scalar.'}
    val = {g['id']: g for g in cases['validation']}
    expected = {(gid, reg) for gid in val for reg in registers}
    result['evaluations'] = []
    for curve in curves:
        step = curve['step']
        gs = [g for g in gens if g['step'] == step]
        require(len(gs) == 48 and {(g['group_id'], g['register']) for g in gs} == expected, 'Generation coverage mismatch')
        groups = {}
        for g in gs:
            case = val[g['group_id']]
            answer = case['record'][case['query']]
            target = f'Value: {answer}.' + registers[g['register']]
            require(g['answer'] == answer and g['target'] == target, 'Generation label/target mismatch')
            require(g['prediction'] == parse(g['generated'].strip()), 'Parser mismatch')
            groups.setdefault(g['group_id'], []).append(g)
        valid = [group for group in groups.values() if all(g['prediction'] is not None for g in group)]
        changes = sum(len({g['prediction'] for g in group}) > 1 for group in valid)
        metrics = {'records': 16, 'responses': 48,
            'strict_correct': sum(g['prediction'] == g['answer'] for g in gs),
            'format_coverage': sum(g['prediction'] is not None for g in gs),
            'exact_target_template_match': sum(g['generated'].strip() == g['target'] for g in gs),
            'all_registers_correct': sum(all(g['prediction'] == g['answer'] for g in group) for group in groups.values()),
            'valid_groups': len(valid), 'invalid_groups': 16-len(valid),
            'valid_group_answer_change_rate': changes/len(valid) if valid else None,
            'answer_or_invalid_status_change_count': sum(len({g['prediction'] for g in group}) > 1 for group in groups.values())}
        for key, value in metrics.items():
            require(curve[key] == value, f'Metric mismatch: {step}: {key}')
        lm = [r for r in lms if r['step'] == step]
        require(len(lm) == 16 and {r['group_id'] for r in lm} == set(val), 'LM coverage mismatch')
        loss = sum(r['lm_loss']*r['target_tokens'] for r in lm)/sum(r['target_tokens'] for r in lm)
        require(math.isclose(loss, curve['validation_lm_loss'], rel_tol=1e-12, abs_tol=1e-12), 'LM loss mismatch')
        result['evaluations'].append({'step': step, **metrics, 'validation_lm_loss': loss,
            'lm_batches': len(lm), 'answer_changing_valid_groups': changes,
            'evaluation_cpu_seconds': curve['generation_cpu_seconds'] + curve['lm_cpu_seconds']})
    result['exposure'] = rows(run / 'exposure.jsonl')
    for row in result['exposure']:
        step = row['step']; unique = len({x['group_id'] for x in train[:step]})
        require(row['source_record_presentations'] == step and row['record_register_presentations'] == 3*step, 'Exposure count mismatch')
        require(row['unique_source_records'] == unique and row['unique_record_register_pairs'] == 3*unique, 'Unique exposure mismatch')
        require(row['record_pass_equivalents'] == step/64 and row['record_register_pass_equivalents'] == step/64, 'Pass equivalent mismatch')
    result['checkpoint_manifests'] = {}
    for checkpoint in sorted(run.glob('checkpoint-*')):
        for filename in ['adapter_manifest.json', 'checkpoint_manifest.json']:
            if (checkpoint / filename).exists():
                result['checkpoint_manifests'][checkpoint.name+'/'+filename] = manifest(checkpoint, filename)
    resource = rows(run / 'resources.jsonl')
    valid_resource = [r for r in resource if 'child_cpu_seconds' in r and 'combined_rss_bytes' in r]
    summary = read(run / 'resource_summary.json')
    peak = max(r['combined_rss_bytes'] for r in valid_resource)
    require(peak == summary['sampled_peak_combined_rss_bytes'], 'Peak RSS mismatch')
    result['resources'] = {'summary': summary, 'last_valid_sample': valid_resource[-1],
        'sample_count': len(valid_resource), 'nonstandard_rows': [r for r in resource if r not in valid_resource]}
    result['status'] = read(run / 'status.json')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('earlier-source', 'earlier-run', 'closure-source', 'closure-run', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    result = {'method': 'Independent read-only standard-library recount and payload comparison; no model execution.'}
    result['earlier'] = audit_run(a.earlier_source, a.earlier_run, [0, 24, 96], 326)
    result['closure'] = audit_run(a.closure_source, a.closure_run, [384], 384)
    first, second = rows(a.earlier_run/'training.jsonl'), rows(a.closure_run/'training.jsonl')
    fields = ['step', 'group_id', 'lm_loss', 'gradient_norm']
    require(all(all(x[k] == y[k] for k in fields) for x,y in zip(first,second)), 'Training prefix mismatch')
    require(read(a.earlier_run/'order.json') == read(a.closure_run/'order.json'), 'Order mismatch')
    result['continuity'] = {'matching_prefix_rows': len(first), 'fields': fields, 'timing_fields_excluded': True}
    result['tensor_continuity'] = {str(s): compare_tensors(a.earlier_run/f'checkpoint-{s:04d}/adapter_model.safetensors', a.closure_run/f'checkpoint-{s:04d}/adapter_model.safetensors') for s in (24,96)}
    result['original_step24'] = compare_tensors(a.earlier_source/'vendor/original-sft24.safetensors', a.earlier_run/'checkpoint-0024/adapter_model.safetensors')
    result['caveats'] = [
        'The earlier run is incomplete after a CPU-budget stop; it contains no step-384 evaluation.',
        'The closure is a separate repeated training run, not a resumed run; 0/24/96 evaluations are earlier observations reused for presentation.',
        'The closure has complete numerical outputs but retains incomplete/accounting_failure supervision status. Worker return code 0 does not establish clean supervised completion.',
        'Telemetry is sampled. Missing terminal accounting prevents asserting complete resource observability; a shutdown race is plausible but not proven.',
        'Compute for the two runs is additive, not one shared original allowance.',
        'One seed, synthetic records, correlated register variants, reused development validation, and changed exposure plus optimization constrain interpretation.',
        'No head comparison, fresh held-out evaluation, statistical significance, convergence, comprehension, accessibility, or model-transfer claim follows.',
        'Recovery-file hashes are checked; no resume test or exact-recovery guarantee is established.'
    ]
    result['audit_passed'] = True
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'audit_passed': True, 'output': str(a.output), 'matching_prefix_rows': len(first), 'tensor_checkpoints': [24,96]}))

if __name__ == '__main__':
    main()
