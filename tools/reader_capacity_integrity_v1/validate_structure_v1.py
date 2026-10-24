#!/usr/bin/env python3
"""Supplement v1: structural consistency only, not semantic re-adjudication.

Keep outside the historical study directory. This does not execute package code,
recompute scores, authenticate provenance, or replace verify_package.py.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

VERSION = '1.0.0'
sys.dont_write_bytecode = True


class ValidationError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise ValidationError(message)


def reject_constant(value):
    raise ValidationError('Non-standard JSON numeric constant: ' + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate JSON object key: ' + key)
        result[key] = value
    return result


def ordered_ids(rows, key, expected, label):
    need(isinstance(rows, list), label + ': expected a list')
    ids = [row[key] for row in rows]
    need(ids == expected, label + ': missing, duplicate, reordered, or unknown ID')


def by_output(rows, expected, label):
    need(isinstance(rows, list), label + ': expected a list')
    ids = [row['output_id'] for row in rows]
    need(len(ids) == len(set(ids)) and set(ids) == set(expected), label + ': missing, duplicate, or unknown output ID')
    return {row['output_id']: row for row in rows}


def output_span(value, text, label, required=True):
    need(isinstance(value, str), label + ': span must be text')
    need(not required or bool(value.strip()), label + ': empty evidence span')
    need(value in text, label + ': span absent from text')


def source_span(start, end, value, text, label):
    need(type(start) is int and type(end) is int and 0 <= start < end <= len(text), label + ': invalid source offsets')
    need(isinstance(value, str) and bool(value.strip()) and text[start:end] == value, label + ': source evidence mismatch')


def validate_structure(root, expected_manifest_sha256=None):
    root = Path(root).resolve()
    cache = {}

    def read(name):
        if name not in cache:
            cache[name] = json.loads((root / name).read_text(encoding='utf-8'), object_pairs_hook=unique_object, parse_constant=reject_constant)
        return cache[name]

    if expected_manifest_sha256 is not None:
        need(hashlib.sha256((root / 'MANIFEST.json').read_bytes()).hexdigest() == expected_manifest_sha256,
             'Manifest differs from supplied trusted external SHA-256')
    manifest = read('MANIFEST.json')
    paths = [item['path'] for item in manifest['files']]
    need(len(paths) == len(set(paths)), 'Duplicate manifest path')
    for name in paths:
        need(isinstance(name, str) and bool(name) and '\\' not in name, 'Invalid manifest path')
        p = PurePosixPath(name)
        need(not p.is_absolute() and '..' not in p.parts and p.as_posix() == name and name != 'MANIFEST.json', 'Noncanonical or unsafe manifest path')
    all_paths = list(root.rglob('*'))
    need(not any(p.is_symlink() for p in all_paths), 'Symlink in package')
    need({p.relative_to(root).as_posix() for p in all_paths if p.is_file()} == set(paths) | {'MANIFEST.json'}, 'Missing or unlisted package file')
    for item in manifest['files']:
        path = root / item['path']
        need(path.stat().st_size == item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], 'Manifest identity mismatch: ' + item['path'])
        if path.suffix == '.json':
            read(item['path'])  # Reject duplicate JSON keys even in retained auxiliary tables.

    cases = read('evaluation/evaluation-cases.json')
    expected = ['O%03d' % n for n in range(1, 25)]
    ordered_ids(cases, 'output_id', expected, 'Cases')
    source = by_output(read('evaluation/source-aware-packet.json'), expected, 'Source packet')
    packet = by_output(read('evaluation/output-only-packet.json'), expected, 'Output-only packet')
    crosswalk = by_output(read('evaluation/question-crosswalk.json'), expected, 'Question crosswalk')
    phase1 = by_output(read('adjudication/output-only-ratings.json'), expected, 'Output-only ratings')
    ratings = {phase: by_output(read('adjudication/source-aware-' + phase + '.json'), expected, phase)
               for phase in ('initial-A', 'initial-B', 'final')}
    outputs = {oid: read('outputs/' + oid + '.json') for oid in expected}
    comparison_keys = set()
    for case in cases:
        oid = case['output_id']
        s = source[oid]
        propositions = case['proposition_ids']
        concepts = case['unknown_concept_ids']
        need(len(propositions) == len(set(propositions)) and len(concepts) == len(set(concepts)), 'Duplicate required proposition/concept ID')
        ordered_ids(s['propositions'], 'proposition_id', propositions, 'Source proposition coverage: ' + oid)
        for p in s['propositions']:
            ev = p['source_evidence']
            source_span(ev['start'], ev['end'], ev['text'], s['source'], 'Source proposition: ' + oid)
        for phase, phase_rows in ratings.items():
            need(all(phase_rows[oid][key] == case[key] for key in ('source_id', 'profile_id')), 'Rating case linkage: ' + phase + '/' + oid)
        questions = crosswalk[oid]['questions']
        need(isinstance(questions, dict), 'Crosswalk must be an object: ' + oid)
        qids = ['Q%02d' % n for n in range(1, len(propositions) + 1)]
        need(questions == dict(zip(qids, propositions)), 'Crosswalk proposition coverage: ' + oid)
        ordered_ids(packet[oid]['questions'], 'question_id', qids, 'Packet question coverage: ' + oid)
        ordered_ids(phase1[oid]['questions'], 'question_id', qids, 'Rated question coverage: ' + oid)
        for q, packet_q, proposition in zip(phase1[oid]['questions'], packet[oid]['questions'], s['propositions']):
            need(q['question'] == packet_q['question'] == proposition['question'], 'Question text linkage: ' + oid)
            need(q['answerability'] in ('answerable', 'unanswerable', 'uncertain'), 'Invalid answerability: ' + oid)
            # O007/Q01 legitimately has no evidence because it is unanswerable.
            output_span(q['evidence_span'], outputs[oid]['scoring_text'], 'Phase-1 evidence: ' + oid,
                        required=q['answerability'] == 'answerable')
        comparison_keys.update((oid, 'propositions', p) for p in propositions)
        comparison_keys.update((oid, 'unknown_concepts', c) for c in concepts)
        comparison_keys.update((oid, key, key) for key in ('fluency', 'task_following'))

    recon = read('adjudication/reconciliation.json')
    items = recon['all_170_compared_decisions']
    keys = [(x['output_id'], x['category'], x['item_id']) for x in items]
    need(len(keys) == len(set(keys)) and set(keys) == comparison_keys and len(keys) == 170, 'Reconciliation decision coverage')
    for item in items:
        oid, category, ident = item['output_id'], item['category'], item['item_id']
        output_span(item['final_span'], outputs[oid]['scoring_text'], 'Reconciliation output evidence: ' + oid)
        source_span(item['source_start'], item['source_end'], item['source_span'], source[oid]['source'], 'Reconciliation source evidence: ' + oid)
        for phase, field in (('initial-A', 'initial'), ('initial-B', 'independent')):
            row = ratings[phase][oid]
            if category in ('fluency', 'task_following'):
                recorded = {'status': row[category], 'rationale': row[category + '_rationale']}
            else:
                idkey = 'proposition_id' if category == 'propositions' else 'concept_id'
                recorded = next(x for x in row[category] if x[idkey] == ident)
            need(item[field] == recorded, 'Reconciliation initial snapshot mismatch: ' + oid)
        need(type(item['categorical_disagreement']) is bool and item['categorical_disagreement'] == (item['initial']['status'] != item['independent']['status']), 'Reconciliation disagreement flag mismatch')
    counts = {category: sum(item['category'] == category for item in items)
              for category in ('propositions', 'unknown_concepts', 'fluency', 'task_following')}
    counts['unsupported_assertion_reviews'] = len(recon['all_24_unsupported_assertion_reviews'])
    need(recon['compared_counts'] == counts, 'Reconciliation count summary mismatch')
    differences = [item for item in items if item['categorical_disagreement']]
    need(recon['initial_categorical_disagreements'] == len(differences), 'Disagreement count summary mismatch')
    need(len(differences) == 5 and recon['categorical_disagreements_and_resolutions'] == differences, 'Disagreement detail coverage')
    reviews = by_output(recon['all_24_unsupported_assertion_reviews'], expected, 'Unsupported-assertion reviews')
    for oid, review in reviews.items():
        need(review['complete_output_review_span'] == outputs[oid]['scoring_text'], 'Unsupported review output span: ' + oid)
        need(review['complete_source_review_span'] == source[oid]['source'], 'Unsupported review source span: ' + oid)
        for phase, field in (('initial-A', 'initial'), ('initial-B', 'independent'), ('final', 'final')):
            need(review[field] == ratings[phase][oid]['unsupported_assertions'], 'Unsupported review decision mismatch: ' + oid)
    aggregate = read('results/aggregate.json')
    reconciled = {key: aggregate['unknown_support' if key == 'all_unknown_supported' else key]
                  for key in ('fluent_truth_complete', 'all_unknown_supported', 'truth_safe', 'truth_safe_and_supported')}
    need(recon['initial_and_final_component_counts']['reconciled'] == reconciled, 'Reconciled component summary mismatch')
    return {'supplement_version': VERSION, 'structural_checks_passed': True,
            'output_ids': len(cases), 'question_mappings': sum(len(row['questions']) for row in phase1.values()),
            'comparison_keys': len(keys),
            'external_manifest_anchor_checked': expected_manifest_sha256 is not None,
            'semantic_judgments_validated': False, 'aggregate_recomputed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', required=True, type=Path)
    parser.add_argument('--expected-manifest-sha256', help='Optional hash obtained from a trusted immutable revision, not the candidate package.')
    args = parser.parse_args()
    try:
        result = validate_structure(args.package_root, args.expected_manifest_sha256)
    except (ValidationError, KeyError, TypeError, ValueError, OSError, StopIteration) as exc:
        print('Supplement rejected package: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
