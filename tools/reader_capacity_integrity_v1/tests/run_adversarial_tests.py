#!/usr/bin/env python3
"""Stdlib integration tests using disposable copies of a trusted frozen package.

Resealing fixtures is test-only: it exposes checks below the byte-identity layer.
Never run this against an untrusted package: the historical verifier is Python.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
VERIFIER_SHA256 = '999af822e30a062ccfd31dd298fc88db5c3eddf522e1b05baafd153789f28845'
MANIFEST_SHA256 = '6ce738fc46e3fffdbfcff2be13be3354a8826f9c8d62e8136cc3bf2655921da5'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(root, name):
    return json.loads((root / name).read_text(encoding='utf-8'))


def write(root, name, value):
    (root / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def edit(name, change):
    def mutate(root):
        data = read(root, name)
        change(data)
        write(root, name, data)
    return mutate


def reseal(root):
    """Refresh test-copy public hashes, then manifest hashes. Not production code."""
    derivation = read(root, 'DERIVATION.json')
    for item in derivation['files']:
        path = root / item['public_path']
        if path.is_file():
            item['public_sha256'] = digest(path)
    write(root, 'DERIVATION.json', derivation)
    manifest = read(root, 'MANIFEST.json')
    for item in manifest['files']:
        path = root / item['path']
        if path.is_file():
            item['bytes'] = path.stat().st_size
            item['sha256'] = digest(path)
    write(root, 'MANIFEST.json', manifest)


def replace_with_first(rows):
    rows[1] = copy.deepcopy(rows[0])


def append_first(rows):
    rows.append(copy.deepcopy(rows[0]))


def duplicate_reconciliation(data):
    # Both are non-disagreements, so total=170 and disagreement count=5 survive.
    replace_with_first(data['all_170_compared_decisions'])


def empty_source_evidence(rows):
    rows[0]['propositions'][0]['source_evidence'] = {'start': 0, 'end': 0, 'text': ''}


def negative_source_evidence(rows):
    source = rows[0]['source']
    rows[0]['propositions'][0]['source_evidence'] = {
        'start': -len(source), 'end': 71, 'text': source[:71]}


def add_unlisted(root):
    (root / 'unlisted.txt').write_text('extra fixture\n', encoding='utf-8')


def corrupt_json(root):
    (root / 'evaluation/evaluation-cases.json').write_text('[', encoding='utf-8')


def duplicate_json_key(root):
    path = root / 'evaluation/evaluation-cases.json'
    text = path.read_text(encoding='utf-8')
    path.write_text(text.replace('"output_id": "O001"', '"output_id": "HIDDEN", "output_id": "O001"', 1), encoding='utf-8')


# Each mutation tests a distinct observable validation contract. The final boolean
# records independently inspected historical behavior, not a desired new verdict.
CASES = [
    ('baseline', None, False, True, True),
    # Scope control: structural consistency cannot decide whether prose is a sound rationale.
    ('changed_nonempty_final_rationale_scope_control', edit('adjudication/source-aware-final.json', lambda d: d[0]['propositions'][0].update(rationale='A deliberately unassessed replacement rationale.')), True, True, True),
    ('unresealed_changed_output', edit('outputs/O001.json', lambda d: d.update(scoring_text='changed')), False, False, False),
    ('missing_output_file', lambda r: (r / 'outputs/O001.json').unlink(), False, False, False),
    ('unlisted_file', add_unlisted, False, False, False),
    ('malformed_cases_json', corrupt_json, True, False, False),
    ('missing_case_mapping_field', edit('evaluation/evaluation-cases.json', lambda d: d[0].pop('profile_id')), True, False, False),
    ('null_case_mapping', edit('evaluation/evaluation-cases.json', lambda d: d[0].update(proposition_ids=None)), True, False, False),
    ('duplicate_case_output_id', edit('evaluation/evaluation-cases.json', replace_with_first), True, False, False),
    ('changed_output_mapping', edit('outputs/O001.json', lambda d: d.update(source_id='WRONG')), True, False, False),
    ('changed_source_evidence_text', edit('evaluation/source-aware-packet.json', lambda d: d[0]['propositions'][0]['source_evidence'].update(text='NOT IN SOURCE')), True, False, False),
    ('changed_final_evidence_span', edit('adjudication/source-aware-final.json', lambda d: d[0]['propositions'][0].update(span='NOT IN OUTPUT')), True, False, False),
    ('missing_passing_evidence', edit('adjudication/source-aware-final.json', lambda d: d[0]['propositions'][0].update(span='')), True, False, False),
    ('duplicate_final_rating', edit('adjudication/source-aware-final.json', replace_with_first), True, False, False),
    ('changed_aggregate_denominator', edit('results/aggregate.json', lambda d: d.update(denominator=23)), True, False, False),
    ('changed_aggregate_gate', edit('results/aggregate.json', lambda d: d.update(capacity_gate_passed=True)), True, False, False),
    ('changed_phase1_evidence', edit('adjudication/output-only-ratings.json', lambda d: d[0]['questions'][0].update(evidence_span='NOT IN OUTPUT')), True, False, False),
    ('unknown_phase1_question_id', edit('adjudication/output-only-ratings.json', lambda d: d[0]['questions'][0].update(question_id='Q99')), True, False, False),
    ('changed_reconciliation_source_span', edit('adjudication/reconciliation.json', lambda d: d['all_170_compared_decisions'][0].update(source_span='NOT IN SOURCE')), True, False, False),
    ('changed_reconciliation_final_status', edit('adjudication/reconciliation.json', lambda d: d['all_170_compared_decisions'][0].update(final_status='omitted')), True, False, False),
    ('reordered_crosswalk_object_members_control', edit('evaluation/question-crosswalk.json', lambda d: d[0].update(questions=dict(reversed(list(d[0]['questions'].items()))))), True, True, True),
    ('changed_initial_source_link', edit('adjudication/source-aware-initial-A.json', lambda d: d[0].update(source_id='WRONG')), True, True, False),
    ('changed_initial_profile_link', edit('adjudication/source-aware-initial-A.json', lambda d: d[0].update(profile_id='WRONG')), True, True, False),
    ('changed_phase1_question_text', edit('adjudication/output-only-ratings.json', lambda d: d[0]['questions'][0].update(question='A different question?')), True, True, False),
    ('invalid_phase1_answerability', edit('adjudication/output-only-ratings.json', lambda d: d[0]['questions'][0].update(answerability='garbage', evidence_span='')), True, True, False),
    ('nonfinite_json_number', edit('outputs/O001.json', lambda d: d.update(fixture_nonfinite=float('nan'))), True, True, False),
    ('changed_reconciliation_count_summary', edit('adjudication/reconciliation.json', lambda d: d['compared_counts'].update(propositions=73)), True, True, False),
    ('changed_reconciled_component_count', edit('adjudication/reconciliation.json', lambda d: d['initial_and_final_component_counts']['reconciled'].update(fluent_truth_complete=0)), True, True, False),
    ('changed_disagreement_count_summary', edit('adjudication/reconciliation.json', lambda d: d.update(initial_categorical_disagreements=0)), True, True, False),
    ('duplicate_manifest_entry', edit('MANIFEST.json', lambda d: append_first(d['files'])), False, True, False),
    ('duplicate_source_packet_output', edit('evaluation/source-aware-packet.json', append_first), True, True, False),
    ('missing_source_proposition', edit('evaluation/source-aware-packet.json', lambda d: d[0]['propositions'].pop()), True, True, False),
    ('empty_source_evidence', edit('evaluation/source-aware-packet.json', empty_source_evidence), True, True, False),
    ('negative_source_offset', edit('evaluation/source-aware-packet.json', negative_source_evidence), True, True, False),
    ('duplicate_phase1_packet_output', edit('evaluation/output-only-packet.json', append_first), True, True, False),
    ('duplicate_crosswalk_output', edit('evaluation/question-crosswalk.json', append_first), True, True, False),
    ('wrong_crosswalk_proposition', edit('evaluation/question-crosswalk.json', lambda d: d[0]['questions'].update(Q01='NONEXISTENT')), True, True, False),
    ('duplicate_phase1_rating_output', edit('adjudication/output-only-ratings.json', replace_with_first), True, True, False),
    ('duplicate_phase1_question', edit('adjudication/output-only-ratings.json', lambda d: replace_with_first(d[0]['questions'])), True, True, False),
    ('empty_phase1_evidence', edit('adjudication/output-only-ratings.json', lambda d: d[0]['questions'][0].update(evidence_span='')), True, True, False),
    ('duplicate_reconciliation_decision', edit('adjudication/reconciliation.json', duplicate_reconciliation), True, True, False),
    ('empty_reconciliation_output_span', edit('adjudication/reconciliation.json', lambda d: d['all_170_compared_decisions'][0].update(final_span='')), True, True, False),
    ('duplicate_unsupported_review', edit('adjudication/reconciliation.json', lambda d: replace_with_first(d['all_24_unsupported_assertion_reviews'])), True, True, False),
    ('changed_reconciliation_initial_status', edit('adjudication/reconciliation.json', lambda d: d['all_170_compared_decisions'][0]['initial'].update(status='omitted')), True, True, False),
    ('missing_disagreement_detail', edit('adjudication/reconciliation.json', lambda d: d['categorical_disagreements_and_resolutions'].pop()), True, True, False),
    ('changed_unsupported_output_review_span', edit('adjudication/reconciliation.json', lambda d: d['all_24_unsupported_assertion_reviews'][0].update(complete_output_review_span='NOT IN OUTPUT')), True, True, False),
    ('changed_unsupported_source_review_span', edit('adjudication/reconciliation.json', lambda d: d['all_24_unsupported_assertion_reviews'][0].update(complete_source_review_span='NOT IN SOURCE')), True, True, False),
    ('changed_unsupported_review_decision', edit('adjudication/reconciliation.json', lambda d: d['all_24_unsupported_assertion_reviews'][0].update(final=[{'span': 'changed', 'rationale': 'changed'}])), True, True, False),
    ('duplicate_json_object_key', duplicate_json_key, True, True, False),
]


def run_frozen(root):
    result = subprocess.run([sys.executable, '-B', str(root / 'verify_package.py')],
                            cwd=root, capture_output=True, text=True, timeout=30)
    return result.returncode == 0, result.stdout, result.stderr


def snapshot(root):
    return {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--mode', choices=('characterize', 'require-rejection', 'supplement'), default='supplement')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    if args.report is not None and args.report.resolve().is_relative_to(root):
        parser.error('Report destination must be outside the historical package; no package code was run.')
    # Trust the independently pinned manifest before any executable package code.
    if digest(root / 'MANIFEST.json') != MANIFEST_SHA256 or digest(root / 'verify_package.py') != VERIFIER_SHA256:
        parser.error('Input is not the pinned historical package; no package code was run.')
    manifest = read(root, 'MANIFEST.json')
    all_paths = list(root.rglob('*'))
    expected_paths = {item['path'] for item in manifest['files']} | {'MANIFEST.json'}
    if any(p.is_symlink() for p in all_paths) or {p.relative_to(root).as_posix() for p in all_paths if p.is_file()} != expected_paths:
        parser.error('Unexpected file or symlink in pinned input; no package code was run.')
    for item in manifest['files']:
        path = root / item['path']
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            parser.error('Input differs from the pinned package; no package code was run: ' + item['path'])
    before = snapshot(root)
    if args.mode == 'supplement':
        sys.path.insert(0, str(HERE))
        from validate_structure_v1 import validate_structure, ValidationError
    results = []
    for name, mutation, seal, historical_accepts, desired_accepts in CASES:
        with tempfile.TemporaryDirectory(prefix='reader-integrity-') as tmp:
            fixture = Path(tmp) / 'package'
            shutil.copytree(root, fixture)
            if mutation is not None:
                mutation(fixture)
            if seal:
                reseal(fixture)
            accepts, stdout, stderr = run_frozen(fixture)
            frozen_accepts = accepts
            if args.mode == 'supplement' and accepts:
                try:
                    validate_structure(fixture)
                except (ValidationError, KeyError, TypeError, ValueError, OSError) as exc:
                    accepts = False
                    stderr += '\nSupplement: ' + type(exc).__name__ + ': ' + str(exc)
            expected = historical_accepts if args.mode == 'characterize' else desired_accepts
            frozen_matched = frozen_accepts == historical_accepts
            passed = frozen_matched and accepts == expected
            record = {'name': name, 'resealed_test_copy': seal, 'expected_acceptance': expected,
                      'frozen_accepted': frozen_accepts, 'frozen_expected_acceptance': historical_accepts,
                      'frozen_behavior_matched': frozen_matched, 'observed_acceptance': accepts,
                      'passed': passed, 'stdout': stdout, 'stderr': stderr}
            results.append(record)
            print(('PASS' if passed else 'FAIL') + ' ' + name + ' (accepted=' + str(accepts) + ')')
    unchanged = before == snapshot(root)
    report = {'schema_version': 1, 'mode': args.mode, 'python': sys.version,
              'trusted_manifest_sha256': MANIFEST_SHA256, 'source_file_count': len(before),
              'source_unchanged': unchanged, 'cases': results,
              'passed': sum(r['passed'] for r in results), 'total': len(results)}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(str(report['passed']) + '/' + str(report['total']) + ' passed; source unchanged=' + str(unchanged))
    return 0 if unchanged and all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
