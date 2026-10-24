# Reader capacity audit: integrity regression suite and structural supplement v1

Prepared 24 October 2026. No historical evidence, manifest, verifier, score, or threshold was changed. No model execution, training, API call, package installation, or external repository write occurred in this audit.

## Result

The original 59-file package passes unchanged and still reports 14/24 fluent/truth-complete outputs and 3/24 all-unknown-supported outputs, below the frozen 18/24 and 12/24 floors. This work identifies validation-coverage limitations, **not corruption of the published package or an altered scientific conclusion**.

The integration suite has 49 cases: the original baseline, two explicitly accepted controls, and 46 malformed variants. The historical verifier rejects 18 variants but accepts 28, including 27 deliberately resealed fixtures and one duplicate manifest entry. The original verifier plus the separately versioned structural supplement rejects all 46 malformed variants and accepts the baseline plus both controls. The suite compares all source-file hashes before and after to verify that the historical input remained unchanged.

The 28 acceptances are test cases, not 28 independent security vulnerabilities. They cluster around six structural weaknesses: duplicate collapse, missing mapping coverage, permissive span checks, count-only reconciliation coverage, unchecked mirrored provenance records, and permissive JSON/schema parsing.

## What is added

- `validate_structure_v1.py`: a standard-library-only **supplement**, version 1.0.0. It checks selected structural relationships and byte identities without executing package code. It does not recompute the aggregate or replace the historical verifier.
- `tests/run_adversarial_tests.py`: a command-line integration suite. It first verifies its input against pinned manifest and verifier hashes, checks the complete file set and rejects symlinks, then tests disposable copies. It runs the real historical verifier, not mocks.
- `tests/test_runner_safety.py`: three input-preservation tests for report destinations and unlisted input files.
- `SOURCE_IDENTITY.json`: original source identities and unchanged-commit provenance.
- `AUDIT.md`: findings, root causes, exact historical source lines, and remaining limits.
- `verification/`: full red/green results, baseline outputs and source-preservation evidence.

Proposed repository location: `tools/reader_capacity_integrity_v1/`. Keep the entire addition **outside** `studies/reader-capacity-gate/`: that frozen package's exact-file-set check correctly rejects additional files. No edits to its `MANIFEST.json` or `DERIVATION.json` are proposed.

## Run from a repository checkout

Assuming these new files were placed in the proposed directory:

```sh
# Preserve and run the original saved-output audit.
python -B studies/reader-capacity-gate/verify_package.py

# Add structural checks, independently anchored to the known historical manifest.
python -B tools/reader_capacity_integrity_v1/validate_structure_v1.py \
  --package-root studies/reader-capacity-gate \
  --expected-manifest-sha256 6ce738fc46e3fffdbfcff2be13be3354a8826f9c8d62e8136cc3bf2655921da5

# Regression suite: expected exit status 0.
python -B tools/reader_capacity_integrity_v1/tests/run_adversarial_tests.py \
  --package-root studies/reader-capacity-gate \
  --mode supplement --report /tmp/reader-integrity-results.json

# Runner safety regression tests: expected exit status 0.
python -B tools/reader_capacity_integrity_v1/tests/test_runner_safety.py \
  --package-root studies/reader-capacity-gate
```

All paths are explicit; no workspace-specific path is embedded in the scripts. Verified with Python 3.12.14 on Linux. No third-party Python modules are required. Other Python/OS combinations were not tested. The original verifier uses OS-native relative-path strings against POSIX manifest names, so native Windows compatibility is not claimed; use a compatible POSIX environment such as WSL. This is a partial source snapshot audit, not a run of the full repository test suite.

## Reproduce the findings

```sh
# Historical behavior characterization: expected exit 0, even though known gaps accept bad fixtures.
python -B tools/reader_capacity_integrity_v1/tests/run_adversarial_tests.py \
  --package-root studies/reader-capacity-gate --mode characterize

# Deliberately red contract test against the unchanged verifier: expected exit 1.
# It requires rejection of all 46 malformed variants and exposes the 28 known acceptances.
python -B tools/reader_capacity_integrity_v1/tests/run_adversarial_tests.py \
  --package-root studies/reader-capacity-gate --mode require-rejection
```

These three modes have different purposes. Every mode also asserts the historical acceptance/rejection result independently, so an accidental early hash failure cannot masquerade as coverage of a supplemental structural check. A green characterization run means the recorded historical behavior was reproduced, **not** that the historical validator rejected every bad fixture. `supplement` means the composition of the original and supplemental checks passed the suite, not that either component independently implements every check.

## Integrity boundary

Most negative fixtures deliberately refresh hashes in their **temporary copies** of `DERIVATION.json` and `MANIFEST.json`. Without this step, a changed file would fail the existing hash layer before the mapping/evidence check being tested. This simulates a malformed repackaging whose checksums were recomputed; it does not demonstrate undetected edits to a package authenticated by an unchanged external manifest hash.

The duplicate-manifest-entry fixture is different: the manifest is not self-hashed, so an extra identical entry is accepted without refreshing any other file. A separately trusted manifest hash rejects that edit too.

A self-consistent manifest is not an authenticity proof. Obtain the expected manifest hash from a trusted immutable revision, not from the candidate package being examined. The optional external hash check is deliberately omitted for adversarial temporary copies so structural checks are actually exercised. The original executable verifier and scoring source are pinned before the test runner executes them; this is not a general-purpose sandbox for untrusted Python packages.

## Scientific boundary

Evidence checks concern textual containment, valid offsets, unique identities, required coverage, and agreement between retained records. They cannot determine whether a span supports a proposition, whether a gloss is correct, whether a rationale is persuasive, whether an AI rating is true, or whether a person understood the output.

The accepted scope-control fixture changes a nonempty final rationale without changing any decision or aggregate. Both validators accept it after test-only resealing. A second positive control reorders JSON object members without changing their mapping; it is also accepted. That is intentional: the suite does not invent a semantic-quality oracle.

One actual output-only record, O007/Q01, is unanswerable and legitimately has an empty evidence span. The supplement preserves that case; it requires nonempty evidence only when an output-only answer is labeled `answerable`. This is not a blanket rule that every empty span is invalid.

No new performance metric, efficacy claim, human-comprehension result, training replay, or revised adjudication is produced. Further structural completeness, schemas, provenance authentication and independent semantic review remain separate work.
