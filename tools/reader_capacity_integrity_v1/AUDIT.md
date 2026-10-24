# Validation coverage audit

## Evidence and conclusion

The retained scientific package contains 59 files. Its manifest SHA-256 is `6ce738fc46e3fffdbfcff2be13be3354a8826f9c8d62e8136cc3bf2655921da5`; its verifier SHA-256 is `999af822e30a062ccfd31dd298fc88db5c3eddf522e1b05baafd153789f28845`. These identify the actual bytes tested, rather than relying only on a branch name. The package was acquired at commit `66c97db6c78fb62ef42739acc2ff67ba6ae84b80` and remained unchanged at `6dd1c8fb2544f55058f5fb6b9ba2af8938d06145`: retained publication receipts record only four changed documentation files. `SOURCE_IDENTITY.json` lists every retained file hash.

The unchanged package passes both its historical audit and supplemental structural checks. Its reported gate still fails at 14/24 fluent/truth-complete and 3/24 all-concept-supported outputs. The unchanged external manifest anchor protects the exact historical identity. The new findings concern what additional logical checks are or are not enforced **if a candidate package is deliberately reserialized and rehashed**.

## Findings in the historical verifier

Line numbers below refer to the pinned `studies/reader-capacity-gate/verify_package.py`, not to a moving branch.

### 1. Duplicate collapse and count-only coverage

Lines 25 and 33 build sets/dictionaries without separately rejecting duplicate entries. Lines 102–112 do the same for packet and crosswalk records, then check only 24 row/dictionary entries and 74 visited questions. Membership of a question ID is checked, not complete one-to-one coverage of every expected question.

Demonstrated acceptances include a duplicate manifest entry; appending a duplicate source, output-only packet or crosswalk row; replacing output-only rating O002 with O001 while retaining 24 rows and 74 questions; and replacing one question with a duplicate question in the same row. The unchanged baseline does not contain these defects.

Supplement: require exactly the expected unique output identities and question identities; compare the actual question-to-proposition map. JSON object-member order remains immaterial. Array order is retained where the historical frozen schema uses it.

### 2. Missing source-proposition and question linkage

Lines 40–44 compare top-level proposition-ID lists but do not establish that every listed proposition has a corresponding retained proposition record. A removed source proposition passes once checksums are refreshed. Line 110 checks only question-ID membership, not whether the crosswalk names an existing required proposition, or whether rated question text matches the packet/source question.

Lines 49–62 do not check initial ratings' source/profile linkage. Final ratings receive this check via `source.scoring.aggregate`, but initial-A rows with unrelated source/profile IDs still pass.

Supplement: compare exact source proposition coverage; check every rating phase's source/profile linkage; compare question IDs, crosswalk values and retained question strings. These are identity checks, not a judgment of whether the question is scientifically appropriate.

### 3. Span containment is not sufficient schema validation

Python accepts the empty string as a substring of any string, and slicing accepts negative offsets. Lines 44, 79–80 and 110 consequently accept selected empty source/output evidence and negative source offsets when their literal comparisons happen to succeed.

Supplement: source evidence uses integer offsets with `0 <= start < end <= len(source)` and nonempty matching text. Reconciliation output evidence must be nonempty and present. Output-only evidence is nonempty when `answerability == 'answerable'`; the enum is first checked against the retained answerable/unanswerable/uncertain labels. The valid empty span at O007/Q01, labeled unanswerable, is preserved.

Neither validator can infer semantic support from substring inclusion. The positive rationale-replacement control makes this limitation observable.

### 4. Reconciliation totals do not prove complete records

Lines 73–87 require 170 rows, five truthy disagreement flags and valid referenced final statuses/spans. They do not require every expected `(output_id, category, item_id)` exactly once. Duplicating an agreed decision over another agreed decision preserves the cardinality and disagreement total and is accepted. The 24 unsupported-assertion review rows are only counted at line 73.

The historical verifier also does not tie the embedded initial/independent records to the separately retained initial snapshots, the five-item disagreement-detail list to the complete decision table, the unsupported-review spans/decisions to the corresponding retained records, or several displayed summaries to their actual counts.

Supplement: require the exact 170 unique comparison keys and 24 unique review IDs; compare retained initial records, disagreement flags/details, full review strings and assertion records; compare declared counts and the final reconciliation summary to the retained aggregate. Equality of retained records is a provenance-consistency check. It is not proof that two adjudicators were independent, that differently worded rationales are semantically equivalent, or that the final choice is correct.

### 5. JSON parser permits ambiguous or non-standard data

Line 20 uses ordinary `json.loads`. Duplicate object keys silently keep the last value, and Python accepts non-standard `NaN`/`Infinity` constants. A duplicate output-ID object key or a non-finite numeric field is accepted after resealing.

Supplement: reject duplicate object keys with `object_pairs_hook` and non-standard constants with `parse_constant`. This is a bounded strictness improvement, not a complete formal schema for every historical JSON field.

## Existing safeguards reproduced successfully

The suite confirms existing rejection of changed unresealed evidence, missing/unlisted package files, malformed JSON, missing/null case mappings, duplicate frozen case IDs, changed output linkage, invalid source/final/phase-1 evidence, missing passing evidence, duplicate final ratings, changed aggregate denominator/gate, unknown phase-1 question IDs, and changed reconciliation source spans/final statuses. The arithmetic implementation and historical verifier were not modified.

## Independent review and corrections

A separate reviewer independently reproduced the initial 39-case result and probed additional structural relationships. That review found a test-runner output-path hazard: a caller-specified report inside the source package could be written after the preservation comparison. It was demonstrated only against temporary copies, then fixed by rejecting resolved report paths inside the historical root before execution. Three runner-safety regression tests now protect existing evidence, prevent adding an unlisted report, and reject unexpected source files before package code runs.

The review also found the object-order false positive, initial linkage/question/enum gaps, permissive numeric parsing and unchecked summary fields. Nine new fixtures first failed (40/49 passed), then passed after the minimal structural corrections. Every composed test now independently asserts the expected historical-layer outcome, so an accidental early rejection cannot hide failure to exercise the intended supplement.

Saved red receipts are intentionally failed tests recorded before the fixes; they are not current test failures. The current green receipts and final review status are listed in `verification/summary.json`.

## Limits and smallest integration

Install only this versioned supplemental directory outside the frozen study, run the original verifier unchanged, then the supplemental script and tests. Do not add files inside the study or refresh its manifests. Do not retrofit these checks into historical claims about what the original release had already validated.

The suite is finite and package-specific. It does not check every optional metadata field, fully formalize every retained JSON schema, reconstruct the original execution archive, authenticate private custody or adjudicator independence, rerun training, or verify semantic judgments. A trusted external manifest hash establishes exact identity, not scientific truth. If a later study schema or expected package identity changes, introduce a new version and re-review its scope rather than silently changing these pins.

Execution was Python 3.12.14 on Linux. Native Windows is not claimed because the frozen verifier uses OS-native path separators. Only the complete saved capacity package and these new tools were run; the surrounding repository snapshot is partial, so no full-project test-suite pass is claimed. No source files, models, credentials, external services or historical manifests were changed.
