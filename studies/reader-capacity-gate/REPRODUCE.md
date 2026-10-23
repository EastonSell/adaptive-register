# Verify the saved evidence; do not confuse this with retraining

From the repository root, use Python 3.12 or another compatible Python 3 standard library:

```sh
cd studies/reader-capacity-gate
python -B verify_package.py
```

No model, API, package installation, GPU, torch import or network access is needed. The verifier reads files and prints a JSON result. It does not edit evidence. It checks the exact published file set and SHA-256 hashes, all 24 source/profile/output mappings, all required evidence spans, 74 output-only questions, the 170 compared source-aware decisions, the five reconciled differences, initial-versus-final diagnostic counts, and the final aggregation at the unchanged 18/24 and 12/24 floors.

Expected final counts are 14 fluent/truth-complete, 3 all-unknown-supported and capacity_gate_passed=false, with all 24 adjudicated. The released [aggregate](results/aggregate.json) is a documented public derivative of the original official aggregate: only its scope prose changed. The public aggregation source likewise changes only that prose; the mathematical counting logic is unchanged. [DERIVATION.json](DERIVATION.json) records original and public hashes.

## What this reproduces

It reproduces the arithmetic from the released final AI decisions and verifies their linkage/evidence within the released data. It does not make new semantic judgments or independently validate whether an AI interpretation is correct. A researcher can inspect or independently re-rate every source/output/profile using the frozen rubric; such new judgments should remain a separate analysis, not overwrite the retained original ratings.

## What this does not reproduce

This is not the original training/generation command, the original strict engineering validator, a gradient-derivative reconstruction, or an end-to-end historical replay. The original execution depended on a chain of environment-specific authenticated ancestors and model/data paths. Those private wrappers, complete training references, model weights, full optimizer/gradient payloads and original command receipts are not bundled. The released scientific kernels support inspection but are not misrepresented as a standalone model runner.

The method and pinned runtime/model revision are recorded in [method/method.json](method/method.json). Those identities specify what was used, not a tested fresh installation recipe or a promise that merely obtaining the base model reproduces this run. Full model reproduction would require a separately prepared, documented source/data/environment package and would create new evidence, not repair this attempt. No such model execution was performed to build or verify this release.
