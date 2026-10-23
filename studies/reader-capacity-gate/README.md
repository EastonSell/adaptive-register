# Reader-specific rewrites: the capacity gate failed

**The run passed its engineering checks but missed both frozen semantic floors.** After one fixed ordinary-supervised-fine-tuning attempt, 14 of 24 outputs were fluent, task-following and truth-complete; at least 18 were required. Only 3 of 24 explained every concept stipulated unfamiliar to that reader; at least 12 were required.

This is a small, exposed **development-set feasibility check**, not a reader study. The source material, scoring ledger and judgments were AI-produced. No human comprehension was measured, and no benefit of an auxiliary head, consistency objective or other non-A comparison arm was established. The result does not justify moving to the planned B/C/S efficacy comparison under the unchanged gate.

## What the two checks ask

1. **Keep all the facts.** A fluent sentence is not enough: participants, quantities, negation, conditions, timing and uncertainty must all survive the rewrite. For example, changing “only after the dock closes” to “after the dock closes” loses an important restriction.
2. **Explain what this reader does not know.** An accurate, plain-language replacement can pass without keeping the technical term. Merely naming it, dropping it, or putting a correct definition on the wrong participant does not pass.

The two floors are separate: the successful outputs need not be the same. Unsupported added claims are also reported, but were not retrospectively turned into a third numeric gate.

## Read or audit

- [Methods, results and limitations](METHODS_AND_RESULTS.md): the fixed recipe, all diagnostic counts and what the result does and does not show.
- [Verify the saved result](REPRODUCE.md): a standard-library-only check of all 24 outputs, evidence spans, ratings and aggregation. No model is loaded.
- [Provenance](PROVENANCE.md): the initial judgments, five reconciled differences, public metadata redactions and exact-copy versus derivative distinctions.
- [Source scope](SOURCE_SCOPE.md): scientific source snapshots and why this release is not an end-to-end reproduction of the historical training run.
- [Frozen rubric](evaluation/rubric.md), [source/profile/proposition packet](evaluation/source-aware-packet.json), [all output JSON files](outputs), [final AI ratings](adjudication/source-aware-final.json), and [aggregate result](results/aggregate.json).

This package is an explicitly labeled **public semantic-audit derivative**, not the original execution archive. The output JSON files and several frozen evaluation artifacts are byte-for-byte copies. Evaluator identity/coordination metadata is replaced with stable public roles; semantic statuses, evidence and rationales are preserved. Every copied or transformed file has an original/public hash mapping in [DERIVATION.json](DERIVATION.json). The original private files were not edited.
