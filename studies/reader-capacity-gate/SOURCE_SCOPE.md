# Scientific source scope

This is a narrowly reproducible saved-output semantic audit. It deliberately does not pretend that environment-specific execution wrappers form a portable historical model runner.

- `source/scoring.py` is an explicitly labeled public derivative of the executed aggregation module. Only its final scientific-scope string changes; the conservative denominator, statuses and 18/12 floors do not.
- `source/schedule.py` is a byte-for-byte snapshot of the executed fixed-order/input-checking module. It supplies the aggregation module's small validation helper without loading a model.
- `source/optimizer.py` and `source/custody.py` are byte-for-byte mathematical/audit and custody kernel snapshots. They are supplied for inspection. The optimizer audit needs the original generated tensor payloads and pinned torch environment, which this public package does not distribute; the public verifier does not import or run them.
- `method-inspection/core.py`, `method-inspection/serialization.py` and `method-inspection/run.py` are byte-for-byte original loss/controller, prompt-serialization and A-orchestration source snapshots. The core contains non-A objective implementations, but those arms were not run in this attempt. They are inspection-only, not a public runnable pipeline.
- `method-inspection/decoder_public_derivative.py` removes exactly one non-executable caller-control sentence from an original docstring; executable code is unchanged. Its original dependency imports remain unresolved in this narrow release. It is explicitly a derivative, not the executed original.
- `source/__init__.py` and `verify_package.py` are new public verification support, not executed historical model code.

The historical runner imported authenticated ancestor modules and environment-specific model/data/review paths. Its original import/strict-validator wrappers are omitted rather than silently rewritten and called originals. The complete training dataset and the entire gradient/optimizer evidence archive are outside this minimal semantic-audit release. The model's base weights are external and are not redistributed. This means public inspection can reproduce the reported semantic count arithmetic, but cannot fully replay training/generation or independently re-establish all engineering evidence from this package alone.

The original frozen source identity and original input identity remain recorded in `method/method.json`. `DERIVATION.json` binds each released source file to its original bytes or documented transformation. A later portable full-model implementation would be a new derivative, not the executed original.
