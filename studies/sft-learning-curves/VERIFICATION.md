# Publication verification

A new archive was extracted into an empty directory and verified independently of the populated source checkout. Using the previously installed pinned Python environment, all 17 bounded-source tests and all 23 closure-source tests passed. Every pinned distribution version matched the lock. No installation, model load, training or inference was performed.

The independent standard-library evidence audit passed in that extraction: source/run/checkpoint hashes, expected response coverage, strict parser and metric recounts, token-weighted LM loss, training order/exposure, 326-row common prefix and 120 tensors at each of steps 24 and 96. Original frozen artifacts were also audited before derivation: 36/37 source files and 25/42 run-manifest entries for bounded/closure respectively.

Original-source verification and public-derivative verification are distinct. Regenerated public manifests cover sanitized metadata, whereas original hashes in SOURCE-IDENTITY.json and DERIVATION.json identify historical artifacts. Numerical artifacts remain unchanged. The source supervision bug is retained, not fixed. Closure numerical completion does not override its incomplete accounting-failure status.
