# Verify first; reproduction is a separate experiment

Use Python 3.12 on Linux. The original runtime versions are recorded in each run's identity.json and the complete environment lock is source/vendor/requirements.lock.txt. Model-free tests still require torch and safetensors; no model download or training is needed for those tests.

From the repository root, with the pinned environment available:

```sh
cd studies/sft-learning-curves
(cd bounded/source && python -m unittest discover -s tests -v)
(cd closure/source && python -m unittest discover -s tests -v)
python verify_package.py
python audit_evidence.py --earlier-source bounded/source --earlier-run bounded/run --closure-source closure/source --closure-run closure/run --output /tmp/sft-audit.json
```

To create an environment separately, use an official Python/PyTorch package source, install CPU torch 2.5.1+cpu from https://download.pytorch.org/whl/cpu, then install the exact versions in bounded/source/vendor/requirements.lock.txt. This publication verified the existing pinned environment without installing packages. Package availability is not guaranteed indefinitely; the lock is an execution identity, not a claim of fresh resolver testing.

For an optional new experiment, obtain only the files listed in source/vendor/model_manifest.json from https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac and verify every SHA-256 and byte size. Respect the model's Apache-2.0 license. Do not use a moving branch or substitute model revision.

Create a separate execution copy, not a modification of retained evidence:

```sh
python prepare_reproduction.py closure /absolute/path/to/model /absolute/path/to/new-source-copy
cd /absolute/path/to/new-source-copy
python supervise.py --out /absolute/path/to/new-exclusive-results --approved-freeze-sha256 "$(sha256sum FREEZE.json | cut -d' ' -f1)"
```

The historical flag name is an exact-source hash guard. It does not express independent scientific approval. The preparation helper verifies model files, copies the public source, substitutes only the absolute model directory and regenerates its source manifest. This is a new derivative and any resulting run is new evidence, not a reconstruction of missing historical telemetry. Do not invoke runner.py directly. The retained supervisor intentionally remains unfixed: it may reproduce the observed accounting failure. Its sampled limits are not instantaneous resource guarantees.
