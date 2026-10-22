# Pretrained model fine-tuning pilot

This source checkpoint implements actual LoRA adaptation of the pinned Apache-2.0 SmolLM2-135M-Instruct model with an optional auxiliary explicit-register head and optional answer-consistency loss. At this checkpoint, source/unit tests and safe model loading are verified; the full pilot has not yet run. Do not confuse this with the existing NumPy experiments or claim improved language quality.

Read [the locally recorded pre-run protocol](PROTOCOL.md). This small synthetic, single-seed pilot is designed to test execution and expose factual degradation, not prove human understanding.

## Dependencies and model

Use an isolated Python 3.12 environment. `requirements.lock.txt` records the tested installation. The CPU Torch wheel came from the official https://download.pytorch.org/whl/cpu index; remaining wheels came from the default PyPI registry. No dependency installation is performed by the experiment script.

Download the files named in `model_manifest.json` from the public model repository at its exact revision into a directory of your choice. Only safetensors weights are used, no pickled training arguments or model-provided Python. The manifest includes hashes and original byte sizes. Its README hash records license provenance; the model card's Apache-2.0 declaration is not a claim that this project relicenses the pretrained model. Model metadata and license reference are in the protocol.

`MODEL_DIR` below is your downloaded directory. Set `HF_HOME` to a writable local cache if needed. Execution is strictly local-only and refuses mismatched model-file hashes.

```sh
export PYTHONPATH=pretrained
export MODEL_DIR=/your/local/model-directory
python -m unittest discover -s pretrained/tests -v
python pretrained/pilot.py --model-dir "$MODEL_DIR" --out pretrained/local-results
sha256sum -c pretrained/MANIFEST.sha256
```

The command compares untouched pretrained weights and three LoRA arms. Assistant-only labels mask prompts/padding. All three register variants share a record in each training batch. No gold answer is used to repair generation. Test outputs are never used for checkpoint selection. A validation-selected zero-update adapter can legitimately win. Full precision CPU computation uses two threads and cooperative 30-minute budget checks (not a hard wall-clock cap; one load, update, or generation can exceed the deadline before the next check); no paid compute, account creation or API credentials.

Source-checkpoint tests use the actual pinned tokenizer plus a small randomly initialized Llama only for adapter/backprop mechanics. That random model is not substituted for the actual pretrained pilot. Full experiment results will be a later checkpoint with raw generations and adapter provenance.

Output safety: every invocation requires a new, nonexistent output directory. Reusing the same directory, including with --arm, is refused so stale artifacts cannot mix.
