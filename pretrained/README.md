# Pretrained model fine-tuning pilot

This source checkpoint implements actual LoRA adaptation of the pinned Apache-2.0 SmolLM2-135M-Instruct model with an optional auxiliary explicit-register head and optional answer-consistency loss. The bounded actual pilot has now completed; see [measured results](results/RESULTS.md), raw generations, and saved-adapter replay verification. Do not confuse this with the existing NumPy experiments or claim improved language quality.

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

Source-checkpoint tests use the actual pinned tokenizer plus a small randomly initialized Llama only for adapter/backprop mechanics. That random model is not substituted for the actual pretrained pilot. The complete results checkpoint includes raw generations, all training traces and adapter provenance. It does not establish human comprehension or robust natural-language improvement.

Output safety: every invocation requires a new, nonexistent output directory. Reusing the same directory, including with --arm, is refused so stale artifacts cannot mix.

## Saved adapter replay

The adapter configuration names the canonical base model and immutable revision rather than the original local path. Weight bytes are unchanged. Download the pinned base files described above, then use the packaged safetensors adapters:

```sh
python pretrained/replay.py --model-dir "$MODEL_DIR" --run-dir pretrained/results --adapter-dir pretrained/adapters --out /your/new-replay-receipt.json
```

You can alternatively pass a fresh local pilot output directory as `--run-dir`. The replay checks all 48 held-out outputs per trained arm; it does not retrain. It refuses an existing receipt path. Full training also requires a fresh output directory.

The pretrained model and derived adapter weights are Apache-2.0; see [license](APACHE-2.0.txt) and [adapter attribution](adapters/README.md). New project code and synthetic data remain MIT licensed.
