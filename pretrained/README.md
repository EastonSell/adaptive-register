# Pretrained model fine-tuning pilot

This source checkpoint implements actual LoRA adaptation of the pinned Apache-2.0 SmolLM2-135M-Instruct model with an optional auxiliary explicit-register head and optional answer-consistency loss. The bounded actual pilot has now completed; see [measured results](results/RESULTS.md), raw generations, and saved-adapter replay verification. Do not confuse this with the existing NumPy experiments or claim improved language quality.

Read [the locally recorded pre-run protocol](PROTOCOL.md). This small synthetic, single-seed pilot is designed to test execution and expose factual degradation, not prove human understanding.

## Dependencies and model

The following post-run setup recipe targets Linux x86_64 with Python 3.12, the pilot platform. Run from the repository root. The frozen `requirements.lock.txt` includes `torch==2.5.1+cpu`, which is not supplied by the default PyPI index alone. Install that exact wheel from the [official PyTorch CPU index](https://pytorch.org/get-started/previous-versions/) first, then install the remaining pinned requirements. The lock's original bytes and version pins are preserved.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1+cpu'
python -m pip install --only-binary=:all: -r pretrained/requirements.lock.txt
python -m pip check
export HF_HOME="$PWD/.cache/huggingface"
export MODEL_DIR="$PWD/.local-models/SmolLM2-135M-Instruct-12fd25f"
python pretrained/fetch_model.py --destination "$MODEL_DIR"
python pretrained/fetch_model.py --destination "$MODEL_DIR" --verify-only
python tooling/run_tests.py --pretrained --model-dir "$MODEL_DIR"
```

The standard-library fetch helper downloads only public data files listed in `model_manifest.json` at the immutable model revision, verifies every SHA256 and byte count, then atomically publishes verified files. It sends no credentials, executes no downloaded code and never loads pickle. An existing mismatched file is refused rather than overwritten. Verified existing files are reused, so an interrupted download can be restarted safely; `--verify-only` never uses the network. Model/data cache folders are ignored by Git. Allow roughly 275 MB for model assets plus the separate Python runtime. No download occurs during tests: helper tests use tiny local byte fixtures.

The model card's Apache-2.0 declaration is not a claim that this project relicenses the base model. Full provenance and exact file sizes are in the manifest. The fresh installation recipe is documented from the official package sources; this tooling checkpoint reuses the already-installed pilot runtime rather than claiming a second clean installation or redownload.

Keep `MODEL_DIR` exported from the setup block above. Set `HF_HOME` to a writable local cache if needed. Execution is strictly local-only and refuses mismatched model-file hashes.

```sh
export PYTHONPATH=pretrained
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
