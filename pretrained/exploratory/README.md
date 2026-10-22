# Exploratory fourth instruction label and base repeat

Pre-run source checkpoint only. No new model generations or outcomes are included yet. Read PROTOCOL.md before interpreting this follow-up. Frozen original training and results remain untouched.

With the previously documented pretrained runtime and pinned model assets:

```sh
PYTHONPATH=pretrained:pretrained/exploratory python -m unittest discover -s pretrained/exploratory/tests -v
sha256sum -c pretrained/exploratory/MANIFEST.sha256
# Run only after this source/protocol checkpoint is independently reviewed and locally frozen:
python pretrained/exploratory/probe.py --model-dir "$MODEL_DIR" --out /your/new/exploratory-output
```

The runner verifies frozen inputs and model files, then performs inference only. `inputs.json` pins original source, saved results, adapter metadata/weights, the new protocol/source and exact prompts. The eight unit tests are independent of model inference. They are a separate exploratory suite in addition to the existing 34 aggregate release tests.

Execution timing was prospectively amended in AMENDMENT.md before any new generations: remote publication may follow the local reviewed freeze. This is not public preregistration. Per-branch JSONL files preserve completed responses during interruption.
