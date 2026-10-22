# Post-run release tooling

Added after pretrained pilot commit e906cb1517a8efbdc4da09002abf5c26c2e9a091. No training code, protocol, dependency pins, raw results or adapter weights changed. Thirty frozen source/provenance/result/adapter files were checked byte-for-byte against the prior release staging copy.

## Checks

- Aggregate optional run: 17 NumPy + 8 tooling + 9 pretrained tests pass (34 total), with the previously downloaded pinned tokenizer/model.
- Default aggregate run: 17 NumPy + 8 tooling tests pass (25 total); optional pretrained tests are explicitly not requested, not silently skipped.
- Download-helper tests use tiny in-memory local fixtures, covering verified streaming, cached reuse, refusal to overwrite a mismatched existing file, checksum rejection without publishing corrupt bytes, unsafe paths/unpinned revisions, and network-free/non-mutating verify-only failure.
- Offline verify-only checked all nine existing model files against the frozen manifest. No model redownload or retraining occurred.
- The eight new tests initially failed because the helpers did not exist. A later verify-only regression first failed because a missing directory was created, then passed after removing that side effect.
- Existing runtime passes `pip check`; `pip install --dry-run --no-index -r pretrained/requirements.lock.txt` confirms every frozen requirement is already satisfied. A second clean dependency installation was not performed.

## Installation scope

The copy-paste CPU recipe in pretrained/README.md targets Linux x86_64 / Python 3.12. It installs the CPU Torch wheel from the official PyTorch index first, then the unchanged lock from PyPI. The lock is not a standalone PyPI-only installation recipe. Other platforms are not validated here. The fetch helper uses standard HTTPS validation and follows the public model host's redirects, with no credentials. Byte counts and SHA256 hashes are checked before atomic no-overwrite publication. Partial successfully verified files may remain after interruption and are reusable. Local hard-link support is required for atomic publication.
