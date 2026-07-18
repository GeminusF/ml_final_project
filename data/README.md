# Dataset cache

This directory is intentionally version-control friendly: large raw datasets and caches are ignored, while this provenance note is tracked.

Supported local files:

- `adult.data` and optional `adult.test` from UCI Adult.
- `covtype.data` or `covtype.data.gz` from UCI Covertype.

Run `bash download_data.sh` to acquire the canonical UCI files. If local files are still absent, the loader can fall back to scikit-learn’s OpenML/Covertype download utilities. Local-file SHA-256 hashes, realized class counts, shapes, missingness, source URLs, and dataset versions are written to `results/manifest.json`. The committed `dataset_audit.json` records the verified 10,000-row Covertype selection and the exact local hashes used during validation.
