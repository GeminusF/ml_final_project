#!/usr/bin/env bash
set -euo pipefail

mkdir -p data

download_if_missing() {
  local url="$1"
  local output="$2"
  if [[ ! -f "$output" ]]; then
    curl --fail --location --retry 3 "$url" --output "$output"
  fi
}

download_if_missing \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" \
  "data/adult.data"
download_if_missing \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test" \
  "data/adult.test"
download_if_missing \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz" \
  "data/covtype.data.gz"

python -c "from pathlib import Path; from src.experiments.datasets import load_project_datasets; bundles=load_project_datasets(Path('data'), names=('breast_cancer','adult','covertype'), covertype_size=10000, random_state=42); print({name: bundle.metadata()['shape'] for name, bundle in bundles.items()})"
