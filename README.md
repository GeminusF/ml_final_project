# Ensemble Methods: Boosting vs. Bagging

From-scratch NumPy implementations and reproducible experiments for the Machine Learning final project. The project compares how boosting reduces systematic error by focusing successive learners on difficult observations, while bagging reduces variance by averaging decorrelated bootstrap trees.

## Implemented scope

- Weighted multiclass CART `DecisionTree` with Gini/entropy, midpoint search, per-node feature sampling, probabilities, depth/leaves, and impurity importance.
- `AdaBoostClassifier` with discrete SAMME, staged predictions, early stopping, stable sample weights, and bonus SAMME.R.
- `RandomForestClassifier` with deterministic bootstrap samples, OOB indices/score/coverage, per-node feature sampling, aligned class probabilities, and Windows-safe multiprocessing.
- Centered PCA, K-Means with k-means++ and deterministic empty-cluster recovery, and DBSCAN with core/border/noise semantics.
- Bonus binary log-loss Gradient Boosting, deterministic t-SNE comparisons, and an executable interactive notebook.
- Seven required experiment families, machine-readable results, report-ready figures, and a reproducibility manifest.

scikit-learn estimators are used only for the explicitly permitted reference baselines and t-SNE/ARI utilities. The assessed tree, AdaBoost, Random Forest, PCA, K-Means, DBSCAN, and Gradient Boosting implementations are project code.

## Repository layout

```text
README.md                       Setup, methods, commands, and audit notes
requirements.txt               Fully pinned Python environment
download_data.sh               Dataset acquisition and verification entry point
data/dataset_audit.json          Verified shapes, distributions, missingness, and hashes
src/trees/                      Decision Tree, AdaBoost, Random Forest
src/boosting/                   Gradient Boosting bonus and compatibility import
src/unsupervised/               PCA, K-Means, DBSCAN
src/experiments/                Dataset registry, configs, runners, plots, manifest
src/metrics/                    Classification and bias-variance metrics
src/utils/                      Leakage-safe preprocessing and oversampling
tests/                          Unit, edge-case, reproducibility, and smoke tests
notebooks/exploration.ipynb     Executed interactive bonus notebook
figures/                        Generated report-ready plots
results/                        CSV/JSON/NPZ experiment evidence and manifest
report/report.tex               Final 7-page assembly; core/extension TeX alongside
report/report.pdf               Compiled report
presentation/presentation.pptx  Editable 11-slide defense deck
presentation/presentation.pdf   Matching 11-page submission deck
contribution_report.pdf         Four-member contribution declaration
```

## Environment setup

Python 3.12 is the validated interpreter.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All paths are resolved relative to the repository root. No machine-specific absolute path is embedded in project code.

## Datasets and provenance

The locked dataset registry records source, version, shape, class counts, missingness, and local SHA-256 hashes where source files are present. The verified snapshot is [`data/dataset_audit.json`](data/dataset_audit.json): Adult has 48,842 rows, 14 raw features, and 6,465 missing cells; the deterministic 10,000-row Covertype subset has 54 features, all seven classes, and a realized 0.47% minority class.

| Dataset | Role | Source/version | Expected checks |
|---|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | Balanced binary baseline and bias-variance study | scikit-learn bundled copy of the UCI dataset | 569 rows, 30 features, two classes |
| Adult / Census Income | Binary mixed-type data with missing categorical values | UCI Adult; OpenML version 2 fallback | unknown categories accepted; imputation/encoding fit on training data only |
| Covertype | Seven-class, high-dimensional, severely imbalanced task | UCI Covertype via scikit-learn cache or local file | 54 features; deterministic stratified subset; realized minority frequency must be at most 1% |

Sources: [Breast Cancer loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), and [UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype). Dataset creators and UCI must be attributed in the report and repository; the project redistributes no raw data because the downloaded files and sklearn cache are Git-ignored. Local source hashes are recorded in `data/dataset_audit.json` and each full-run manifest.

Acquire or verify all data:

```bash
bash download_data.sh
```

The loader checks the actual realized distributions. It does not assume that a proposed sample is severely imbalanced; the deterministic Covertype sample is enlarged until a minority class is at most 1%, or loading fails explicitly.

## Leakage-safe protocol

Every holdout or CV fold is split before preprocessing. Numeric medians, means, and scales; categorical modes and one-hot vocabularies; PCA; and oversampling are learned from the training partition only. Unknown Adult categories map to an all-zero one-hot block. The deterministic random oversampler raises classes below 5% of the original training-fold size without downsampling majorities. Test labels and test features never influence fitting, selection, scaling, or threshold choices.

## Reproduce the project

The binding PDF command runs the complete configuration:

```bash
python src/experiments/run_all.py
```

Interactive terminals show at most two live green progress lines: an overall
slash-filled bar and a current-operation bar. They include the dataset,
experiment, fold/replicate/tree/restart detail, elapsed time, estimated time
remaining, estimated finish time, and a rotating `| / - \\` activity marker.
The estimate is intentionally labelled because work differs by dataset. In
redirected output the runner emits stable ANSI-free progress lines instead.
Use either control when needed:

```bash
python src/experiments/run_all.py --no-color
python src/experiments/run_all.py --no-progress
```

The `NO_COLOR` environment variable also disables ANSI color without disabling
progress. A completed interactive run retains the full green overall bar before
printing its artifact summary.

The full profile includes five-fold CV, 100-estimator comparisons, five label-noise replicates at 0/5/10/20%, 100 shared bootstrap replicates, estimator and depth scaling, ten K-Means restarts for each `k=1..10`, PCAs minimum 90% variance threshold, and all bonuses except GitHub Actions.

For architecture checks only:

```bash
python src/experiments/run_all.py --profile quick --datasets breast_cancer
```

The quick profile is never a substitute for final experimental claims. It is deliberately smaller and marks itself as `profile=quick` in the manifest.

Outputs are deterministic CSV/JSON/NPZ files under `results/` and PNG figures under `figures/`. `results/manifest.json` records configuration, seed, package versions, dataset metadata, output paths, runtime, and the Git commit when available. `results/runtime.json` additionally records planned/observed work units, status, and elapsed seconds for every phase. Timing and display state never enter model seeds or numerical artifacts. Numeric outputs should match across repeated runs with the same environment; timestamps and runtime fields may differ.

### Runtime and resources

The quick Breast Cancer profile takes roughly 15-30 seconds on the validated machine. The full suite is intentionally much heavier because it fits thousands of custom trees; allow several hours, at least 8 GB RAM, and several hundred MB of disk cache. Run the full suite from a terminal that can remain active. Random Forest uses deterministic child seeds and may use spawned workers; if Windows policy denies process pipes it emits a warning and falls back to identical sequential fitting.

## Quality gates

```bash
python -m pytest -q --cov=src --cov-report=term-missing --cov-fail-under=85
python -m ruff check src tests
python -m mypy src
python -m compileall src
python -m jupyter execute notebooks/exploration.ipynb --inplace --timeout=120 --kernel_name=python3
```

The internal target is at least 85% overall coverage and at least 75% for every core algorithm module, above the briefs 60% deduction threshold. Tests cover mathematical hand cases, weighted splits, arbitrary labels, zero weights, early stopping, numerical clipping, OOB complements, class alignment, sequential/parallel equivalence, PCA orthogonality, K-Means empty clusters, DBSCAN boundary semantics, leakage, fold-local oversampling, experiment schemas, and two-level reproducibility.

## Experiment evidence

1. Baseline: custom unpruned tree, custom stump, and sklearn reference on an 80/20 split; the custom/reference tree accuracy gap must be at most two absolute percentage points.
2. AdaBoost scaling: staged training/test error from 1 through 200 estimators, including both endpoints.
3. Random Forest scaling: estimator count through 200 and depth 1 through 20, with test accuracy, OOB accuracy, and OOB coverage.
4. Head-to-head: single tree, AdaBoost, Random Forest, and sklearn RF reference on identical stratified five-fold splits.
5. Noise robustness: clean plus 5/10/20% nested training-label corruption, five deterministic replicates, clean test data.
6. Bias-variance: Breast Cancer, 100 shared bootstrap index sets, fixed test set, stored probabilities, and Brier decomposition residual.
7. Unsupervised: PCA scree/cumulative variance, minimum 90% PCs, K-Means elbow, DBSCAN k-distance policy, PCA scatter plots, ARI, and noise fraction.

Bonus outputs compare Gradient Boosting with AdaBoost on binary data, SAMME with SAMME.R, and PCA with t-SNE. t-SNE plots are interpreted only as local-neighborhood visualizations; global distances and cluster areas are not treated as metric evidence.

## Troubleshooting

- Network unavailable: place `adult.data`, optional `adult.test`, and `covtype.data` or `covtype.data.gz` in `data/`, then rerun.
- Multiprocessing denied: use the default deterministic fallback or set `n_jobs=1` in an experiment configuration.
- Jupyter secure-file error on managed Windows: set `IPYTHONDIR`, `JUPYTER_CONFIG_DIR`, and `JUPYTER_RUNTIME_DIR` to writable repository-local directories before execution.
- LaTeX unavailable: use the bundled compile script described in `validation_report.md`, or install a standard TeX distribution.

## Team ownership

Percentages represent the planned substantive contribution balance; each member remains responsible for genuinely completing, reviewing, and defending the work assigned to them.

| Team member | Contribution share | Primary responsibilities |
|---|---:|---|
| Milana Karimova | 24.11% | Repository foundations, CI, preprocessing, evaluation metrics, weighted Decision Tree, Gradient Boosting, data acquisition guidance, README, project guide, and contribution report |
| Sharaf Feyzullayev | 25.09% | Dataset registry and audit, AdaBoost, SAMME.R, DBSCAN, package exports, experiment utilities, and report components |
| Nijat Agayev | 25.95% | Random Forest, OOB behavior, bagging exports, PCA, K-Means, progress and ETA system, validation report, slide guide, and defense guide |
| Farah Feyzullayev | 24.85% | Experiment runner, `run_all.py`, smoke tests, full-run results and figures, interactive notebook, final report, and presentation |


## Academic integrity and defense

Substantial AI assistance was used to help scaffold code, tests, documentation, and artifact generation. The team remains responsible for checking every formula, validating every experiment, accurately reporting actual human contributions, and explaining every submitted line during defense. AI assistance is disclosed in the report and contribution materials. No authorship or Git history may be fabricated; each member must commit genuine implementation, review, experimentation, writing, or defense work under their own Git identity.