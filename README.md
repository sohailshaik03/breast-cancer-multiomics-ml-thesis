# Breast Cancer Multi-Omics Survival Classification

An MSc Bioinformatics thesis project.

> **Research and educational use only.** This is a retrospective machine-learning
> exercise on a supplied dataset. It is not a diagnostic or prognostic medical
> device and must not be used to make decisions about patient care.

## Project overview

The thesis, *Development of a Machine Learning-Based Diagnostic Tool for Early
Detection of Breast Cancer using Multi-Omics Data Integration*, investigates
whether molecular features can help classify breast-cancer patient outcomes.
The supplied analysis file, however, predicts the `vital.status` outcome (0/1);
it does **not** classify cancer versus non-cancer or establish early detection.
Keep this distinction in mind when interpreting the thesis title and the
results.

The repository contains the submitted thesis, its supplied tabular dataset, a
Python implementation of four classifiers, and saved result tables, a run log,
and figures. `BRCA.py` is the executable analysis; the thesis describes a
broader research plan and findings than this script implements.

## Repository contents

| Path | Description |
| --- | --- |
| `thesis.docx` | Submitted MSc thesis document |
| `data.csv` | Supplied feature matrix and `vital.status` labels |
| `BRCA.py` | Preprocessing, train/test split, model fitting, evaluation and figure generation |
| `requirements.txt` | Python dependencies |
| `results/model_metrics.csv` | Reproduced evaluation metrics for each classifier |
| `results/run.log` | Console output from the saved-results run |
| `results/model_accuracy_comparison.png` | Accuracy comparison across classifiers |
| `results/*_confusion_matrix.png` | Per-classifier confusion matrices |
| `results/*_roc_curve.png` | Per-classifier ROC curves |

## Dataset

The supplied CSV contains **705 samples**, **1,936 feature columns**, and one
target column (`vital.status`), for 1,937 columns total. The target has two
values:

| `vital.status` | Samples | Proportion |
| --- | ---: | ---: |
| 0 | 611 | 86.7% |
| 1 | 94 | 13.3% |

The feature column prefixes in this file are `rs_` (604 features), `cn_` (860),
`mu_` (249), and `pp_` (223). The CSV does not include a data dictionary that
defines these prefixes or identifies the exact source cohort/accession. The
thesis discusses public repositories including TCGA and GEO, but the supplied
files alone do not establish a precise data provenance or sample-level
permissions. Confirm provenance, reuse terms, and any disclosure/privacy
requirements before redistributing the dataset.

The dataset has no blank cells in the supplied version. The script retains a
`dropna()` step as a basic safeguard, then treats every column except
`vital.status` as a model feature.

## Analysis workflow

`BRCA.py` performs the following steps:

1. Loads `data.csv` and removes rows with missing values.
2. Separates all remaining columns into predictors and the `vital.status`
   target.
3. Scales the complete feature matrix with `StandardScaler`, then creates a
   shuffled 80/20 train/test split with random seed 42. The split is not
   stratified; this run has 564 training samples and 141 test samples (125
   class-0 and 16 class-1).
4. Fits four classifiers with the settings shown in `BRCA.py`:
   - Random Forest (100 trees)
   - Neural Network / MLP (one hidden layer with 100 units; maximum 500
     iterations)
   - Linear Support Vector Machine with probability estimates enabled
   - Gradient Boosting (100 estimators)

   The script fits and evaluates Gradient Boosting twice; the second fit is
   used for the saved metrics and figures.
5. Prints accuracy, per-class precision/recall/F1, and confusion matrices;
   calculates ROC-AUC from predicted probabilities.
6. The checked-in metric table and PNG figures were saved during this run by a
   temporary output-capture wrapper; the source script itself only displays
   plots and does not write these files.

The positive class used for class-1 metrics and ROC analysis is
`vital.status = 1`. The script does not perform the thesis-described LASSO or
statistical feature selection, hyperparameter search, cross-validation,
independent cohort validation, or a separate multi-omics integration
algorithm. The four feature prefixes are simply passed together as model
inputs.

## Reproduced results

The following table summarizes this run. Accuracy alone is not a sufficient
measure here: the dataset is imbalanced, so the recall and F1-score for class 1
are particularly important.

| Model | Accuracy | Class-1 precision | Class-1 recall | Class-1 F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.9007 | 0.7500 | 0.1875 | 0.3000 | 0.4965 |
| Neural Network | 0.8936 | 0.5714 | 0.2500 | 0.3478 | 0.5685 |
| SVM | 0.8440 | 0.2857 | 0.2500 | 0.2667 | 0.5935 |
| Gradient Boosting | 0.8936 | 0.6000 | 0.1875 | 0.2857 | 0.5620 |

Random Forest has the highest accuracy, Neural Network the highest class-1
F1-score, and SVM the highest ROC-AUC on this run. All models miss most
class-1 cases: recall ranges from 0.19 to 0.25. These single-split results are
exploratory and do not demonstrate clinical utility or reliable generalization.

### Accuracy comparison

![Model accuracy comparison](results/model_accuracy_comparison.png)

### Confusion matrices

| Random Forest | Neural Network |
| --- | --- |
| ![Random Forest confusion matrix](results/random_forest_confusion_matrix.png) | ![Neural Network confusion matrix](results/neural_network_confusion_matrix.png) |

| SVM | Gradient Boosting |
| --- | --- |
| ![SVM confusion matrix](results/svm_confusion_matrix.png) | ![Gradient Boosting confusion matrix](results/gradient_boosting_confusion_matrix.png) |

### ROC curves

| Random Forest | Neural Network |
| --- | --- |
| ![Random Forest ROC curve](results/random_forest_roc_curve.png) | ![Neural Network ROC curve](results/neural_network_roc_curve.png) |

| SVM | Gradient Boosting |
| --- | --- |
| ![SVM ROC curve](results/svm_roc_curve.png) | ![Gradient Boosting ROC curve](results/gradient_boosting_roc_curve.png) |

## Results stated in the thesis

The thesis reports these test accuracies: Random Forest **90.07%**, Neural
Network **89.36%**, SVM **84.39%**, and Gradient Boosting **89.36%**. It also
describes ROC-AUC values of approximately 0.50 (Random Forest), 0.57 (Neural
Network), 0.59 (SVM), and 0.56 (Gradient Boosting). Those are thesis-reported
values, not the current run's outputs.

The current script scales before splitting and does not stratify, so test-set
information leaks into preprocessing and the test class balance varies from a
stratified split. The current run's results should not be treated as an
independent or clinical validation. See `results/model_metrics.csv` and
`results/run.log` for the saved outputs.

## Interpretation and limitations

- **Outcome and title differ:** the executable predicts the recorded
  `vital.status`, not early cancer diagnosis. This is an outcome-classification
  analysis, not evidence of a screening or diagnostic tool.
- **Class imbalance:** only about 13% of records have label 1. Accuracy is
  dominated by the majority class; the low class-1 recall means many positive
  cases are missed.
- **High dimensionality:** 1,936 input features are evaluated with 705
  samples. This creates substantial overfitting risk, especially without
  nested validation or feature selection performed entirely inside each
  training fold.
- **Preprocessing leakage:** `StandardScaler` is fitted before the train/test
  split, so information from the held-out test data influences preprocessing.
- **Unstratified split:** the test set contains only 16 of the 94 class-1
  samples in this run; the reported metrics may be sensitive to the split.
- **Limited validation:** the reported evaluation uses one random hold-out
  split. There is no cross-validation or independent external cohort in the
  executable analysis.
- **No feature selection in code:** although the thesis discusses biomarkers
  and selection methods, `BRCA.py` does not implement the statistical tests,
  LASSO selection, or an interpretable feature-importance analysis. Biomarker
  and pathway claims in the thesis are not independently reproduced by this
  script.
- **No clinical validation:** no calibration assessment, decision analysis,
  prospective validation, or assessment of clinical consequences is included.
- **Data provenance:** source accession, cohort details, and a feature
  dictionary are not present alongside the supplied CSV.

These limitations mean that the reported performance should be treated as
exploratory research only.

## Run the analysis

Use Python 3.10 or newer and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python BRCA.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
As supplied, `BRCA.py` reads from the Kaggle path
`/kaggle/input/brca-multiomics-tcga/data.csv`, not from the repository's
`data.csv`. In this local run, a temporary in-memory wrapper redirected that
input to the included CSV, captured the displayed figures, and saved the
metrics and terminal output under `results/`; no Python source file was
changed. The wrapper was not checked in, so direct execution outside Kaggle
requires making the data available at the expected path. The script itself
does not save result files.

## Thesis reference

*Development of a Machine Learning-Based Diagnostic Tool
for Early Detection of Breast Cancer using Multi-Omics Data Integration*.
MSc Bioinformatics project (2024). The anonymized submitted document is
included as [`thesis.docx`](thesis.docx).
