# Breast Cancer Multi-Omics Outcome Classification

MSc Bioinformatics thesis project, University of West London (2024), with a leakage-corrected re-analysis (2026).

> **Research and educational use only.** This is a retrospective machine-learning exercise on a public dataset. It is not a diagnostic or prognostic medical device.

## Summary

Four supervised classifiers were compared for predicting recorded patient vital status from 1,936 molecular features in 705 breast cancer samples. **Random Forest was the best-performing model by accuracy** in both the original thesis (90.07%) and the corrected re-analysis (89.36%).

The re-analysis also showed that high accuracy hides weak performance on the rare outcome. Only 13% of patients have label 1, and the models found 21–37% of them.

## Repository contents

| Path | Description |
| --- | --- |
| `BRCA.py` | Current, leakage-corrected analysis (2026) |
| `results/` | Metrics, run log and figures from `BRCA.py` |
| `original-2024/BRCA_2024.py` | Original thesis script, as run on Kaggle in 2024 |
| `original-2024/results/` | Metrics, run log and figures from the original script |
| `data.csv` | Feature matrix and `vital.status` labels |
| `thesis.docx` | Thesis document, revised in 2026 to report the corrected re-analysis |
| `requirements.txt` | Python dependencies |

## Dataset

A public TCGA-derived breast cancer dataset distributed on Kaggle (`brca-multiomics-tcga`). It has 705 samples, 1,936 features and one binary outcome, `vital.status`.

| `vital.status` | Samples | Share |
| --- | ---: | ---: |
| 0 | 611 | 86.7% |
| 1 | 94 | 13.3% |

Features fall into four blocks by prefix: `rs_` (604), `cn_` (860), `mu_` (249) and `pp_` (223). The file has no data dictionary, so the blocks are not formally defined here.

## Original thesis results and the 2026 correction

### What changed

The original 2024 script made two preprocessing choices that are now known to bias evaluation.

1. **Scaling before splitting.** `StandardScaler` was fitted on all 705 samples before the train/test split. The test samples therefore influenced the preprocessing applied to the training data. This is data leakage.
2. **No stratification.** The split did not preserve the class balance. The test set happened to contain 16 positive cases instead of the 19 a stratified split gives.

The corrected `BRCA.py` uses a stratified split and fits the scaler on the training data only. Models, settings and the random seed are otherwise unchanged.

### Side-by-side results

Test set of 141 samples in both runs.

| Model | Accuracy 2024 | Accuracy 2026 | Class-1 recall 2024 | Class-1 recall 2026 | ROC-AUC 2024 | ROC-AUC 2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Random Forest** | **0.9007** | **0.8936** | 0.1875 | 0.2105 | 0.4965 | 0.6564 |
| Neural network (MLP) | 0.8936 | 0.8794 | 0.2500 | 0.3158 | 0.5685 | 0.5846 |
| Linear SVM | 0.8440 | 0.8298 | 0.2500 | 0.3684 | 0.5935 | 0.6700 |
| Gradient Boosting | 0.8936 | 0.8865 | 0.1875 | 0.3684 | 0.5620 | 0.6562 |

Random Forest has the highest accuracy in both runs. The small drop in accuracy after the correction is expected once leakage is removed.

### How to read these numbers

Accuracy is misleading here. In the 2024 test set, 125 of 141 patients were class 0. A model that always predicts class 0 would score 88.65% and find no class-1 patients. Random Forest scored 90.07% but found 3 of the 16 class-1 patients.

The more informative measures are class-1 recall, F1-score and ROC-AUC, all reported in `results/model_metrics.csv`.

## Current results in full (2026)

| Model | Accuracy | Class-1 precision | Class-1 recall | Class-1 F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.8936 | 1.0000 | 0.2105 | 0.3478 | 0.6564 |
| Neural network (MLP) | 0.8794 | 0.6000 | 0.3158 | 0.4138 | 0.5846 |
| Linear SVM | 0.8298 | 0.3684 | 0.3684 | 0.3684 | 0.6700 |
| Gradient Boosting | 0.8865 | 0.6364 | 0.3684 | 0.4667 | 0.6562 |

## Research question and data: a known mismatch

The thesis title refers to **early detection** of breast cancer. Detection means separating people with cancer from people without it, which needs healthy or normal-tissue controls.

This dataset contains only breast cancer patients, labelled by vital status. It can answer a **prognosis** question (which patients have a poorer outcome), not a detection question. The results above should be read as outcome classification.

A detection study would use, for example, TCGA-BRCA tumour samples with matched normal tissue, or a GEO cohort with healthy controls. A stronger prognosis study would use survival methods that account for follow-up time.

## Limitations

- **Single split.** Results come from one train/test split and may vary with a different split.
- **High dimensionality.** There are 1,936 features for 705 samples, with no feature selection inside cross-validation, so overfitting is likely.
- **Class imbalance.** No resampling, class weighting or threshold tuning was used.
- **Outcome definition.** Vital status ignores follow-up time. Patients alive at last contact may still have an event later.
- **No external validation.** No independent cohort, calibration analysis or clinical evaluation was performed.

## Reproduce

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python BRCA.py
```

`BRCA.py` reads `data.csv` from the repository folder and writes metrics, a log and figures to `results/`. The original script, `original-2024/BRCA_2024.py`, reads the Kaggle input path and must be run on Kaggle or with that path changed.

Original Kaggle notebook: https://www.kaggle.com/code/shaiksohaik03/brca-multi-omics-model/notebook

## Thesis reference

*Development of a Machine Learning-Based Diagnostic Tool for Early Detection of Breast Cancer using Multi-Omics Data Integration.* MSc Bioinformatics thesis, University of West London (2024).
