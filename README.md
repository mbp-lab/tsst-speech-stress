# Automatic Detection of Stress from Speech in the Trier Social Stress Test

This repository contains the code for preprocessing, machine learning analyses, and evaluation for the paper “Automatic Detection of Stress from Speech in the Trier Social Stress Test.” 

## Notebooks
- **`01_…_Reading_Cutting`** — Preprocesses audio, identifies the participant, trims minutes 7–16, and extracts voice features.
- **`02_…_StressInduction`** — Checks whether the stress induction worked using voice and behavioral data.
- **`03_…_GroupAnalysisOfVoice`** — Compares stress markers and voice features between conditions and sexes.
- **`04_…_MLClassification`** — Classifies TSST and fTSST using several machine-learning models and cross-validation.
- **`05_…_MLRegression_loo`** - Predicts stress markers for the full dataset.
- **`06_…_MLRegression_TSST_loo`** - Predicts stress markers for the TSST condition only.
- **`07_…_StatTesting_Classification`** - Tests classifier performance against the majority baseline.
- **`08_…_StatTesting_Regression`** - Tests regression errors against the mean baseline.

## Scripts (`scripts/`)

- mycut.py — Audio cutting and speaker diarization.
- myvoice.py — Voice feature extraction and merging.
- mystress.py — Data loading and stress marker calculation.
- myml.py — Cross-validation, regression, classification, plots, and ROC analysis.
- mystats.py — Corrected statistical tests.


