# Predicting 30-Day Mortality in Sepsis Patients — MIMIC-IV

An interactive R Shiny dashboard comparing four machine learning models for predicting 30-day mortality in ICU sepsis patients, built on the MIMIC-IV critical care database.
Author：Zihan (Hannah) Teng
**Live demo:** https://hannahteng.shinyapps.io/sepsis-mortality-demo/

## Why this matters

Sepsis is the leading cause of in-hospital mortality worldwide and accounts for one in three hospital deaths in the United States. Early identification of high-risk patients enables timely escalation of care — broader-spectrum antibiotics, ICU admission, vasopressor support — and is one of the few interventions consistently shown to reduce sepsis mortality. This project builds an interpretable, interactive prediction tool that surfaces the patient-level features driving 30-day mortality risk in a transparent way.

## What the dashboard does

The app has three tabs:

1. **Cohort Explorer** — descriptive statistics, demographics, and outcome distributions of the sepsis cohort. Filter by age, gender, SOFA score, or comorbidity burden to see how the population shifts.
2. **Patient Trajectory** — pick an individual patient and visualize their vital signs and lab trajectories during the ICU stay alongside their final outcome.
3. **ML Prediction App** — a risk calculator. Enter a patient's baseline features and get a predicted 30-day mortality probability from each of four trained models, with model-comparison ROC and calibration plots.

## Methods

**Data source.** MIMIC-IV v2.2 (Medical Information Mart for Intensive Care), a freely available but credentialed-access database of de-identified ICU stays from Beth Israel Deaconess Medical Center.

**Cohort.** Adult ICU admissions meeting Sepsis-3 criteria, n = 30,121.

**Outcome.** 30-day all-cause mortality (`label_30d`), prevalence ≈ 19.4%.

**Features.** 172 baseline variables including SOFA score components, vital signs (heart rate, blood pressure, respiratory rate, SpO₂, temperature — min/max/mean per ICU stay), routine labs (CBC, metabolic panel, liver function, coagulation), Charlson comorbidity index, and demographics.

**Models compared.**

| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline; coefficients give direct effect sizes |
| LASSO (L1-regularized) | Sparse model with automatic feature selection |
| Random Forest with Boruta | Non-linear ensemble; Boruta-selected feature subset |
| Neural Network (single hidden layer) | Captures complex feature interactions |

**Evaluation.** Stratified 80/20 train-test split, AUC-ROC and calibration on held-out test set, model performance reported within the dashboard's ML tab.

## Demo vs. real analysis (important)

This deployed demo runs on **synthetic data** (n = 2,000) generated to preserve the statistical properties of the original cohort — column-wise distributions, missingness patterns, and known mortality predictor relationships (SOFA score, age, Charlson index). This is necessary because MIMIC-IV is governed by a PhysioNet Data Use Agreement that prohibits redistribution to non-credentialed users. The synthetic data is regenerated from the real distribution at preprocessing time and is suitable for demonstrating the modeling pipeline and dashboard UI; **it should not be used for any clinical inference.**

The original analysis used the full MIMIC-IV sepsis cohort with the identical modeling pipeline. Real-data results are documented in the project report (available on request).

## Tech stack

R · Shiny · shinydashboard · DT · plotly · tidyverse · glmnet (LASSO) · ranger (Random Forest) · nnet (Neural Network) · caret · pROC

## Run locally

```bash

# Install R dependencies (one-time)
Rscript -e 'install.packages(c("shiny","shinydashboard","DT","plotly","tidyverse","readxl","glmnet","ranger","nnet","caret","pROC","openxlsx"))'

# Generate synthetic data (only needed once)
Rscript make_synthetic.R

# Launch the app
Rscript -e 'shiny::runApp("app.R", launch.browser = TRUE)'
```

The app should open at `http://localhost:xxxx` in your browser.

## Project structure

```
.
├── app.R                                       # Main Shiny app (UI + server + model training)
├── make_synthetic.R                            # Synthetic data generator
├── mimic_sepsis_30d_full_features.xlsx         # Synthetic dataset (used by the app)
├── mimic_sepsis_30d_full_features_REAL.xlsx    # Real MIMIC-IV data (gitignored, credentialed only)
└── README.md
```

## Limitations

- Single-center data (Beth Israel Deaconess); generalizability to other health systems is unknown.
- The Sepsis-3 cohort definition relies on retrospective EHR data; some sepsis cases may be misclassified.
- The deployed model is trained on synthetic data and is **not** a validated clinical decision tool.
- Several lab features (D-dimer, GGT, globulin) have >95% missingness in the source data and contribute little signal.

## Author

**Hannah Teng**
M.S. Data Science in Health, UCLA
BIOSTAT 212B Spring 2026 — Individual Extra Credit Project

## Disclaimer

This project is for educational and research demonstration purposes only. It is not a medical device, has not been validated for clinical use, and must not be used to inform patient care decisions. All patient data shown in the deployed demo is synthetic.

## References

1. Johnson AEW, Bulgarelli L, Shen L, et al. *MIMIC-IV, a freely accessible electronic health record dataset.* Scientific Data. 2023;10:1.
2. Singer M, Deutschman CS, Seymour CW, et al. *The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).* JAMA. 2016;315(8):801-810.
3. Rudd KE, Johnson SC, Agesa KM, et al. *Global, regional, and national sepsis incidence and mortality, 1990-2017.* Lancet. 2020;395(10219):200-211.
