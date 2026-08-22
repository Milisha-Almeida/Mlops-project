# Pharmaceutical Cold-Chain Potency Prediction

## 1. Project Title

**Pharmaceutical Cold-Chain Potency Prediction Using Machine Learning**

---

## 2. Problem Statement

Temperature-sensitive pharmaceutical products such as vaccines can lose potency during transportation due to changes in temperature, thermal exposure, transit conditions, and other cold-chain factors.

This project uses pharmaceutical cold-chain monitoring data to predict the remaining vaccine potency **360 minutes into the future**.

---

## 3. Objective

To develop a machine learning regression model that predicts:

`potency_remaining_pct_t_plus_H`

using pharmaceutical cold-chain environmental, biological, thermal, and logistics features.

---

## 4. Dataset / Source

**Dataset:** Vaccine Cold-Chain Cyber–Physical Logistics Dataset (VCC-CPLD)

**Source:** Zenodo

https://zenodo.org/records/18527963

**DOI:** 10.5281/zenodo.18527963

The original dataset contains **445,603 time-indexed records** collected at a **1-minute sampling resolution**.

A refined subset of relevant pharmaceutical cold-chain features was selected for this project.

---

## 5. Target Variable

**Target:** `potency_remaining_pct_t_plus_H`

The target represents the predicted remaining vaccine potency at a **360-minute forecasting horizon**.

This is a **supervised regression problem** because the target is a continuous numerical value.

---

## 6. Selected Features

The refined dataset contains features covering:

- Biological and baseline state
- Potency proxy and vaccine stability
- Vaccine vial monitor status
- Remaining shelf life
- Thermal exposure and cumulative thermal dose
- Freeze and thaw-refreeze events
- Thermal excursions
- Internal and ambient temperature
- Temperature variation and rate of change
- Door activity and handling conditions
- Vaccine type and packaging type
- Route stage
- Cumulative transit time
- Time since pack-out
- Estimated time of arrival
- Weather risk

The constant feature `time_below_threshold_min` was excluded because all observations contained a value of zero.

---

## 7. Methodology

1. Load the refined dataset.
2. Separate predictors and target variable.
3. Identify numerical and categorical features.
4. One-hot encode categorical variables.
5. Standardize features for linear models.
6. Perform an 80/20 time-based train-test split.
7. Train multiple regression models.
8. Evaluate models using MAE, RMSE, and R².
9. Select the best-performing model.
10. Save the final trained Ridge Regression model.
11. Use the saved model to make future potency predictions.

---

## 8. Models Tested

The following regression models were evaluated:

- Linear Regression (OLS)
- Ridge Regression (L2)
- Random Forest Regressor
- Gradient Boosting Regressor
- HistGradientBoosting Regressor

---

## 9. Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Ridge Regression (L2)** | **1.6262** | **1.9806** | **0.1986** |
| Linear Regression (OLS) | 1.6270 | 1.9813 | 0.1980 |
| Gradient Boosting | 1.6454 | 2.0046 | 0.1790 |
| Random Forest | 1.6542 | 2.0160 | 0.1697 |
| HistGradientBoosting | 1.6721 | 2.0609 | 0.1322 |

---

## 10. Final Model

**Ridge Regression (L2)** was selected as the final model because it achieved the best overall performance among the evaluated models.

### Final Performance

- **MAE:** 1.6262
- **RMSE:** 1.9806
- **R²:** 0.1986

The trained model is saved as:

`models/ridge_regression_model.pkl`

---

## 11. Results

The final Ridge Regression model achieved an **MAE of 1.6262 percentage points**, meaning that the model's predictions differ from the actual future potency by approximately 1.63 percentage points on average.

The model achieved an **RMSE of 1.9806** and an **R² of 0.1986** on the test set.

Feature analysis also showed that `potency_proxy_index`, internal temperature, temperature variation, remaining shelf life, temperature rate of change, ETA, weather risk, and transit-related variables were among the important predictors.

---

## 12. Project Structure

```text
MLOps-project/
│
├── data/
│   ├── README.md
│   └── vaccine_coldchain_cleaned.csv
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── ridge_regression_model.pkl
│
├── requirements.txt
├── .gitignore
└── README.md

```

---

## 13. Limitations

-The final model has an R² of 0.1986, indicating that substantial variation in future potency remains unexplained.
-The target values are concentrated toward relatively high potency levels.
-The available features may not capture every factor affecting future pharmaceutical potency.
-Model performance may vary under different transportation and environmental conditions.
-The model is a project prototype and should not be used as a standalone pharmaceutical safety or regulatory decision-making system.