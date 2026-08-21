# Pharmaceutical Cold-Chain Potency Prediction

## 1. Project Title

**Pharmaceutical Cold-Chain Potency Prediction Using Machine Learning**

---

## 2. Problem Statement

Temperature-sensitive pharmaceutical products such as vaccines can lose potency during transportation due to changes in temperature, transit duration, humidity, and refrigeration conditions.

This project uses cold-chain monitoring data to predict the remaining pharmaceutical potency 360 minutes into the future.

---

## 3. Objective

To develop a machine learning regression model that predicts:

`potency_remaining_pct_t_plus_H`

using pharmaceutical cold-chain environmental and operational data.

---

## 4. Dataset / Source

**Dataset:** Vaccine Cold-Chain Cyber–Physical Logistics Dataset (VCC-CPLD)

**Source:** Zenodo  
https://zenodo.org/records/18527963

**DOI:** 10.5281/zenodo.18527963

The dataset contains 445,603 time-indexed records collected at a 1-minute sampling resolution.

A processed subset of the dataset was used for this project.

---

## 5. Target Variable

**Target:** `potency_remaining_pct_t_plus_H`

The target represents predicted remaining pharmaceutical potency at a **360-minute forecasting horizon**.

This is a **supervised regression problem** because the target is a continuous numerical value.

---

## 6. Selected Features

The model uses 30 predictor variables covering:

- Vaccine and packaging information
- Route stage
- Internal and ambient temperature
- Temperature statistics and changes
- Thermal excursions
- Humidity and condensation
- Compressor and refrigeration conditions
- Power and equipment conditions
- Time since pack-out
- Cumulative transit time
- Time above temperature threshold

The constant feature `time_below_threshold_min` was excluded because all 445,603 observations contained a value of zero.

---

## 7. Methodology

1. Load the processed dataset.
2. Separate predictors and target.
3. Remove the constant feature.
4. Identify numerical and categorical features.
5. One-hot encode categorical variables.
6. Perform an 80/20 time-based train-test split.
7. Train regression models.
8. Evaluate model performance using MAE, RMSE and R².
9. Select the best-performing model.
10. Save the trained model pipeline.
11. Use the saved pipeline for future predictions.

---

## 8. Models Tested

- Mean Baseline
- Time-only Linear Regression
- Random Forest Regressor
- Multivariable Linear Regression
- Gradient Boosting Regressor

---

## 9. Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean Baseline | 1.7645 | 2.2565 | ~0.0000 |
| Random Forest | 1.7448 | 2.2396 | 0.0149 |
| Time-only Linear Regression | 1.6971 | 2.2001 | 0.0493 |
| **Multivariable Linear Regression** | **1.6964** | **2.1965** | **0.0525** |
| Gradient Boosting | 1.6976 | 2.1966 | 0.0523 |

---

## 10. Final Model

**Multivariable Linear Regression** was selected as the final model because it achieved the best performance among the tested models.

The trained pipeline is saved as:

`models/linear_regression_pipeline.pkl`

---

## 11. Results

Final model performance:

- **MAE:** 1.6964
- **RMSE:** 2.1965
- **R²:** 0.0525

The model predicts future potency with an average error of approximately **1.70 percentage points**.

---

## 12. Project Structure

```text
MLOps-project/
│
├── data/
│   └── README.md
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── linear_regression_pipeline.pkl
│
├── requirements.txt
├── .gitignore
└── README.md


### 13. Limitations

Finally:

```markdown
## 13. Limitations

- The final model has a relatively low R² of 0.0525.
- The target values are heavily concentrated near high potency levels.
- The available features may not capture all factors affecting future potency.
- The model is a project prototype and should not be used as a standalone pharmaceutical safety or regulatory decision-making system.