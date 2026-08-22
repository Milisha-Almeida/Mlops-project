# Dataset

This project uses the **Vaccine Cold-Chain Cyber–Physical Logistics Dataset (VCC-CPLD)**.

## Original Dataset

The original dataset is available from Zenodo:

https://zenodo.org/records/18527963

**DOI:** 10.5281/zenodo.18527963

The original dataset contains **445,603 time-indexed cold-chain monitoring records** collected at a **1-minute sampling resolution**.

## Refined Dataset

For this project, relevant pharmaceutical cold-chain features were selected from the original dataset.

The refined dataset contains:

- 445,603 records
- Selected cold-chain predictor variables
- 1 target variable

**Target variable:**

`potency_remaining_pct_t_plus_H`

The target represents the predicted remaining vaccine potency at a **360-minute forecasting horizon**.

The refined dataset is stored locally because of its large size and is excluded from the Git repository using `.gitignore`.

**Local file:**

`vaccine_coldchain_cleaned.csv`