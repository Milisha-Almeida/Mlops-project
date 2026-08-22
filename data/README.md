# Dataset

This project uses the Vaccine Cold-Chain Cyber–Physical Logistics Dataset (VCC-CPLD).

## Original Dataset

The original dataset is available from Zenodo:

https://zenodo.org/records/18527963

DOI: 10.5281/zenodo.18527963

The original dataset contains 445,603 time-indexed cold-chain monitoring records collected at a 1-minute sampling resolution.

## Processed Dataset

For this project, relevant pharmaceutical cold-chain features were selected from the original dataset.

The processed dataset contains:

- 445,603 records
- 31 predictor variables
- 1 target variable

Target:

`potency_remaining_pct_t_plus_H`

The target represents predicted remaining vaccine potency at a 360-minute forecasting horizon.

## Local Dataset

The processed CSV file is stored locally because of its large size and is excluded from the Git repository using `.gitignore`.

File:

`pharmaceutical_coldchain_preprocessed_dataset.csv`

https://colab.research.google.com/drive/18iJy-yyVZKJEblgMQTJ58JcaJt5yd7I5?usp=sharing