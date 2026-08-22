import joblib
import pandas as pd


# ============================================================
# 1. LOAD MODEL
# ============================================================

MODEL_PATH = "models/ridge_regression_model.pkl"
DATA_PATH = "data/vaccine_coldchain_cleaned.csv"

print("Loading trained model...")

package = joblib.load(MODEL_PATH)

model = package["model"]
scaler = package["scaler"]
feature_columns = package["feature_columns"]

print("Model loaded successfully.")


# ============================================================
# 2. LOAD REAL DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

target = "potency_remaining_pct_t_plus_H"

X = df.drop(columns=[target])

# Remove classification target if present
if "safe_to_use_flag_t_plus_H" in X.columns:
    X = X.drop(columns=["safe_to_use_flag_t_plus_H"])


# ============================================================
# 3. SELECT ONE REAL RECORD
# ============================================================

sample = X.iloc[[0]].copy()

actual_value = df.iloc[0][target]


# ============================================================
# 4. ENCODE CATEGORICAL VARIABLES
# ============================================================

categorical_columns = sample.select_dtypes(
    include=["object", "category"]
).columns.tolist()

sample_encoded = pd.get_dummies(
    sample,
    columns=categorical_columns,
    drop_first=True
)


# ============================================================
# 5. MATCH TRAINING FEATURES
# ============================================================

sample_encoded = sample_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)


# ============================================================
# 6. SCALE
# ============================================================

sample_scaled = scaler.transform(
    sample_encoded
)


# ============================================================
# 7. PREDICT
# ============================================================

prediction = model.predict(
    sample_scaled
)[0]


# ============================================================
# 8. DISPLAY
# ============================================================

print("\n========================================")
print("PHARMACEUTICAL COLD-CHAIN PREDICTION")
print("========================================")

print(
    f"Actual potency:    {actual_value:.2f}%"
)

print(
    f"Predicted potency: {prediction:.2f}%"
)

print(
    f"Absolute error:    {abs(actual_value - prediction):.2f}%"
)

print("========================================")