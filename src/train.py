# ============================================================
# MODEL TRAINING & BENCHMARKING
# ============================================================
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading dataset...")

DATA_PATH = "data/vaccine_coldchain_cleaned.csv"

df_model = pd.read_csv(DATA_PATH)

print("Dataset shape:", df_model.shape)


# ============================================================
# 2. TARGET
# ============================================================

target = "potency_remaining_pct_t_plus_H"

drop_cols = [target]

# Classification target is not used for regression
if "safe_to_use_flag_t_plus_H" in df_model.columns:
    drop_cols.append("safe_to_use_flag_t_plus_H")


X = df_model.drop(columns=drop_cols)

y = df_model[target]


print("\nFeatures:", X.shape)
print("Target:", y.shape)


# ============================================================
# 3. IDENTIFY CATEGORICAL FEATURES
# ============================================================

cat_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nCategorical columns:")
print(cat_cols)


# ============================================================
# 4. ONE-HOT ENCODING
# ============================================================

X_encoded = pd.get_dummies(
    X,
    columns=cat_cols,
    drop_first=True
)

print("\nEncoded feature shape:", X_encoded.shape)


# ============================================================
# 5. CHRONOLOGICAL 80/20 TRAIN-TEST SPLIT
# ============================================================

split_index = int(len(X_encoded) * 0.80)

X_train = X_encoded.iloc[:split_index]
X_test = X_encoded.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\n========================================")
print("TRAIN / TEST SPLIT")
print("========================================")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 6. STANDARDIZATION FOR LINEAR MODELS
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 7. DEFINE MODELS
# ============================================================

models = {

    "Linear Regression (OLS)": (
        LinearRegression(),
        True
    ),

    "Ridge Regression (L2)": (
        Ridge(alpha=10.0),
        True
    ),

    "Random Forest": (
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        False
    ),

    "Gradient Boosting": (
        GradientBoostingRegressor(
            n_estimators=150,
            random_state=42
        ),
        False
    ),

    "HistGradientBoosting": (
        HistGradientBoostingRegressor(
            max_iter=200,
            random_state=42
        ),
        False
    )
}


# ============================================================
# 8. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

print("\n========================================")
print("MODEL TRAINING")
print("========================================")

for name, (model, requires_scaling) in models.items():

    print(f"\nTraining {name}...")

    if requires_scaling:

        X_tr = X_train_scaled
        X_te = X_test_scaled

    else:

        X_tr = X_train
        X_te = X_test

    model.fit(
        X_tr,
        y_train
    )

    preds = model.predict(
        X_te
    )

    mae = mean_absolute_error(
        y_test,
        preds
    )

    mse = mean_squared_error(
        y_test,
        preds
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        preds
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("Training completed.")


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
).sort_values(
    by="R2 Score",
    ascending=False
)


print("\n========================================")
print("FINAL MODEL COMPARISON")
print("========================================")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 10. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf = models["Random Forest"][0]

importances = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance": rf.feature_importances_

}).sort_values(
    by="Importance",
    ascending=False
)


print("\n========================================")
print("TOP 10 FEATURE IMPORTANCES")
print("========================================")

print(
    importances.head(10).to_string(
        index=False
    )
)


# ============================================================
# 11. BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print("\n========================================")
print("BEST MODEL")
print("========================================")

print("Model:", best_model["Model"])
print(f"MAE  : {best_model['MAE']:.4f}")
print(f"RMSE : {best_model['RMSE']:.4f}")
print(f"R²   : {best_model['R2 Score']:.4f}")

# ============================================================
# 12. SAVE FINAL RIDGE MODEL
# ============================================================

# Create models directory if it does not exist
os.makedirs("models", exist_ok=True)

# Ridge model is the best-performing model
ridge_model = models["Ridge Regression (L2)"][0]

# Save Ridge model + scaler together
model_package = {
    "model": ridge_model,
    "scaler": scaler,
    "feature_columns": X_train.columns.tolist()
}

model_path = "models/ridge_regression_model.pkl"

joblib.dump(
    model_package,
    model_path
)

print("\n========================================")
print("FINAL MODEL SAVED")
print("========================================")

print(f"Model: Ridge Regression (L2)")
print(f"Saved to: {model_path}")