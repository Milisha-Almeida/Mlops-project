import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. Configuration
# ============================================================

DATA_PATH = "data/pharmaceutical_coldchain_preprocessed_dataset.csv"
MODEL_PATH = "models/linear_regression_pipeline.pkl"

TARGET = "potency_remaining_pct_t_plus_H"


# ============================================================
# 2. Load dataset
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 3. Separate features and target
# ============================================================

X = df.drop(
    columns=[
        TARGET,
        "time_below_threshold_min"
    ]
)

y = df[TARGET]

print("\nFeatures:", X.shape)
print("Target:", y.shape)


# ============================================================
# 4. Identify categorical and numerical columns
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumber of numerical columns:")
print(len(numerical_columns))


# ============================================================
# 5. Time-based train/test split
# ============================================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. Preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# ============================================================
# 7. Create Linear Regression model
# ============================================================

model = LinearRegression()


# ============================================================
# 8. Create complete pipeline
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 9. Train model
# ============================================================

print("\nTraining Linear Regression...")

pipeline.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# 10. Make predictions
# ============================================================

print("\nMaking predictions...")

y_pred = pipeline.predict(X_test)


# ============================================================
# 11. Evaluate model
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n========================================")
print("FINAL MODEL PERFORMANCE")
print("========================================")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 12. Save model
# ============================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(MODEL_PATH)