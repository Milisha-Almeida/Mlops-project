import joblib
import pandas as pd


# ============================================================
# 1. Load trained model
# ============================================================

MODEL_PATH = "models/linear_regression_pipeline.pkl"

print("Loading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# 2. New shipment conditions
# ============================================================

new_shipment = pd.DataFrame([
    {
        "vaccine_type": "mRNA_frozen",
        "packaging_type": "cold_box",
        "route_stage": "transit",

        "temp_internal_c": -68.5,
        "temp_ambient_c": 25.0,
        "temp_setpoint_c": -70.0,

        "temp_min_30m": -69.5,
        "temp_max_30m": -67.5,
        "temp_std_30m": 0.6,
        "temp_rate_change": 0.05,

        "thermal_excursion_count": 0,

        "humidity_internal_pct": 50.0,
        "humidity_ambient_pct": 65.0,
        "dew_point_c": 18.0,
        "condensation_risk_score": 0.1,

        "compressor_on_ratio": 0.95,
        "compressor_current_a": 8.0,
        "compressor_rpm": 2500,

        "evaporator_temp_c": -72.0,
        "condenser_temp_c": 35.0,
        "fan_speed_rpm": 1800,
        "coolant_pressure_kpa": 250.0,
        "battery_voltage_v": 24.0,

        "power_mode": "normal",
        "power_outage_count": 0,
        "defrost_cycle_flag": 0,
        "controller_reset_count": 0,

        "time_since_packout_min": 300,
        "cumulative_transit_time_min": 300,

        "time_above_threshold_min": 0
    }
])


# ============================================================
# 3. Make prediction
# ============================================================

print("\nMaking prediction...")

prediction = model.predict(new_shipment)


# ============================================================
# 4. Display result
# ============================================================

predicted_potency = prediction[0]

print("\n========================================")
print("PHARMACEUTICAL COLD-CHAIN PREDICTION")
print("========================================")

print(
    f"Predicted potency after 360 minutes: "
    f"{predicted_potency:.2f}%"
)

print("========================================")