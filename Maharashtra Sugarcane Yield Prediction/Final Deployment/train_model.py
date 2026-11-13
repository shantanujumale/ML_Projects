"""Train the Maharashtra sugarcane yield model using the uploaded Colab workflow.

Run from the project directory:
    python train_model.py

The script writes the selected feature list, trained tuned-XGBoost pipeline,
model metrics, and the district reference profiles used by the Streamlit app.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA_PATH = next(
    (
        path
        for path in (
            ROOT / "data" / "FINAL_SUGARCANE_DATASET.csv",
            ROOT / "FINAL_SUGARCANE_DATASET.csv",
        )
        if path.exists()
    ),
    ROOT / "data" / "FINAL_SUGARCANE_DATASET.csv",
)
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
TARGET = "Yield_Quintal_per_Acre"
RANDOM_STATE = 42


def parse_sunshine(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if ":" in value:
        try:
            hours, minutes = value.split(":", 1)
            return float(hours) + float(minutes) / 60.0
        except (ValueError, TypeError):
            return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def parse_month(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    for prefix, number in month_map.items():
        if text.startswith(prefix):
            return number
    try:
        return int(float(text))
    except (ValueError, TypeError):
        return np.nan


def prepare_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "Sunshine_Hours_hh_mm" in df.columns:
        df["Sunshine_Hours_Decimal"] = df["Sunshine_Hours_hh_mm"].apply(parse_sunshine)
        df.drop(columns=["Sunshine_Hours_hh_mm"], inplace=True)
    if "Month" in df.columns:
        df["Month_Num"] = df["Month"].apply(parse_month).fillna(6)

    df = df.drop_duplicates().reset_index(drop=True)
    leakage_cols = [c for c in df.columns if c.lower() in {
        "id", "serial_no", "record_id", "farmer_id", "total_production",
        "sugar_recovery_tons", "production", "area",
    }]
    if leakage_cols:
        df.drop(columns=leakage_cols, inplace=True)

    df["Temperature_Range"] = df["Temp_Max_C"] - df["Temp_Min_C"]
    df["Climate_Stress"] = df["Heat_Stress_Days"].fillna(0) + df["Frost_Days"].fillna(0)
    df["Water_Retention_Potential"] = (
        df["Soil_Depth_cm"] * df["Water_Holding_Capacity_%"]
    ) / 100.0
    df["N_to_P"] = df["Nitrogen_kg_per_acre"] / (df["Phosphorus_kg_per_acre"] + 1e-4)
    df["N_to_K"] = df["Nitrogen_kg_per_acre"] / (df["Potassium_kg_per_acre"] + 1e-4)
    df["P_to_K"] = df["Phosphorus_kg_per_acre"] / (df["Potassium_kg_per_acre"] + 1e-4)
    df["NPK_Total"] = (
        df["Nitrogen_kg_per_acre"] + df["Phosphorus_kg_per_acre"] + df["Potassium_kg_per_acre"]
    )
    frequency_map = {"low": 1.0, "medium": 2.0, "high": 3.0}
    df["Irrig_Freq_Num"] = (
        df["Irrigation_Frequency_Level"].astype(str).str.lower().map(frequency_map).fillna(2.0)
    )
    df["Irrigation_Intensity"] = df["Irrig_Freq_Num"] * df["Water_Quantity_liters_per_acre"]
    df["Cane_Growth_Index"] = df["Cane_Height_cm"] * df["Cane_Diameter_cm"]
    df["Internode_Height_Ratio"] = df["Internode_Length_cm"] / (df["Cane_Height_cm"] + 1e-4)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12.0)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12.0)
    df["Rainfall_Temperature"] = df["Rainfall_Total_mm"] * df["Temp_Avg_C"]
    df["Temperature_SoilMoisture"] = df["Soil_Moisture_%"] * df["Temp_Avg_C"]
    df["Fertilizer_Irrigation"] = df["Fertilizer_Quantity"] * df["Water_Quantity_liters_per_acre"]
    return df


def build_feature_selection(X_train: pd.DataFrame, y_train: pd.Series):
    train_numeric = X_train.select_dtypes(include=[np.number]).columns.tolist()
    train_categorical = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    selector_preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), train_numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]), train_categorical),
    ])
    X_train_trans = selector_preprocessor.fit_transform(X_train)
    feature_names = selector_preprocessor.get_feature_names_out()
    clean_names = [str(f).replace("num__", "").replace("cat__", "") for f in feature_names]
    selector = RandomForestRegressor(
        n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
    )
    selector.fit(X_train_trans, y_train)
    perm = permutation_importance(
        selector, X_train_trans, y_train, n_repeats=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    importance = pd.DataFrame({
        "Feature": clean_names,
        "Importance_Mean": perm.importances_mean,
        "Importance_Std": perm.importances_std,
    }).sort_values("Importance_Mean", ascending=False).reset_index(drop=True)

    core_domain = [
        "Cane_Growth_Index", "Cane_Height_cm", "Cane_Diameter_cm", "Soil_Moisture_%",
        "Water_Quantity_liters_per_acre", "Fertilizer_Quantity", "Fertilizer_Irrigation",
        "Tillering_Count", "Rainfall_Total_mm", "Rainfall_Temperature", "NPK_Total",
        "Temperature_Range", "Climate_Stress", "Water_Retention_Potential",
        "Month_Sin", "Month_Cos", "Brix_Value", "Disease_Severity",
    ]
    positive_features = set(importance.loc[importance["Importance_Mean"] > 0.001, "Feature"])
    selected_numeric = [c for c in train_numeric if c in positive_features or c in core_domain]
    selected_categorical = [
        c for c in ["Soil_Type", "Irrigation_Method_Type", "District", "Season", "Fertilizer_Type"]
        if c in train_categorical
    ]
    return selected_numeric, selected_categorical, importance


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    if TARGET not in raw.columns:
        raise ValueError(f"Required target column '{TARGET}' is missing.")
    df = prepare_dataset(raw)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    selected_numeric, selected_categorical, importance = build_feature_selection(X_train, y_train)
    selected_cols = selected_numeric + selected_categorical
    X_train_sub = X_train[selected_cols]
    X_test_sub = X_test[selected_cols]

    model_preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), selected_numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]), selected_categorical),
    ])
    # Parameters selected by RandomizedSearchCV in the uploaded Colab notebook.
    model = Pipeline([
        ("prep", model_preprocessor),
        ("model", XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.75,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="reg:squarederror",
        )),
    ])
    model.fit(X_train_sub, y_train)
    pred = model.predict(X_test_sub)
    metrics = {
        "model_name": "Tuned XGBoost Regressor",
        "test_mae": float(mean_absolute_error(y_test, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "test_r2": float(r2_score(y_test, pred)),
        "train_rows": int(len(X_train_sub)),
        "test_rows": int(len(X_test_sub)),
        "selected_numeric_features": int(len(selected_numeric)),
        "selected_categorical_features": int(len(selected_categorical)),
        "feature_names": selected_cols,
        "training_note": "Re-trained from the supplied CSV using the selected XGBoost hyperparameters recorded in the Colab notebook. Metrics are measured on the 20% holdout split.",
        "notebook_recorded_tuned_xgboost": {
            "test_mae": 74.303387,
            "test_r2": 0.290210,
        },
        "data_rows_after_deduplication": int(len(df)),
        "raw_dataset_rows": int(len(raw)),
        "raw_dataset_columns": int(raw.shape[1]),
    }
    joblib.dump(model, ARTIFACT_DIR / "sugarcane_yield_best_model.pkl", compress=3)
    pd.Series(selected_numeric, name="Selected_Numeric_Features").to_csv(ARTIFACT_DIR / "selected_numeric_features.csv", index=False)
    pd.Series(selected_categorical, name="Selected_Categorical_Features").to_csv(ARTIFACT_DIR / "selected_categorical_features.csv", index=False)
    importance.to_csv(ARTIFACT_DIR / "permutation_importance.csv", index=False)
    with open(ARTIFACT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Persist district-specific raw medians/modes as readable data, for reproducible form defaults.
    raw_for_profiles = prepare_dataset(raw)
    profiles = {}
    for district, group in raw_for_profiles.groupby("District", dropna=False):
        numeric = group.select_dtypes(include=[np.number]).median(numeric_only=True).to_dict()
        categorical = {}
        for col in group.select_dtypes(exclude=[np.number]).columns:
            mode = group[col].mode(dropna=True)
            categorical[col] = str(mode.iloc[0]) if not mode.empty else ""
        profiles[str(district)] = {"numeric": numeric, "categorical": categorical}
    with open(ARTIFACT_DIR / "district_profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {ARTIFACT_DIR / 'sugarcane_yield_best_model.pkl'}")


if __name__ == "__main__":
    main()
