"""
Sugarcane Yield Prediction Module
"Maharashtra Sugarcane Yield Prediction and Agro-Resource Optimization Using Explainable Machine Learning"

Provides a simple, production-ready function `predict_sugarcane_yield()`
to predict sugarcane yield (Quintals/Acre & Tonnes/Ha) from farm inputs.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Union, Dict, Any, List

# Cache loaded model globally
_MODEL_CACHE = None

# Regional baseline medians and modes derived from the 3,000-farm Maharashtra dataset
DEFAULT_NUMERIC_FEATURES = {
    'Nitrogen_kg_per_acre': 152.74,
    'Phosphorus_kg_per_acre': 69.74,
    'Potassium_kg_per_acre': 115.66,
    'Water_Quantity_liters_per_acre': 1229.68,
    'Fertilizer_Quantity': 175.60,
    'Soil_pH': 7.24,
    'EC': 1.28,
    'Clay_%': 24.51,
    'Sand_%': 40.23,
    'Temp_Avg_C': 27.84,
    'Temp_Max_C': 33.23,
    'Temp_Min_C': 22.31,
    'Rainfall_Seasonal_mm': 790.83,
    'Dew_Point_C': 17.38,
    'Evapotranspiration_mm_day': 5.02,
    'Groundwater_Level_meters': 10.68,
    'Cane_Diameter_cm': 3.55,
    'Crop_Duration_Days': 325.0,
    'Row_Gap_cm': 105.61,
    'Sulfur_kg_per_acre': 17.16,
    'Copper_mg_per_kg': 1.61,
    'Climate_Stress': 0.0,
    'Water_Retention_Potential': 45.2,
    'Soil_Fertility_Index': 0.05,
    'Stress_Index': 0.0,
    'Temperature_SoilMoisture': 780.0,
    'Rainfall_Temperature': 22018.0,
    'NPK_Total': 338.14,
    'N_to_K': 1.32,
    'N_to_P': 2.19,
    'P_to_K': 0.60
}

DEFAULT_CATEGORICAL_FEATURES = {
    'District': 'Kolhapur',
    'Season': 'Kharif',
    'Soil_Type': 'Black',
    'Irrigation_Method_Type': 'Drip',
    'Irrigation_Frequency_Level': 'Medium',
    'Fertilizer_Type': 'DAP',
    'Variety': 'CoJ64',
    'Disease_Type': 'None',
    'Disease_Severity': 'Low',
    'Pest_Level': 'Low',
    'Soil_Condition_At_Planting': 'Wet'
}

EXPECTED_NUMERIC = [
    'Nitrogen_kg_per_acre', 'NPK_Total', 'N_to_K', 'Stress_Index', 'N_to_P',
    'Soil_Fertility_Index', 'Temperature_SoilMoisture', 'Soil_pH', 'Potassium_kg_per_acre',
    'P_to_K', 'Temp_Avg_C', 'Water_Retention_Potential', 'EC', 'Cane_Diameter_cm',
    'Rainfall_Temperature', 'Crop_Duration_Days', 'Clay_%', 'Temp_Max_C', 'Climate_Stress',
    'Rainfall_Seasonal_mm', 'Copper_mg_per_kg', 'Sulfur_kg_per_acre', 'Row_Gap_cm',
    'Dew_Point_C', 'Sand_%', 'Groundwater_Level_meters', 'Evapotranspiration_mm_day',
    'Temp_Min_C', 'Water_Quantity_liters_per_acre', 'Phosphorus_kg_per_acre', 'Fertilizer_Quantity'
]

EXPECTED_CATEGORICAL = [
    'District', 'Season', 'Soil_Type', 'Irrigation_Method_Type', 'Irrigation_Frequency_Level',
    'Fertilizer_Type', 'Variety', 'Disease_Type', 'Disease_Severity', 'Pest_Level',
    'Soil_Condition_At_Planting'
]

EXPECTED_COLUMNS = EXPECTED_NUMERIC + EXPECTED_CATEGORICAL


def get_model(model_path: str = None):
    """Loads and caches the champion trained model."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if model_path is None:
        candidates = [
            'sugarcane_yield_best_model.pkl',
            os.path.join(os.path.dirname(__file__), 'sugarcane_yield_best_model.pkl'),
            os.path.join(os.path.dirname(__file__), 'Code', 'models', 'sugarcane_yield_best_model.pkl'),
            os.path.join(os.path.dirname(__file__), '..', 'models', 'sugarcane_yield_best_model.pkl')
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Please run `python Code/run_pipeline.py` to generate `sugarcane_yield_best_model.pkl`."
        )

    _MODEL_CACHE = joblib.load(model_path)
    return _MODEL_CACHE


def _prepare_single_row(user_dict: Dict[str, Any]) -> pd.DataFrame:
    """Prepares and validates a single farm input dictionary with defaults and derived features."""
    row = {}

    # 1. Fill base numerical and categorical defaults
    row.update(DEFAULT_NUMERIC_FEATURES)
    row.update(DEFAULT_CATEGORICAL_FEATURES)

    # 2. Override with user provided fields
    for k, v in user_dict.items():
        if v is not None:
            row[k] = v

    # 3. Dynamically compute domain features if raw inputs were supplied
    n = float(row.get('Nitrogen_kg_per_acre', DEFAULT_NUMERIC_FEATURES['Nitrogen_kg_per_acre']))
    p = float(row.get('Phosphorus_kg_per_acre', DEFAULT_NUMERIC_FEATURES['Phosphorus_kg_per_acre']))
    k = float(row.get('Potassium_kg_per_acre', DEFAULT_NUMERIC_FEATURES['Potassium_kg_per_acre']))

    if 'NPK_Total' not in user_dict:
        row['NPK_Total'] = n + p + k
    if 'N_to_P' not in user_dict:
        row['N_to_P'] = n / (p + 1e-4)
    if 'N_to_K' not in user_dict:
        row['N_to_K'] = n / (k + 1e-4)
    if 'P_to_K' not in user_dict:
        row['P_to_K'] = p / (k + 1e-4)

    t_avg = float(row.get('Temp_Avg_C', DEFAULT_NUMERIC_FEATURES['Temp_Avg_C']))
    rain_s = float(row.get('Rainfall_Seasonal_mm', DEFAULT_NUMERIC_FEATURES['Rainfall_Seasonal_mm']))
    if 'Rainfall_Temperature' not in user_dict:
        row['Rainfall_Temperature'] = rain_s * t_avg

    # 4. Construct DataFrame aligned strictly with expected pipeline columns
    df_row = pd.DataFrame([{col: row.get(col, DEFAULT_NUMERIC_FEATURES.get(col, 'Standard')) for col in EXPECTED_COLUMNS}])
    return df_row


def predict_sugarcane_yield(
    input_data: Union[Dict[str, Any], pd.DataFrame, pd.Series, None] = None,
    return_details: bool = False,
    model_path: str = None,
    **kwargs
) -> Union[float, np.ndarray, Dict[str, Any]]:
    """
    Predicts sugarcane yield based on the trained champion Machine Learning model.

    Parameters
    ----------
    input_data : dict, pd.DataFrame, pd.Series, optional
        Farm attributes (e.g., District, Soil_Type, Water_Quantity_liters_per_acre, Fertilizer_Quantity).
        Any unsupplied fields automatically take regional baseline defaults.
    return_details : bool, default False
        If True and a single record is provided, returns a detailed dictionary
        including Quintal/Acre, Tonnes/Ha, category, and comparison to state average.
    model_path : str, optional
        Custom path to the serialized `.pkl` pipeline file.
    **kwargs : optional
        Additional farm attributes passed as direct keyword arguments.

    Returns
    -------
    float or np.ndarray or dict
        Predicted yield in Quintals per Acre, array of predictions, or detailed report dict.
    """
    model = get_model(model_path)

    # Case 1: Pandas DataFrame passed (batch prediction)
    if isinstance(input_data, pd.DataFrame):
        df_in = input_data.copy()
        # Ensure all expected columns exist
        for col in EXPECTED_COLUMNS:
            if col not in df_in.columns:
                default_val = DEFAULT_NUMERIC_FEATURES.get(col, DEFAULT_CATEGORICAL_FEATURES.get(col, 0))
                df_in[col] = default_val
        preds = model.predict(df_in[EXPECTED_COLUMNS])
        return preds

    # Case 2: Single instance (dict, Series, or kwargs)
    record = {}
    if isinstance(input_data, pd.Series):
        record.update(input_data.to_dict())
    elif isinstance(input_data, dict):
        record.update(input_data)
    record.update(kwargs)

    df_single = _prepare_single_row(record)
    pred_q_acre = float(model.predict(df_single)[0])
    pred_tonnes_ha = pred_q_acre * 0.2471  # Conversion: 1 Q/Acre = 2.471 Tonnes/Ha (100 kg / 0.404686 ha = 247.1 kg/ha)

    if not return_details:
        return round(pred_q_acre, 2)

    # State baseline reference
    state_avg = 267.42
    diff = pred_q_acre - state_avg

    if pred_q_acre >= 320.0:
        category = "High Yield (Optimal Agro-Ecological Management)"
    elif pred_q_acre >= 240.0:
        category = "Moderate Yield (Regional Standard)"
    else:
        category = "Below Average Yield (Moisture/Nutrient Constrained)"

    return {
        "predicted_yield_quintal_per_acre": round(pred_q_acre, 2),
        "predicted_yield_tonnes_per_hectare": round(pred_tonnes_ha, 2),
        "yield_category": category,
        "comparison_to_maharashtra_average": f"{'+' if diff >= 0 else ''}{diff:.2f} Quintals/Acre",
        "inputs_used": {
            "District": record.get("District", DEFAULT_CATEGORICAL_FEATURES["District"]),
            "Soil_Type": record.get("Soil_Type", DEFAULT_CATEGORICAL_FEATURES["Soil_Type"]),
            "Water_Quantity_liters_per_acre": record.get("Water_Quantity_liters_per_acre", DEFAULT_NUMERIC_FEATURES["Water_Quantity_liters_per_acre"]),
            "Fertilizer_Quantity": record.get("Fertilizer_Quantity", DEFAULT_NUMERIC_FEATURES["Fertilizer_Quantity"]),
            "Irrigation_Method": record.get("Irrigation_Method_Type", DEFAULT_CATEGORICAL_FEATURES["Irrigation_Method_Type"]),
            "Cane_Diameter_cm": record.get("Cane_Diameter_cm", DEFAULT_NUMERIC_FEATURES["Cane_Diameter_cm"])
        }
    }


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    print("=" * 75)
    print("MAHARASHTRA SUGARCANE YIELD PREDICTOR DEMONSTRATION")
    print("=" * 75)

    # Demo 1: High-Yield Kolhapur Farm
    kolhapur_farm = {
        'District': 'Kolhapur',
        'Soil_Type': 'Black',
        'Water_Quantity_liters_per_acre': 1500.0,
        'Fertilizer_Quantity': 220.0,
        'Irrigation_Method_Type': 'Drip',
        'Cane_Diameter_cm': 3.8,
        'Temp_Avg_C': 28.5
    }
    res1 = predict_sugarcane_yield(kolhapur_farm, return_details=True)
    print("\n[Scenario 1: Kolhapur Farm with Drip Irrigation]")
    print(f"Predicted Yield: {res1['predicted_yield_quintal_per_acre']} Quintals/Acre ({res1['predicted_yield_tonnes_per_hectare']} Tonnes/Ha)")
    print(f"Category:        {res1['yield_category']}")
    print(f"State Benchmark: {res1['comparison_to_maharashtra_average']}")

    # Demo 2: Semi-Arid Solapur Deficit Farm
    solapur_farm = {
        'District': 'Solapur',
        'Soil_Type': 'Sandy',
        'Water_Quantity_liters_per_acre': 850.0,
        'Fertilizer_Quantity': 130.0,
        'Irrigation_Method_Type': 'Flood',
        'Cane_Diameter_cm': 2.6,
        'Temp_Avg_C': 31.0
    }
    res2 = predict_sugarcane_yield(solapur_farm, return_details=True)
    print("\n[Scenario 2: Solapur Farm with Water Deficit]")
    print(f"Predicted Yield: {res2['predicted_yield_quintal_per_acre']} Quintals/Acre ({res2['predicted_yield_tonnes_per_hectare']} Tonnes/Ha)")
    print(f"Category:        {res2['yield_category']}")
    print(f"State Benchmark: {res2['comparison_to_maharashtra_average']}")

    # Demo 3: Quick Direct Keyword Arguments
    quick_yield = predict_sugarcane_yield(District="Pune", Water_Quantity_liters_per_acre=1350.0, Fertilizer_Quantity=190.0)
    print(f"\n[Scenario 3: Quick Keyword Query (Pune)]: Predicted Yield = {quick_yield} Quintals/Acre")
    print("=" * 75)
