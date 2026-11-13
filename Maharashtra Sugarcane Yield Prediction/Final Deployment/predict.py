"""
Sugarcane Yield Prediction Module
"Maharashtra Sugarcane Yield Prediction and Agro-Resource Optimization Using Explainable Machine Learning"

Provides a simple, production-ready function `predict_sugarcane_yield()`
to predict sugarcane yield (Quintals/Acre & Tonnes/Ha) from farm inputs.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from typing import Union, Dict, Any, List

# Cache loaded model globally
_MODEL_CACHE = None

# Baseline defaults derived from the 3,000-farm Maharashtra dataset
DEFAULT_NUMERIC_FEATURES = {
    'Brix_Value': 19.5,
    'Cane_Diameter_cm': 3.5,
    'Cane_Height_cm': 250.0,
    'Clay_%': 24.51,
    'Copper_mg_per_kg': 1.61,
    'Crop_Duration_Days': 325.0,
    'Dew_Point_C': 17.38,
    'EC': 1.28,
    'Evapotranspiration_mm_day': 5.02,
    'Fertilizer_Quantity': 175.60,
    'Frost_Days': 0.0,
    'Germination_%': 82.0,
    'Groundwater_Level_meters': 10.68,
    'Heat_Stress_Days': 12.0,
    'Humidity_%': 65.0,
    'Internode_Length_cm': 12.5,
    'Iron_mg_per_kg': 5.8,
    'Manganese_mg_per_kg': 4.2,
    'Nitrogen_kg_per_acre': 152.74,
    'Organic_Carbon_%': 0.65,
    'Pesticide_Usage': 2.5,
    'Phosphorus_kg_per_acre': 69.74,
    'Plant_Density': 38000.0,
    'Potassium_kg_per_acre': 115.66,
    'Rainfall_Seasonal_mm': 790.83,
    'Rainfall_Total_mm': 850.0,
    'Row_Gap_cm': 105.61,
    'Row_Spacing_cm': 90.0,
    'Sand_%': 40.23,
    'Seed_Rate': 30000.0,
    'Silt_%': 35.26,
    'Soil_Depth_cm': 65.0,
    'Soil_Moisture_%': 32.0,
    'Soil_pH': 7.24,
    'Solar_Radiation_MJ_m2_day': 18.5,
    'Sulfur_kg_per_acre': 17.16,
    'Sunshine_Hours_Decimal': 7.8,
    'Temp_Avg_C': 27.84,
    'Temp_Max_C': 33.23,
    'Temp_Min_C': 22.31,
    'Tillering_Count': 7.0,
    'Water_Holding_Capacity_%': 45.2,
    'Water_Quantity_liters_per_acre': 1229.68,
    'Wind_Speed_kmph': 6.2,
    'Zinc_mg_per_kg': 1.8
}

DEFAULT_CATEGORICAL_FEATURES = {
    'District': 'Kolhapur',
    'Season': 'Kharif',
    'Soil_Type': 'Black',
    'Irrigation_Method_Type': 'Drip',
    'Irrigation_Frequency_Level': 'Medium',
    'Fertilizer_Type': 'DAP',
    'Month': 'October'
}


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


def _prepare_single_row(user_dict: Dict[str, Any], model) -> pd.DataFrame:
    """Prepares and validates a single farm input dictionary with defaults and derived features."""
    row = {}
    row.update(DEFAULT_NUMERIC_FEATURES)
    row.update(DEFAULT_CATEGORICAL_FEATURES)
    if user_dict:
        row.update(user_dict)

    # 1. Month encoding
    month_map = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    m_val = str(row.get('Month', 'October')).strip().lower()
    m_num = month_map.get(m_val, 10)
    row['Month_Num'] = m_num
    row['Month_Sin'] = float(np.sin(2 * np.pi * m_num / 12.0))
    row['Month_Cos'] = float(np.cos(2 * np.pi * m_num / 12.0))

    if 'Sunshine_Hours_Decimal' not in row:
        row['Sunshine_Hours_Decimal'] = 7.5

    # 2. Temperature range & Climate stress
    row['Temperature_Range'] = float(row.get('Temp_Max_C', 33.2) - row.get('Temp_Min_C', 22.3))
    row['Climate_Stress'] = float(row.get('Heat_Stress_Days', 0) + row.get('Frost_Days', 0))

    # 3. Soil Water Retention Potential
    row['Water_Retention_Potential'] = float((row.get('Soil_Depth_cm', 65.0) * row.get('Water_Holding_Capacity_%', 45.0)) / 100.0)

    # 4. Stoichiometric Nutrient Ratios
    n = float(row.get('Nitrogen_kg_per_acre', 150.0))
    p = float(row.get('Phosphorus_kg_per_acre', 70.0))
    k = float(row.get('Potassium_kg_per_acre', 115.0))
    row['N_to_P'] = n / (p + 1e-4)
    row['N_to_K'] = n / (k + 1e-4)
    row['P_to_K'] = p / (k + 1e-4)
    row['NPK_Total'] = n + p + k

    # 5. Irrigation Intensity
    freq_map = {'low': 1.0, 'medium': 2.0, 'high': 3.0}
    freq_str = str(row.get('Irrigation_Frequency_Level', 'Medium')).strip().lower()
    freq_num = freq_map.get(freq_str, 2.0)
    row['Irrigation_Intensity'] = float(freq_num * row.get('Water_Quantity_liters_per_acre', 1200.0))

    # 6. Cane Growth & Morphology
    h = float(row.get('Cane_Height_cm', 250.0))
    d = float(row.get('Cane_Diameter_cm', 3.5))
    internode_len = float(row.get('Internode_Length_cm', 12.0))
    row['Cane_Growth_Index'] = float(h * d)
    row['Internode_Height_Ratio'] = float(internode_len / (h + 1e-4))

    # 7. Interaction terms
    temp_avg = float(row.get('Temp_Avg_C', 28.0))
    rain_tot = float(row.get('Rainfall_Total_mm', 850.0))
    soil_moist = float(row.get('Soil_Moisture_%', 30.0))
    fert_qty = float(row.get('Fertilizer_Quantity', 175.0))
    water_qty = float(row.get('Water_Quantity_liters_per_acre', 1200.0))

    row['Rainfall_Temperature'] = float(rain_tot * temp_avg)
    row['Temperature_SoilMoisture'] = float(soil_moist * temp_avg)
    row['Fertilizer_Irrigation'] = float(fert_qty * water_qty)

    # Align with model expectations
    expected_cols = list(model.feature_names_in_)
    df_out = pd.DataFrame([row])[expected_cols]
    return df_out


def predict_sugarcane_yield(
    input_data: Union[Dict[str, Any], pd.DataFrame, None] = None,
    model_path: str = None,
    return_details: bool = False,
    **kwargs
) -> Union[float, Dict[str, Any], np.ndarray]:
    """
    Predicts sugarcane yield for Maharashtra farms.

    Parameters:
    -----------
    input_data : dict or pd.DataFrame, optional
        A dictionary containing farm input parameters (e.g. District, Water_Quantity, Fertilizer).
    model_path : str, optional
        Path to .pkl model pipeline.
    return_details : bool, default False
        If True, returns a structured dictionary with metrics and benchmark comparison.
    **kwargs :
        Direct key-value inputs (e.g., District="Kolhapur", Water_Quantity_liters_per_acre=1400).
    """
    model = get_model(model_path)

    # Resolve input dictionary
    record = {}
    if isinstance(input_data, dict):
        record.update(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df_prepared = input_data.copy()
        pred = model.predict(df_prepared)
        return pred

    if kwargs:
        record.update(kwargs)

    df_single = _prepare_single_row(record, model)
    pred_q_acre = float(model.predict(df_single)[0])
    pred_tonnes_ha = pred_q_acre * 0.2471  # 1 Q/Acre = 0.2471 Tonnes/Ha

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
