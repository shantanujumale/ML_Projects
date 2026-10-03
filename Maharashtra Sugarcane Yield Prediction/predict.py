"""
Sugarcane Yield Prediction Module & CLI
Provides real-time yield prediction for Maharashtra districts given agricultural inputs.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_sugarcane_model.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'model_metadata.json')

def load_predictor():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Model artifacts not found! Please run 'python train_models.py' first.")
    
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return model, metadata

def predict_yield(input_dict):
    """
    Takes a dictionary of input parameters and returns the predicted sugarcane yield.
    Yield is returned in Quintal/Acre and Metric Tons/Hectare.
    """
    model, metadata = load_predictor()
    feature_names = metadata['feature_names']
    num_medians = metadata['num_medians']
    cat_mappings = metadata['cat_mappings']

    row = {}
    for feat in feature_names:
        if feat in input_dict and input_dict[feat] is not None:
            raw_val = input_dict[feat]
            if feat in cat_mappings:
                mapping = cat_mappings[feat]['mapping']
                encoded_val = mapping.get(str(raw_val), 0)
                row[feat] = encoded_val
            else:
                try:
                    row[feat] = float(raw_val)
                except (ValueError, TypeError):
                    row[feat] = num_medians.get(feat, 0.0)
        else:
            # Impute default
            if feat in cat_mappings:
                default_cat = cat_mappings[feat]['default']
                row[feat] = cat_mappings[feat]['mapping'].get(default_cat, 0)
            else:
                row[feat] = num_medians.get(feat, 0.0)

    input_df = pd.DataFrame([row])[feature_names]
    pred_quintal = float(model.predict(input_df)[0])
    
    # Clip to realistic non-negative yield range
    pred_quintal = max(10.0, round(pred_quintal, 2))
    
    # 1 Quintal = 100 kg = 0.1 Metric Ton. 1 Acre = 0.404686 Hectare.
    # Yield (Tons/Ha) = Yield (Quintal/Acre) * 0.1 / 0.404686 = Quintal/Acre * 0.2471
    pred_ton_per_ha = round(pred_quintal * 0.2471, 2)

    return {
        'yield_quintal_per_acre': pred_quintal,
        'yield_ton_per_hectare': pred_ton_per_ha,
        'district': input_dict.get('District', 'Not Specified'),
        'model_used': metadata['best_model']
    }

if __name__ == '__main__':
    # Test sample prediction
    sample_input = {
        'District': 'Kolhapur',
        'Season': 'Kharif',
        'Soil_Type': 'Clay',
        'Soil_pH': 7.2,
        'Soil_Moisture_%': 35.0,
        'Nitrogen_kg_per_acre': 150.0,
        'Phosphorus_kg_per_acre': 60.0,
        'Potassium_kg_per_acre': 120.0,
        'Temp_Avg_C': 28.5,
        'Rainfall_Total_mm': 1450.0,
        'Humidity_%': 75.0,
        'Irrigation_Method_Type': 'Drip',
        'Irrigation_Frequency_Level': 'Medium',
        'Variety': 'Co0238',
        'Cane_Height_cm': 280.0,
        'Cane_Diameter_cm': 3.5,
        'Brix_Value': 21.0,
        'Disease_Severity': 'Low',
        'Pest_Level': 'Low'
    }
    res = predict_yield(sample_input)
    print("Prediction Result:")
    for k, v in res.items():
        print(f"  {k}: {v}")
