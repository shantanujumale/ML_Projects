# 🌱 CaneVision — Maharashtra Sugarcane Intelligence

An interactive Streamlit web application built from the PBL project:

**Maharashtra Sugarcane Yield Prediction and Agro-Resource Optimization Using Explainable Machine Learning**

---

## 🚀 Key Features

- **🌾 Overview & Architecture**: Executive summary, dataset metrics, yield distributions, district rankings, and an interactive 9-stage notebook pipeline roadmap.
- **🔮 Yield Prediction Engine**: Practical farm input form (Location, Soil Properties, Climate, Nutrients, Crop Measurements) predicting yield in **Quintals/Acre** and **Tonnes/Hectare**.
- **🧬 Explainable AI (SHAP Insights)**:
  - Global SHAP feature importance plot (`shap_bar.png`) explaining overall model drivers.
  - Real-time local **TreeSHAP** waterfall/contribution charts for individual farm predictions.
- **🧪 Agro-Resource Optimizer**:
  - Benchmark what-if scenarios (`what_if_scenarios.csv`).
  - Interactive 2D Water × Fertilizer grid-search simulation with dynamic Plotly response surface heatmap and CSV export.
- **📈 Model Performance & Benchmarks**: Direct comparison of 5-Fold Cross-Validation and 20% Holdout Test results across Linear Ridge, Decision Tree, Random Forest, and XGBoost.
- **🔍 Dataset Explorer**: Interactive record viewer with district and season filters, district yield statistics, and full CSV download.

---

## 📁 Project Structure

```text
├── Maharashtra_Sugarcane_Yield_Prediction_and_Agro_Resource_Optimization.ipynb  # Executed project notebook
├── FINAL_SUGARCANE_DATASET.csv       # Source dataset (3,000 records, 70 columns)
├── sugarcane_yield_best_model.pkl    # Serialized champion XGBoost model pipeline
├── selected_features.csv             # 61 selected numeric features from permutation importance
├── model_comparison.csv             # Holdout test set performance comparison table
├── cross_validation_results.csv     # 5-fold cross-validation performance comparison table
├── what_if_scenarios.csv            # Agro-resource what-if simulation table
├── shap_bar.png                     # Exported global SHAP feature importance plot
├── model_metrics.json               # Serialized model performance metrics
├── app.py                           # Full Streamlit web application
├── train_model.py                   # Script to train and serialize models
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## 💻 Running the Web Application

The application requires Python 3.10+:

```powershell
# 1. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (if not already installed)
pip install -r requirements.txt

# 3. Start the Streamlit application
streamlit run app.py
```

The app will open automatically in your browser at:
`http://localhost:8501`

---

## 🔬 Workflow & Model Details

1. **Ingestion & Data Types**: Cleans raw data and standardizes sunshine duration (`hh:mm` to decimal hours).
2. **Feature Engineering**: Constructs 10 domain agronomic indicators (`Cane_Growth_Index`, `NPK_Total`, `Irrigation_Intensity`, `Temperature_SoilMoisture`, `Water_Retention_Potential`).
3. **Feature Selection**: Uses Random Forest permutation importance to select top numeric and categorical predictors.
4. **Model Benchmarks**: Trains Ridge Regression, Decision Tree, Random Forest, and XGBoost models.
5. **Cross-Validation**: 5-Fold CV on training split + untouched 20% holdout test evaluation.
6. **Hyperparameter Tuning**: `RandomizedSearchCV` optimization for Random Forest and XGBoost.
7. **Explainability (XAI)**: `TreeExplainer` from `shap` produces global and localized farm-level attributions.
8. **Agro-Resource Optimization**: Evaluates yield response across water and fertilizer gradients.
9. **Serialization**: Full pipeline export with `joblib`.

---

## 📝 Agronomic Interpretation Note

Model predictions and optimizer outputs represent empirical machine learning estimates derived from the 3,000-record Maharashtra dataset. They serve as valuable directional decision-support benchmarks. Field management decisions should always be cross-referenced with local soil tests, extension services, and agricultural specialists.
