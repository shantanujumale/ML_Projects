"""
Sugarcane Crop Yield Prediction - Model Training and Evaluation Pipeline
Algorithms: Linear Regression, Decision Tree Regressor, Random Forest Regressor, XGBoost Regressor
Metrics: MAE, MSE, RMSE, R²
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Enable UTF-8 encoding for console output
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def run_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'demo', 'FINAL_SUGARCANE_DATASET.csv')
    print("Loading dataset from:", dataset_path)
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    target = 'Yield_Quintal_per_Acre'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset!")

    drop_cols = [target, 'State', 'Sunshine_Hours_hh_mm', 'Month', 'Khasra_No', 'Planting_Date', 'Harvesting_Date']
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = feature_df.select_dtypes(include=['object']).columns.tolist()

    num_medians = {}
    for col in num_cols:
        med = float(feature_df[col].median())
        num_medians[col] = med
        feature_df[col] = feature_df[col].fillna(med)

    cat_mappings = {}
    for col in cat_cols:
        mode_val = str(feature_df[col].mode()[0]) if not feature_df[col].mode().empty else 'Unknown'
        feature_df[col] = feature_df[col].fillna(mode_val)
        
        unique_vals = sorted(feature_df[col].astype(str).unique().tolist())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        cat_mappings[col] = {
            'mapping': mapping,
            'default': mode_val,
            'classes': unique_vals
        }
        feature_df[col] = feature_df[col].astype(str).map(mapping).fillna(0).astype(int)

    y = df[target].fillna(df[target].median())
    X = feature_df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=7, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42)
    }

    results = []
    trained_models = {}

    header = f"{'Algorithm':<20} | {'Train R2':<10} | {'Test R2':<10} | {'Test MAE':<10} | {'Test MSE':<12} | {'Test RMSE':<10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_mse = mean_squared_error(y_test, test_preds)
        test_rmse = np.sqrt(test_mse)

        results.append({
            'Model': name,
            'Train R2': round(float(train_r2), 4),
            'Test R2': round(float(test_r2), 4),
            'Test MAE': round(float(test_mae), 2),
            'Test MSE': round(float(test_mse), 2),
            'Test RMSE': round(float(test_rmse), 2)
        })

        print(f"{name:<20} | {train_r2:<10.4f} | {test_r2:<10.4f} | {test_mae:<10.2f} | {test_mse:<12.2f} | {test_rmse:<10.2f}")

    print("=" * len(header))

    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(base_dir, 'model_benchmark_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved benchmark summary to: {results_csv_path}")

    best_model_name = results_df.sort_values(by='Test R2', ascending=False).iloc[0]['Model']
    best_model = trained_models[best_model_name]
    best_test_r2 = results_df.loc[results_df['Model'] == best_model_name, 'Test R2'].values[0]
    print(f"\nBest Performing Model: {best_model_name} (Test R2: {best_test_r2})")

    model_save_path = os.path.join(base_dir, 'best_sugarcane_model.pkl')
    metadata_save_path = os.path.join(base_dir, 'model_metadata.json')

    joblib.dump(best_model, model_save_path)
    print(f"Saved best model object to: {model_save_path}")

    metadata = {
        'best_model': best_model_name,
        'feature_names': list(X.columns),
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'num_medians': num_medians,
        'cat_mappings': cat_mappings,
        'benchmark_results': results
    }

    with open(metadata_save_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved preprocessing metadata to: {metadata_save_path}")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.barplot(data=results_df, x='Model', y='Test R2', ax=axes[0], palette='viridis')
        axes[0].set_title('Model Comparison - Test R2 Score (Higher is Better)', fontsize=12, fontweight='bold')
        axes[0].set_ylim(0, 1.0)
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                             ha='center', va='center', color='white', fontweight='bold')

        results_melted = results_df.melt(id_vars=['Model'], value_vars=['Test MAE', 'Test RMSE'], 
                                         var_name='Metric', value_name='Error')
        sns.barplot(data=results_melted, x='Model', y='Error', hue='Metric', ax=axes[1], palette='magma')
        axes[1].set_title('Model Error Comparison - MAE and RMSE (Lower is Better)', fontsize=12, fontweight='bold')

        plt.tight_layout()
        chart_path = os.path.join(base_dir, 'model_comparison_chart.png')
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"Saved performance chart to: {chart_path}")

        if hasattr(best_model, 'feature_importances_'):
            importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances.values, y=importances.index, palette='crest')
            plt.title(f'Top 15 Most Important Features ({best_model_name})', fontsize=12, fontweight='bold')
            plt.xlabel('Relative Feature Importance')
            plt.tight_layout()
            feat_chart_path = os.path.join(base_dir, 'feature_importance.png')
            plt.savefig(feat_chart_path, dpi=300)
            plt.close()
            print(f"Saved feature importance plot to: {feat_chart_path}")
    except Exception as e:
        print(f"Plotting note: {e}")

if __name__ == '__main__':
    run_pipeline()
