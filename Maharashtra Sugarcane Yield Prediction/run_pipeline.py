"""
Master Pipeline Runner for:
Maharashtra Sugarcane Yield Prediction and Agro-Resource Optimization Using Explainable Machine Learning
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_clean_data
from feature_engineering import create_engineered_features, get_stage_features, create_preprocessor
from model_trainer import train_and_benchmark
from explainability import SugarcaneExplainer
from resource_optimizer import AgroResourceOptimizer

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')

def run_all():
    print("=" * 80)
    print("MAHARASHTRA SUGARCANE YIELD PREDICTION & AGRO-RESOURCE OPTIMIZATION")
    print("           END-TO-END EXPLAINABLE MACHINE LEARNING PIPELINE        ")
    print("=" * 80)

    # 1. Ingestion & EDA plots
    df = load_clean_data()
    print(f"\n1. Ingested Clean Dataset: {df.shape[0]} rows, {df.shape[1]} columns across 14 Maharashtra districts.")

    # Target plot
    plt.figure(figsize=(10, 4.5))
    sns.histplot(df['Yield_Quintal_per_Acre'], kde=True, color='#16a34a', bins=30)
    plt.title("Sugarcane Yield Distribution (Quintals/Acre)", fontweight='bold')
    plt.xlabel("Yield (Quintals/Acre)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_target_distribution.png'), dpi=300)
    plt.savefig(os.path.join(ROOT_DIR, 'eda_target_distribution.png'), dpi=300)
    plt.close()

    # District plot
    plt.figure(figsize=(12, 5))
    dist_order = df.groupby('District')['Yield_Quintal_per_Acre'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='District', y='Yield_Quintal_per_Acre', order=dist_order, color='#22c55e')
    plt.xticks(rotation=40, ha='right')
    plt.title("Sugarcane Yield Distribution Across 14 Maharashtra Districts", fontweight='bold')
    plt.ylabel("Yield (Quintals/Acre)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_district_yield.png'), dpi=300)
    plt.savefig(os.path.join(ROOT_DIR, 'eda_district_yield.png'), dpi=300)
    plt.close()

    # 2. Train and Benchmark Models
    champion_pipeline, eval_df, cv_df = train_and_benchmark(output_dir=OUTPUT_DIR)

    # Copy CSVs to root
    for f in ['model_comparison.csv', 'cross_validation_results.csv', 'permutation_importance.csv', 'selected_features.csv']:
        src = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(src):
            pd.read_csv(src).to_csv(os.path.join(ROOT_DIR, f), index=False)

    # 3. Explainable AI with SHAP
    print("\n>>> Computing SHAP Explanations...", flush=True)
    bundle_path = os.path.join(OUTPUT_DIR, 'model_data_bundle.joblib')
    explainer = SugarcaneExplainer(bundle_path=bundle_path)
    imp_df = explainer.get_global_importance(n_samples=100)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df.head(15), y='Feature', x='Mean_Abs_SHAP', color='#16a34a')
    plt.title("Top 15 Global Features by Mean |SHAP Value| (Yield Impact in Q/Acre)", fontweight='bold')
    plt.xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_bar.png'), dpi=300)
    plt.savefig(os.path.join(ROOT_DIR, 'shap_bar.png'), dpi=300)
    plt.close()

    # 4. Agro-Resource What-If and Optimization
    print("\n>>> Running Agro-Resource Optimization & Scenarios...", flush=True)
    optimizer = AgroResourceOptimizer(bundle_path=bundle_path)
    bundle = joblib.load(bundle_path)
    sample_farmer = bundle['X_test_sel'].iloc[0]

    sc_df = optimizer.run_what_if_scenarios(sample_farmer)
    sc_df.to_csv(os.path.join(OUTPUT_DIR, 'what_if_scenarios.csv'), index=False)
    sc_df.to_csv(os.path.join(ROOT_DIR, 'what_if_scenarios.csv'), index=False)

    opt_res = optimizer.optimize_resources(sample_farmer)
    print("Baseline:", opt_res['baseline'])
    print("Optimized:", opt_res['optimized'])
    print("Impact:", opt_res['comparison'])

    print("\n Pipeline run completed successfully. All artifacts and models saved.")

if __name__ == '__main__':
    run_all()
