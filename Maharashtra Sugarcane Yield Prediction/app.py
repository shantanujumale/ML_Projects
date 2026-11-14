from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent

# Target variable definition
TARGET = "Yield_Quintal_per_Acre"


def resolve_path(*candidates: Path) -> Path:
    """Return the first existing candidate path or the first path in the list."""
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


DATA_PATH = resolve_path(
    ROOT / "FINAL_SUGARCANE_DATASET.csv",
    ROOT / "data" / "FINAL_SUGARCANE_DATASET.csv",
)
MODEL_PATH = resolve_path(
    ROOT / "sugarcane_yield_best_model.pkl",
    ROOT / "artifacts" / "sugarcane_yield_best_model.pkl",
)
METRICS_PATH = resolve_path(
    ROOT / "model_metrics.json",
    ROOT / "artifacts" / "model_metrics.json",
)
MODEL_COMP_PATH = resolve_path(
    ROOT / "model_comparison.csv",
    ROOT / "artifacts" / "model_comparison.csv",
)
CV_RESULTS_PATH = resolve_path(
    ROOT / "cross_validation_results.csv",
    ROOT / "artifacts" / "cross_validation_results.csv",
)
WHAT_IF_PATH = resolve_path(
    ROOT / "what_if_scenarios.csv",
    ROOT / "artifacts" / "what_if_scenarios.csv",
)
SELECTED_FEATURES_PATH = resolve_path(
    ROOT / "selected_features.csv",
    ROOT / "artifacts" / "selected_features.csv",
)
SHAP_PLOT_PATH = resolve_path(
    ROOT / "shap_bar.png",
    ROOT / "artifacts" / "shap_bar.png",
)

st.set_page_config(
    page_title="CaneVision | Maharashtra Sugarcane Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Theme Custom Styling
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      
      html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: #e5ece7;
      }
      .stApp {
        background-color: #090d0b !important;
        background: linear-gradient(180deg, #090d0b 0%, #0c130f 50%, #0a0f0d 100%) !important;
        color: #e5ece7 !important;
      }
      
      h1, h2, h3, h4, h5, h6 {
        font-family: 'Manrope', sans-serif !important;
        color: #f1f7f3 !important;
      }
      
      p, span, label {
        color: #c9d8ce !important;
      }
      
      [data-testid="stSidebar"] {
        background-color: #060a08 !important;
        border-right: 1px solid #16261d !important;
      }
      [data-testid="stSidebar"] * {
        color: #e5ece7 !important;
      }
      
      .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 16px;
        color: white;
        background: linear-gradient(135deg, #071f14 0%, #0f3d27 55%, #185e3a 100%);
        margin-bottom: 1.3rem;
        border: 1px solid #1a4d33;
        box-shadow: 0 4px 22px rgba(0, 0, 0, 0.45);
      }
      .hero h1 {
        color: #ffffff !important;
        margin: 0.15rem 0 0.35rem 0;
        font-size: 2.05rem;
      }
      .hero p {
        color: #cdecd6 !important;
        margin: 0;
        font-size: 1rem;
      }
      .eyebrow {
        color: #4ade80 !important;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        font-size: .74rem;
      }
      
      .metric-card {
        background: #111a15 !important;
        border: 1px solid #1b2f23 !important;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
      }
      .metric-label {
        font-size: .74rem;
        color: #8da496 !important;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 600;
      }
      .metric-value {
        font: 800 1.6rem 'Manrope', sans-serif;
        color: #4ade80 !important;
        margin-top: .3rem;
      }
      .metric-help {
        font-size: .78rem;
        color: #728c7c !important;
        margin-top: .2rem;
      }
      
      .callout {
        border-left: 4px solid #22c55e;
        background: #0d2217;
        padding: .95rem 1.2rem;
        border-radius: 8px;
        color: #d1fae5 !important;
        font-size: .94rem;
        border: 1px solid #163d28;
        border-left-width: 4px;
      }
      .callout * {
        color: #d1fae5 !important;
      }
      
      .warning-box {
        border-left: 4px solid #eab308;
        background: #231c07;
        padding: .95rem 1.2rem;
        border-radius: 8px;
        color: #fef08a !important;
        font-size: .93rem;
        border: 1px solid #3d310e;
        border-left-width: 4px;
      }
      .warning-box * {
        color: #fef08a !important;
      }
      
      div[data-testid="stForm"] {
        background: #111a15 !important;
        padding: 1.35rem 1.5rem;
        border: 1px solid #1b2f23 !important;
        border-radius: 14px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.3);
      }
      
      .stButton > button, .stFormSubmitButton > button {
        background: #16a34a !important;
        color: #ffffff !important;
        border: 0 !important;
        border-radius: 8px;
        font-weight: 700;
        transition: all 0.2s ease;
      }
      .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #22c55e !important;
        color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.35);
      }
      
      [data-testid="stMetric"] {
        background: #111a15 !important;
        border: 1px solid #1b2f23 !important;
        border-radius: 12px;
        padding: .85rem 1.1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
      }
      [data-testid="stMetricValue"] {
        color: #4ade80 !important;
        font-weight: 800 !important;
      }
      [data-testid="stMetricLabel"] {
        color: #8da496 !important;
      }
      
      [data-testid="stDataFrame"] {
        background: #111a15 !important;
        border: 1px solid #1b2f23 !important;
        border-radius: 10px;
      }
      
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_dataframe(df: pd.DataFrame, **kwargs) -> None:
    """Render dataframe safely using st.dataframe, falling back to markdown table if needed."""
    try:
        st.dataframe(df, **kwargs)
    except Exception:
        try:
            st.table(df.head(50))
        except Exception:
            st.markdown(df.head(50).to_markdown())


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Ensure Plotly charts seamlessly blend into the dark palette."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111a15",
        plot_bgcolor="#111a15",
        font={"color": "#d1ded5", "family": "DM Sans"},
    )
    return fig


def parse_sunshine(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    val_str = str(value).strip()
    if ":" in val_str:
        try:
            h, m = val_str.split(":", 1)
            return float(h) + float(m) / 60.0
        except (ValueError, TypeError):
            return np.nan
    try:
        return float(val_str)
    except (ValueError, TypeError):
        return np.nan


def parse_month(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    for prefix, num in month_map.items():
        if text.startswith(prefix):
            return float(num)
    try:
        return float(text)
    except (ValueError, TypeError):
        return np.nan


def prepare_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "Sunshine_Hours_hh_mm" in df.columns:
        df["Sunshine_Hours_Decimal"] = df["Sunshine_Hours_hh_mm"].apply(parse_sunshine)
        df.drop(columns=["Sunshine_Hours_hh_mm"], inplace=True)
    if "Month" in df.columns:
        df["Month_Num"] = df["Month"].apply(parse_month).fillna(6.0)

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
    freq_map = {"low": 1.0, "medium": 2.0, "high": 3.0}
    df["Irrig_Freq_Num"] = (
        df["Irrigation_Frequency_Level"].astype(str).str.lower().map(freq_map).fillna(2.0)
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


@st.cache_data(show_spinner="Loading Maharashtra sugarcane dataset...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")
    raw = pd.read_csv(DATA_PATH)
    prepared = prepare_dataset(raw)
    return raw, prepared


@st.cache_resource(show_spinner="Loading trained yield prediction model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model artifact missing at {MODEL_PATH}.")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        res = float(value)
        return res if np.isfinite(res) else default
    except (TypeError, ValueError):
        return default


def get_reference_row(data: pd.DataFrame, district: str) -> pd.Series:
    """Build a representative profile from one district's medians and modes."""
    subset = data[data["District"].astype(str) == str(district)]
    if subset.empty:
        subset = data
    numeric = subset.select_dtypes(include=[np.number]).median(numeric_only=True)
    row = numeric.to_dict()
    for col in subset.select_dtypes(exclude=[np.number]).columns:
        m = subset[col].mode(dropna=True)
        row[col] = m.iloc[0] if not m.empty else "Unknown"
    row["District"] = district
    return pd.Series(row)


def refresh_engineered_features(row: pd.Series) -> pd.Series:
    """Recompute all derived interaction features after inputs are updated."""
    row = row.copy()
    number = lambda name, default=0.0: safe_float(row.get(name, default), default)
    month = int(np.clip(round(number("Month_Num", 6.0)), 1, 12))
    row["Month_Num"] = month
    row["Temperature_Range"] = number("Temp_Max_C") - number("Temp_Min_C")
    row["Climate_Stress"] = number("Heat_Stress_Days") + number("Frost_Days")
    row["Water_Retention_Potential"] = (
        number("Soil_Depth_cm") * number("Water_Holding_Capacity_%") / 100.0
    )
    nitrogen = number("Nitrogen_kg_per_acre")
    phosphorus = number("Phosphorus_kg_per_acre")
    potassium = number("Potassium_kg_per_acre")
    water = number("Water_Quantity_liters_per_acre")
    fertilizer = number("Fertilizer_Quantity")
    temp_avg = number("Temp_Avg_C")
    irrig_freq = str(row.get("Irrigation_Frequency_Level", "Medium")).strip().lower()
    freq_map = {"low": 1.0, "medium": 2.0, "high": 3.0}
    row["Irrig_Freq_Num"] = freq_map.get(irrig_freq, 2.0)
    row["Irrigation_Intensity"] = row["Irrig_Freq_Num"] * water
    row["N_to_P"] = nitrogen / (phosphorus + 1e-4)
    row["N_to_K"] = nitrogen / (potassium + 1e-4)
    row["P_to_K"] = phosphorus / (potassium + 1e-4)
    row["NPK_Total"] = nitrogen + phosphorus + potassium
    row["Cane_Growth_Index"] = number("Cane_Height_cm") * number("Cane_Diameter_cm")
    row["Internode_Height_Ratio"] = number("Internode_Length_cm") / (number("Cane_Height_cm") + 1e-4)
    row["Month_Sin"] = float(np.sin(2 * np.pi * month / 12.0))
    row["Month_Cos"] = float(np.cos(2 * np.pi * month / 12.0))
    row["Rainfall_Temperature"] = number("Rainfall_Total_mm") * temp_avg
    row["Temperature_SoilMoisture"] = number("Soil_Moisture_%") * temp_avg
    row["Fertilizer_Irrigation"] = fertilizer * water
    return row


def prediction_frame(model, row: pd.Series) -> pd.DataFrame:
    prep = model.named_steps["prep"]
    numeric_features = list(prep.transformers_[0][2])
    categorical_features = list(prep.transformers_[1][2])
    feature_columns = numeric_features + categorical_features
    d = row.to_dict()
    for col in feature_columns:
        if col not in d:
            d[col] = np.nan
    return pd.DataFrame([{c: d[c] for c in feature_columns}], columns=feature_columns)


def predict_row(model, row: pd.Series) -> float:
    refreshed = refresh_engineered_features(row)
    frame = prediction_frame(model, refreshed)
    return float(model.predict(frame)[0])


def make_shap_explanation(model, row: pd.Series) -> tuple[pd.DataFrame, float] | None:
    try:
        import shap

        frame = prediction_frame(model, refresh_engineered_features(row))
        prep = model.named_steps["prep"]
        transformed = prep.transform(frame)
        underlying = model.named_steps["model"]
        explainer = shap.TreeExplainer(underlying)
        values = np.asarray(explainer.shap_values(transformed)).reshape(-1)
        feature_names = [
            str(n).replace("num__", "").replace("cat__", "")
            for n in prep.get_feature_names_out()
        ]
        df_exp = pd.DataFrame({"Feature": feature_names, "SHAP impact": values})
        df_exp["Direction"] = np.where(df_exp["SHAP impact"] >= 0, "Increases prediction", "Decreases prediction")
        df_exp["Magnitude"] = df_exp["SHAP impact"].abs()
        df_exp = df_exp.sort_values("Magnitude", ascending=False).head(12).sort_values("SHAP impact")
        base_value = explainer.expected_value
        if isinstance(base_value, (list, tuple, np.ndarray)):
            base_value = float(np.asarray(base_value).reshape(-1)[0])
        return df_exp, float(base_value)
    except Exception as exc:
        st.warning(f"SHAP explanation note: {exc}")
        return None


def hero(title: str, subtitle: str, eyebrow: str = "AGRI INTELLIGENCE PLATFORM") -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-help">{help_text}</div></div>',
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------
# 1. OVERVIEW PAGE
# -------------------------------------------------------------
def render_overview(raw: pd.DataFrame, data: pd.DataFrame, metrics: dict[str, Any]) -> None:
    hero(
        "Maharashtra Sugarcane Intelligence",
        "Predict crop yield, explain agronomic drivers with SHAP, and optimize water and fertilizer resources.",
        "PROJECT WORKBENCH",
    )
    st.markdown(
        "<p style='color: #c9d8ce; font-size: 1.02rem; line-height: 1.6;'>"
        "Welcome to <b style='color: #4ade80;'>CaneVision</b>, the interactive web implementation of the machine learning study "
        "<i>'Maharashtra Sugarcane Yield Prediction and Agro-Resource Optimization Using Explainable Machine Learning'</i>. "
        "The application integrates data ingestion, agronomic feature engineering, cross-validated multi-model benchmarks, "
        "Explainable AI (TreeSHAP), and prescriptive water–fertilizer optimization."
        "</p>",
        unsafe_allow_html=True,
    )

    district_count = int(raw["District"].nunique()) if "District" in raw else 0
    avg_yield = float(raw[TARGET].mean())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Dataset Records", f"{len(raw):,}", "Field samples from Maharashtra")
    with c2:
        metric_card("Model Features", "66 Features", "61 Numeric + 5 Categorical selected")
    with c3:
        metric_card("Districts Covered", str(district_count), "Major sugarcane cane belts")
    with c4:
        metric_card("Benchmark Yield", f"{avg_yield:.1f} Q/Acre", f"~{avg_yield * 0.2471:.1f} Tonnes / Hectare")

    st.markdown("<h3 style='margin-top: 1.5rem;'>📊 Field Data & Yield Profile</h3>", unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])
    with left:
        hist = px.histogram(
            raw, x=TARGET, nbins=35, marginal="box",
            title="Sugarcane Yield Distribution (Quintals/Acre)",
            color_discrete_sequence=["#22c55e"],
        )
        hist.update_layout(xaxis_title="Yield (Quintals/Acre)", yaxis_title="Number of Farms", bargap=0.08)
        apply_dark_theme(hist)
        st.plotly_chart(hist, use_container_width=True)

    with right:
        district_yield = (
            raw.groupby("District", as_index=False)[TARGET].mean().sort_values(TARGET, ascending=True)
        )
        district_fig = px.bar(
            district_yield, x=TARGET, y="District", orientation="h",
            title="Average Sugarcane Yield by District", color=TARGET,
            color_continuous_scale=["#143a26", "#22c55e", "#4ade80"],
        )
        district_fig.update_layout(xaxis_title="Mean Yield (Quintals/Acre)", yaxis_title="")
        apply_dark_theme(district_fig)
        st.plotly_chart(district_fig, use_container_width=True)

    st.markdown("<h3 style='margin-top: 1.5rem;'>🔄 9-Stage Notebook Workflow Architecture</h3>", unsafe_allow_html=True)
    stages = [
        ("01", "Data Ingestion", "3,000 records, sunshine parsing, null verification"),
        ("02", "Feature Engineering", "10 domain features: NPK ratio, Cane Growth Index"),
        ("03", "Feature Selection", "RandomForest permutation importance (66 top features)"),
        ("04", "Model Training", "Ridge, Decision Tree, Random Forest, XGBoost"),
        ("05", "Cross-Validation", "Strict 5-Fold CV on training split + holdout test"),
        ("06", "Hyperparameter Tuning", "RandomizedSearchCV on RF and XGBoost pipelines"),
        ("07", "Explainable AI", "SHAP TreeExplainer global & single-farm waterfall"),
        ("08", "Resource Optimization", "What-if scenarios & 2D water-fertilizer grid search"),
        ("09", "Serialization", "Pipeline artifact export & production reload validation"),
    ]
    cols1 = st.columns(5)
    for col, (num, title, desc) in zip(cols1, stages[:5]):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">STAGE {num}</div>'
                f'<div style="font-weight:700;color:#f1f7f3;margin:.3rem 0;font-size:0.92rem">{title}</div>'
                f'<div style="font-size:0.8rem;color:#8fa597">{desc}</div></div>',
                unsafe_allow_html=True,
            )
    cols2 = st.columns(4)
    for col, (num, title, desc) in zip(cols2, stages[5:]):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">STAGE {num}</div>'
                f'<div style="font-weight:700;color:#f1f7f3;margin:.3rem 0;font-size:0.92rem">{title}</div>'
                f'<div style="font-size:0.8rem;color:#8fa597">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="callout"><b>Production Pipeline Status:</b> The champion model pipeline '
        '<code>sugarcane_yield_best_model.pkl</code> is active and loaded. '
        'Use the sidebar to test predictions on custom farm parameters, simulate resource changes, or inspect the dataset.</div>',
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------
# 2. YIELD PREDICTION PAGE
# -------------------------------------------------------------
def render_prediction(raw: pd.DataFrame, data: pd.DataFrame, model) -> None:
    hero("Yield Prediction Engine", "Enter farm conditions to predict expected sugarcane yield and review local SHAP drivers.", "PREDICTIVE ANALYTICS")

    districts = sorted(raw["District"].dropna().astype(str).unique().tolist())

    col_dist, col_info = st.columns([1.2, 2.5])
    with col_dist:
        district = st.selectbox(
            "Select Farm District",
            districts,
            index=districts.index("Pune") if "Pune" in districts else 0,
            help="Selecting a district automatically pre-fills the typical baseline soil and climate profile for that region.",
        )
    with col_info:
        district_mean_init = float(raw.loc[raw["District"].astype(str) == district, TARGET].mean())
        st.markdown(
            f'<div class="callout" style="padding: 0.65rem 1rem; margin-top: 0.2rem;">'
            f'Baseline loaded for <b>{district}</b> (Historical District Mean: <b>{district_mean_init:.1f} Q/Acre</b>). '
            f'Adjust the inputs below to calculate your customized field estimate.</div>',
            unsafe_allow_html=True,
        )

    district_base = get_reference_row(data, district)

    with st.form("yield_prediction_form"):
        st.markdown("#### 1. Geographic & Field Profile")
        a, b, c = st.columns(3)
        with a:
            season_values = sorted(raw["Season"].dropna().astype(str).unique().tolist()) if "Season" in raw else ["Kharif", "Rabi"]
            season = st.selectbox("Season", season_values, index=0)
            soil_values = sorted(raw["Soil_Type"].dropna().astype(str).unique().tolist()) if "Soil_Type" in raw else ["Black"]
            soil_type = st.selectbox("Soil Type", soil_values)
        with b:
            irrigation_values = sorted(raw["Irrigation_Method_Type"].dropna().astype(str).unique().tolist()) if "Irrigation_Method_Type" in raw else ["Drip", "Flood"]
            irrigation_method = st.selectbox("Irrigation Method", irrigation_values)
            fertilizer_types = sorted(raw["Fertilizer_Type"].dropna().astype(str).unique().tolist()) if "Fertilizer_Type" in raw else ["Urea"]
            fertilizer_type = st.selectbox("Fertilizer Type", fertilizer_types)
        with c:
            month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            month_label = st.selectbox("Crop Month", month_labels, index=5)
            irrigation_frequency = st.selectbox("Irrigation Frequency", ["Low", "Medium", "High"], index=1)

        st.markdown("#### 2. Climate & Soil Properties")
        c1, c2, c3 = st.columns(3)
        with c1:
            rainfall = st.number_input("Total Rainfall (mm)", value=round(safe_float(district_base.get("Rainfall_Total_mm"), 750), 1), step=25.0)
            temp_min = st.number_input("Min Temperature (°C)", value=round(safe_float(district_base.get("Temp_Min_C"), 20), 1), step=0.5)
            temp_max = st.number_input("Max Temperature (°C)", value=round(safe_float(district_base.get("Temp_Max_C"), 35), 1), step=0.5)
        with c2:
            temp_avg = st.number_input("Avg Temperature (°C)", value=round(safe_float(district_base.get("Temp_Avg_C"), 27.5), 1), step=0.5)
            soil_moisture = st.number_input("Soil Moisture (%)", value=round(safe_float(district_base.get("Soil_Moisture_%"), 25.0), 1), step=1.0)
            soil_ph = st.number_input("Soil pH", value=round(safe_float(district_base.get("Soil_pH"), 7.2), 2), step=0.1)
        with c3:
            soil_depth = st.number_input("Soil Depth (cm)", value=round(safe_float(district_base.get("Soil_Depth_cm"), 60), 1), step=5.0)
            water_capacity = st.number_input("Water Holding Capacity (%)", value=round(safe_float(district_base.get("Water_Holding_Capacity_%"), 35.0), 1), step=1.0)
            heat_days = st.number_input("Heat Stress Days", value=int(round(safe_float(district_base.get("Heat_Stress_Days"), 5))), min_value=0, max_value=365, step=1)

        st.markdown("#### 3. Nutrients, Irrigation & Crop Growth")
        d1, d2, d3 = st.columns(3)
        with d1:
            nitrogen = st.number_input("Nitrogen (kg/acre)", value=round(safe_float(district_base.get("Nitrogen_kg_per_acre"), 135), 1), step=5.0)
            phosphorus = st.number_input("Phosphorus (kg/acre)", value=round(safe_float(district_base.get("Phosphorus_kg_per_acre"), 55), 1), step=5.0)
            potassium = st.number_input("Potassium (kg/acre)", value=round(safe_float(district_base.get("Potassium_kg_per_acre"), 80), 1), step=5.0)
        with d2:
            water = st.number_input("Applied Water (liters/acre)", value=round(safe_float(district_base.get("Water_Quantity_liters_per_acre"), 1600), 1), step=50.0)
            fertilizer_quantity = st.number_input("Total Fertilizer (kg/acre)", value=round(safe_float(district_base.get("Fertilizer_Quantity"), 190), 1), step=5.0)
            cane_height = st.number_input("Cane Height (cm)", value=round(safe_float(district_base.get("Cane_Height_cm"), 220), 1), step=5.0)
        with d3:
            cane_diameter = st.number_input("Cane Diameter (cm)", value=round(safe_float(district_base.get("Cane_Diameter_cm"), 3.4), 2), step=0.1)
            internode_length = st.number_input("Internode Length (cm)", value=round(safe_float(district_base.get("Internode_Length_cm"), 18.0), 1), step=0.5)
            seed_rate = st.number_input("Seed Rate (sets/acre)", value=round(safe_float(district_base.get("Seed_Rate"), 25000), 1), step=500.0)

        submit_prediction = st.form_submit_button("🚜 Compute Predicted Yield", use_container_width=True)

    if submit_prediction:
        row = district_base.copy()
        row.update({
            "District": district,
            "Season": season,
            "Soil_Type": soil_type,
            "Irrigation_Method_Type": irrigation_method,
            "Fertilizer_Type": fertilizer_type,
            "Month_Num": month_labels.index(month_label) + 1,
            "Rainfall_Total_mm": rainfall,
            "Temp_Min_C": temp_min,
            "Temp_Max_C": temp_max,
            "Temp_Avg_C": temp_avg,
            "Soil_Moisture_%": soil_moisture,
            "Soil_pH": soil_ph,
            "Soil_Depth_cm": soil_depth,
            "Water_Holding_Capacity_%": water_capacity,
            "Heat_Stress_Days": heat_days,
            "Nitrogen_kg_per_acre": nitrogen,
            "Phosphorus_kg_per_acre": phosphorus,
            "Potassium_kg_per_acre": potassium,
            "Water_Quantity_liters_per_acre": water,
            "Irrigation_Frequency_Level": irrigation_frequency,
            "Fertilizer_Quantity": fertilizer_quantity,
            "Cane_Height_cm": cane_height,
            "Cane_Diameter_cm": cane_diameter,
            "Internode_Length_cm": internode_length,
            "Seed_Rate": seed_rate,
        })
        row = refresh_engineered_features(row)
        predicted = predict_row(model, row)
        district_mean = float(raw.loc[raw["District"].astype(str) == district, TARGET].mean())

        st.session_state["last_prediction"] = predicted
        st.session_state["last_prediction_row"] = row.to_dict()
        st.session_state["last_prediction_district"] = district
        st.session_state["last_prediction_baseline"] = district_mean

        st.markdown("### 🎯 Predicted Yield Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Yield", f"{predicted:,.2f} Q/Acre", f"{predicted - district_mean:+.2f} vs District")
        r2.metric("Metric Equivalent", f"{predicted * 0.247105:,.2f} Tonnes / Hectare")
        r3.metric(f"{district} Historical Average", f"{district_mean:,.2f} Q/Acre")

        st.success("✅ Prediction calculated with the champion model pipeline.")

        st.markdown("#### 🔍 Explainable AI: Feature Attribution (TreeSHAP)")
        with st.spinner("Computing local SHAP contributions..."):
            explanation = make_shap_explanation(model, row)
        if explanation is not None:
            shap_df, base_value = explanation
            fig = px.bar(
                shap_df, x="SHAP impact", y="Feature", orientation="h", color="SHAP impact",
                color_continuous_scale=["#dc2626", "#475569", "#22c55e"],
                title="Top Local Feature Contributions (SHAP Directional Impact on Yield)",
            )
            fig.add_vline(x=0, line_dash="dash", line_color="#86efac")
            fig.update_layout(xaxis_title="Contribution to Yield (Quintals/Acre)", yaxis_title="", coloraxis_showscale=False)
            apply_dark_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Baseline Expected Value: {base_value:.2f} Q/Acre. Green bars push the prediction above the base; red bars reduce it.")

    elif "last_prediction" in st.session_state:
        st.info(
            f"Active estimate in session: **{st.session_state['last_prediction']:.2f} Q/Acre** "
            f"for **{st.session_state.get('last_prediction_district', 'the selected district')}**. "
            "You can adjust parameters and submit again anytime."
        )


# -------------------------------------------------------------
# 3. AGRO-RESOURCE OPTIMIZER
# -------------------------------------------------------------
def render_optimizer(raw: pd.DataFrame, data: pd.DataFrame, model) -> None:
    hero("Agro-Resource Optimizer", "Simulate water and fertilizer trade-offs and discover optimal resource configurations.", "PRESCRIPTIVE SCENARIO LAB")

    st.markdown("### 📋 Notebook Benchmark What-If Scenarios")
    if WHAT_IF_PATH.exists():
        df_whatif = pd.read_csv(WHAT_IF_PATH)
        safe_dataframe(df_whatif, use_container_width=True, hide_index=True)
        st.caption("Pre-computed what-if stress tests from Stage 8 of the project notebook.")
    else:
        st.info("What-if scenario CSV will appear here once generated.")

    st.markdown("---")
    st.markdown("### 🔬 Interactive 2D Water × Fertilizer Grid Simulation")
    st.markdown(
        '<div class="callout">This simulator sweeps across water and fertilizer ranges around your farm\'s baseline '
        'to evaluate the non-linear yield response of the trained model.</div>',
        unsafe_allow_html=True,
    )

    districts = sorted(raw["District"].dropna().astype(str).unique().tolist())
    last_district = st.session_state.get("last_prediction_district")
    default_idx = districts.index(last_district) if last_district in districts else 0

    use_last = st.checkbox("Use my latest custom prediction profile as baseline", value="last_prediction_row" in st.session_state)
    if use_last and "last_prediction_row" in st.session_state:
        base_row = pd.Series(st.session_state["last_prediction_row"])
        district = st.session_state.get("last_prediction_district", districts[default_idx])
    else:
        district = st.selectbox("Baseline District Profile", districts, index=default_idx, key="opt_dist")
        base_row = get_reference_row(data, district)

    default_water = max(0.0, safe_float(base_row.get("Water_Quantity_liters_per_acre"), 1500))
    default_fert = max(0.0, safe_float(base_row.get("Fertilizer_Quantity"), 180))

    c1, c2, c3 = st.columns(3)
    with c1:
        base_water = st.number_input("Baseline Water (litres/acre)", min_value=100.0, value=round(default_water, 1), step=50.0)
    with c2:
        base_fert = st.number_input("Baseline Fertilizer (kg/acre)", min_value=10.0, value=round(default_fert, 1), step=10.0)
    with c3:
        grid_res = st.selectbox("Grid Search Resolution", [10, 15, 20], index=1)

    low_mult, high_mult = st.slider(
        "Search Range Multiplier (Relative to Baseline)",
        min_value=0.5, max_value=1.5, value=(0.7, 1.3), step=0.05,
    )

    run_sim = st.button("🚀 Run Grid Search Optimization", type="primary", use_container_width=True)
    if run_sim:
        base_row = refresh_engineered_features(base_row)
        baseline_yield = predict_row(model, base_row)

        w_values = np.linspace(base_water * low_mult, base_water * high_mult, int(grid_res))
        f_values = np.linspace(base_fert * low_mult, base_fert * high_mult, int(grid_res))

        sim_rows = []
        for f_val in f_values:
            for w_val in w_values:
                cand = base_row.copy()
                cand["Water_Quantity_liters_per_acre"] = float(w_val)
                cand["Fertilizer_Quantity"] = float(f_val)
                cand = refresh_engineered_features(cand)
                sim_rows.append(cand)

        prep = model.named_steps["prep"]
        selected_cols = list(prep.transformers_[0][2]) + list(prep.transformers_[1][2])
        cand_frame = pd.DataFrame([
            {c: r.get(c, np.nan) for c in selected_cols} for r in sim_rows
        ], columns=selected_cols)

        predictions = model.predict(cand_frame)
        result_df = pd.DataFrame({
            "Water (L/Acre)": [r["Water_Quantity_liters_per_acre"] for r in sim_rows],
            "Fertilizer (kg/Acre)": [r["Fertilizer_Quantity"] for r in sim_rows],
            "Predicted Yield (Q/Acre)": predictions,
        })

        best_idx = int(np.argmax(predictions))
        best_rec = result_df.iloc[best_idx]
        diff = float(best_rec["Predicted Yield (Q/Acre)"] - baseline_yield)

        st.session_state["optimizer_results"] = result_df
        st.session_state["optimizer_best"] = {
            "water": float(best_rec["Water (L/Acre)"]),
            "fertilizer": float(best_rec["Fertilizer (kg/Acre)"]),
            "yield": float(best_rec["Predicted Yield (Q/Acre)"]),
            "baseline_yield": float(baseline_yield),
            "improvement": diff,
            "district": district,
        }

    if "optimizer_best" in st.session_state:
        b = st.session_state["optimizer_best"]
        res_df = st.session_state["optimizer_results"]

        st.markdown("### 🏆 Optimal Candidate Point Found")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Baseline Yield", f"{b['baseline_yield']:.2f} Q/Acre")
        q2.metric("Optimal Simulated Yield", f"{b['yield']:.2f} Q/Acre", f"{b['improvement']:+.2f} Q/Acre")
        q3.metric("Recommended Water", f"{b['water']:,.1f} L/Acre")
        q4.metric("Recommended Fertilizer", f"{b['fertilizer']:.1f} kg/Acre")

        heat_data = res_df.pivot(index="Fertilizer (kg/Acre)", columns="Water (L/Acre)", values="Predicted Yield (Q/Acre)")
        heat_fig = go.Figure(data=go.Heatmap(
            z=heat_data.values,
            x=[round(float(v), 1) for v in heat_data.columns],
            y=[round(float(v), 1) for v in heat_data.index],
            colorscale=[[0, "#0b1c11"], [0.5, "#15803d"], [1, "#4ade80"]],
            colorbar={"title": "Yield<br>(Q/Acre)"},
        ))
        heat_fig.update_layout(
            title="2D Yield Response Surface (Water vs. Fertilizer Quantity)",
            xaxis_title="Water Quantity (liters/acre)",
            yaxis_title="Fertilizer Quantity (kg/acre)",
        )
        apply_dark_theme(heat_fig)
        st.plotly_chart(heat_fig, use_container_width=True)

        st.download_button(
            "📥 Download Grid Results (CSV)",
            data=res_df.to_csv(index=False).encode("utf-8"),
            file_name="sugarcane_resource_optimization_grid.csv",
            mime="text/csv",
        )


# -------------------------------------------------------------
# 4. EXPLAINABLE AI (XAI) PAGE
# -------------------------------------------------------------
def render_xai() -> None:
    hero("Explainable AI (XAI)", "Unbox model predictions using Shapley Additive Explanations (SHAP).", "TRANSPARENCY & INTERPRETABILITY")

    st.markdown(
        "<p style='color: #c9d8ce; font-size: 1.02rem; line-height: 1.6;'>"
        "Modern agricultural decisions require trust and transparency. To ensure the ML predictions are not a black box, "
        "we use <b style='color: #4ade80;'>TreeSHAP</b> to quantify the exact contribution of each agronomic factor to the predicted sugarcane yield."
        "</p>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### 🌐 Global SHAP Feature Importance")
        if SHAP_PLOT_PATH.exists():
            st.image(str(SHAP_PLOT_PATH), caption="Global SHAP Summary: Directional impact on yield across holdout samples", use_container_width=True)
        else:
            st.info("Global SHAP summary plot will display here once exported.")

    with c2:
        st.markdown("### 🌾 Key Agronomic Insights")
        insights = [
            ("Cane Growth Index", "Primary physical driver: cane height × diameter strongly correlates with biomass."),
            ("Soil Moisture & Water Retention", "Consistent moisture is critical; severe water deficit drops yield rapidly."),
            ("Fertilizer-Irrigation Interaction", "Nutrients require adequate water solvent to be absorbed efficiently."),
            ("Climate Stress (Heat & Frost)", "Excess heat stress days during tillering cause significant yield reduction."),
            ("NPK Balance", "Excess nitrogen without balanced phosphorus/potassium leads to vegetative growth without sugar recovery."),
        ]
        for name, desc in insights:
            st.markdown(
                f'<div class="metric-card" style="margin-bottom:0.65rem">'
                f'<div style="font-weight:700;color:#4ade80;font-size:0.92rem">{name}</div>'
                f'<div style="font-size:0.83rem;color:#a3b8ab;margin-top:0.25rem">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### 📋 Selected Model Features (61 Numeric Predictors)")
    if SELECTED_FEATURES_PATH.exists():
        df_feats = pd.read_csv(SELECTED_FEATURES_PATH)
        safe_dataframe(df_feats, use_container_width=True, hide_index=True)
        st.caption("Selected numeric features identified through permutation importance in the notebook workflow.")


# -------------------------------------------------------------
# 5. MODEL PERFORMANCE PAGE
# -------------------------------------------------------------
def render_model_performance(metrics: dict[str, Any]) -> None:
    hero("Model Performance & Benchmarks", "Compare cross-validation and holdout evaluation across four machine learning algorithms.", "EVALUATION & BENCHMARKS")

    st.markdown("### 📊 5-Fold Cross-Validation on Training Split")
    if CV_RESULTS_PATH.exists():
        df_cv = pd.read_csv(CV_RESULTS_PATH)
        safe_dataframe(df_cv.round(3), use_container_width=True, hide_index=True)
    else:
        st.info("Cross-validation CSV not found.")

    st.markdown("### 🧪 Holdout 20% Test Set Benchmark")
    if MODEL_COMP_PATH.exists():
        df_comp = pd.read_csv(MODEL_COMP_PATH)
        safe_dataframe(df_comp.round(3), use_container_width=True, hide_index=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_r2 = px.bar(
                df_comp.sort_values("Test_R2"),
                x="Test_R2", y="Model", orientation="h", color="Test_R2",
                color_continuous_scale=["#143a26", "#22c55e", "#4ade80"],
                title="Holdout Test Set: R² Score Comparison",
            )
            fig_r2.update_layout(xaxis_title="Test R² Score", yaxis_title="", coloraxis_showscale=False)
            apply_dark_theme(fig_r2)
            st.plotly_chart(fig_r2, use_container_width=True)

        with col_c2:
            fig_mae = px.bar(
                df_comp.sort_values("Test_MAE", ascending=False),
                x="Test_MAE", y="Model", orientation="h", color="Test_MAE",
                color_continuous_scale=["#22c55e", "#eab308", "#dc2626"],
                title="Holdout Test Set: Error Comparison (Lower MAE is Better)",
            )
            fig_mae.update_layout(xaxis_title="Mean Absolute Error (Q/Acre)", yaxis_title="", coloraxis_showscale=False)
            apply_dark_theme(fig_mae)
            st.plotly_chart(fig_mae, use_container_width=True)
    else:
        st.info("Model comparison results will display here.")

    st.markdown("### ⚙️ Packaged Champion Model Details")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Model Algorithm", metrics.get("model_name", "Tuned XGBoost"))
    p2.metric("Holdout R²", f"{metrics.get('test_r2', 0.2902):.4f}")
    p3.metric("Holdout MAE", f"{metrics.get('test_mae', 74.30):.2f} Q/Acre")
    p4.metric("Holdout RMSE", f"{metrics.get('test_rmse', 91.81):.2f} Q/Acre")

    st.markdown(
        '<div class="warning-box"><b>Agronomic Model Context:</b> The dataset exhibits natural agricultural field variance '
        'across multiple micro-climatic zones in Maharashtra. An R² around 0.29 with an MAE of ~74 Q/Acre provides a strong baseline '
        'for directional resource exploration and decision support, rather than exact deterministic yield prediction.</div>',
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------
# 6. DATASET EXPLORER PAGE
# -------------------------------------------------------------
def render_dataset_explorer(raw: pd.DataFrame) -> None:
    hero("Dataset Explorer", "Inspect field records, filter by district/season, and explore agricultural distributions.", "DATA & DISCOVERY")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(raw):,}")
    c2.metric("Input Variables", str(raw.shape[1]))
    c3.metric("Missing Cells", f"{int(raw.isna().sum().sum()):,}")
    c4.metric("Districts", str(raw["District"].nunique()) if "District" in raw else "—")

    left, right = st.columns(2)
    with left:
        dists = ["All Districts"] + sorted(raw["District"].dropna().astype(str).unique().tolist())
        sel_district = st.selectbox("Filter by District", dists)
    with right:
        seasons = ["All Seasons"] + (sorted(raw["Season"].dropna().astype(str).unique().tolist()) if "Season" in raw else [])
        sel_season = st.selectbox("Filter by Season", seasons)

    view = raw.copy()
    if sel_district != "All Districts":
        view = view[view["District"].astype(str) == sel_district]
    if sel_season != "All Seasons" and "Season" in view:
        view = view[view["Season"].astype(str) == sel_season]

    st.markdown(f"#### Matching Field Records: {len(view):,}")
    safe_dataframe(view.head(200), use_container_width=True, hide_index=True)
    st.caption("Showing the first 200 records. Download full dataset below.")

    st.download_button(
        "📥 Download Full Dataset (CSV)",
        data=raw.to_csv(index=False).encode("utf-8"),
        file_name="FINAL_SUGARCANE_DATASET.csv",
        mime="text/csv",
    )

    st.markdown("#### District Yield Summary Table")
    dist_stats = raw.groupby("District")[TARGET].agg(["count", "mean", "median", "std"]).reset_index()
    dist_stats.columns = ["District", "Records", "Mean Yield (Q/Acre)", "Median Yield (Q/Acre)", "Std Dev"]
    dist_stats = dist_stats.sort_values("Mean Yield (Q/Acre)", ascending=False)
    safe_dataframe(dist_stats.round(2), use_container_width=True, hide_index=True)

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        if "Soil_Type" in raw.columns:
            st.markdown("#### Yield by Soil Type")
            soil_stats = raw.groupby("Soil_Type", as_index=False)[TARGET].mean().sort_values(TARGET, ascending=False)
            fig_soil = px.bar(
                soil_stats, x="Soil_Type", y=TARGET, color=TARGET,
                color_continuous_scale=["#143a26", "#22c55e", "#4ade80"],
                title="Mean Yield by Soil Classification",
            )
            fig_soil.update_layout(xaxis_title="Soil Type", yaxis_title="Mean Yield (Q/Acre)", coloraxis_showscale=False)
            apply_dark_theme(fig_soil)
            st.plotly_chart(fig_soil, use_container_width=True)

    with col_chart2:
        if "Irrigation_Method_Type" in raw.columns:
            st.markdown("#### Yield by Irrigation Method")
            irrig_stats = raw.groupby("Irrigation_Method_Type", as_index=False)[TARGET].mean().sort_values(TARGET, ascending=False)
            fig_irrig = px.bar(
                irrig_stats, x="Irrigation_Method_Type", y=TARGET, color=TARGET,
                color_continuous_scale=["#143a26", "#22c55e", "#4ade80"],
                title="Mean Yield by Irrigation Method",
            )
            fig_irrig.update_layout(xaxis_title="Irrigation Method", yaxis_title="Mean Yield (Q/Acre)", coloraxis_showscale=False)
            apply_dark_theme(fig_irrig)
            st.plotly_chart(fig_irrig, use_container_width=True)


# -------------------------------------------------------------
# MAIN APP ORCHESTRATION
# -------------------------------------------------------------
def main() -> None:
    raw = None
    data = None
    model = None
    metrics = {}
    try:
        raw, data = load_data()
        model = load_model()
        metrics = load_metrics()
    except Exception as exc:
        st.error(f"Error initializing app: {exc}")
        st.info("Ensure `FINAL_SUGARCANE_DATASET.csv` and `sugarcane_yield_best_model.pkl` are in the project folder.")
        st.stop()

    with st.sidebar:
        st.markdown("# 🌱 CaneVision")
        st.caption("Maharashtra Sugarcane Intelligence")
        st.divider()
        page = st.radio(
            "NAVIGATION",
            [
                "Overview",
                "Yield Prediction",
                "Agro-Resource Optimizer",
                "Explainable AI (SHAP)",
                "Model Performance",
                "Dataset Explorer",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Project**")
        st.caption("Maharashtra Sugarcane Yield Prediction and Agro-Resource Optimization Using Explainable Machine Learning")
        st.caption("SIT Nagpur · PBL Final Project")
        st.divider()
        st.caption("v2.0 · Streamlit & Scikit-Learn / XGBoost")

    if page == "Overview":
        render_overview(raw, data, metrics)
    elif page == "Yield Prediction":
        render_prediction(raw, data, model)
    elif page == "Agro-Resource Optimizer":
        render_optimizer(raw, data, model)
    elif page == "Explainable AI (SHAP)":
        render_xai()
    elif page == "Model Performance":
        render_model_performance(metrics)
    elif page == "Dataset Explorer":
        render_dataset_explorer(raw)

    st.markdown("---")
    st.caption("CaneVision · Academic PBL Prototype · Maharashtra Sugarcane Yield Prediction & Agro-Resource Optimization")


if __name__ == "__main__":
    main()



# .\.venv\Scripts\python.exe -m streamlit run app.py

# .\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502