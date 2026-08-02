# -*- coding: utf-8 -*-
"""
🌊 Monitor de Enchentes — Rio Guaíba
Streamlit web application for flood prediction monitoring.

Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
from pathlib import Path
import json
import pickle as pkl
import time

# ── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌊 Monitor Guaíba",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT / "data" / "processed" / "dataset_historico.parquet"
UPDATE_SCRIPT = PROJECT / "update_dataset.py"

# ── Feature definitions ──────────────────────────────────────────────────────
FEATURES_3D = [
    "taquari_mucum_chuva", "encantado_wind_dir_deg", "taquari_mucum_delta3",
    "sinos_sl_nivel_mean", "jacui_rp_chuva_roll7", "mostardas_v_wind_roll2",
    "campo_bom_precip_mm", "cachoeira_do_sul_wind_dir_deg", "catsul_nivel_mean_lag5",
    "chuva_total_acc_30d", "n_bacias_muito_chuvosas", "guaiba_chuva_roll3",
    "feliz_wind_max_kmh",
]
FEATURES_5D = [
    "taquari_mucum_chuva", "mostardas_v_wind_roll3", "sinos_sl_nivel_mean_lag5",
    "encantado_wind_dir_deg", "cai_pm_chuva_roll3", "jacui_rp_nivel_mean_lag7",
    "sinos_sl_chuva_roll3", "dias_chuvosos_7d", "mostardas_v_wind",
    "dias_chuvosos_14d", "cachoeira_do_sul_wind_dir_deg", "catsul_chuva",
    "estrela_wind_max_kmh",
]
ALL_MODEL_FEATURES = sorted(set(FEATURES_3D + FEATURES_5D))
EXTRA_KEYS = {"guaiba_nivel_mean", "chuva_total_raw", "guaiba_delta1", "guaiba_delta3"}
OFF_MODEL_1D = {
    "gravatai_sl_chuva", "mucum_wind_dir_deg", "campo_bom_wind_dir_deg",
    "gravatai_sl_nivel_max", "encantado_precip_mm", "cachoeira_do_sul_wind_max_kmh",
    "cachoeira_do_sul_precip_mm", "catsul_nivel_max", "taquari_mucum_chuva_roll3",
    "rio_grande_v_wind",
}
# SFS LogReg+OptBin features (binary classification)
SFS_LOGREG = {
    "sinos_sl_nivel_mean_lag5", "sinos_cb_chuva_roll3", "sinos_sl_nivel_mean_lag3",
    "represamento_3d", "rio_grande_v_wind_roll2", "jacui_rp_chuva_roll3",
    "sinos_sl_nivel_mean_lag1",
}
# Features exclusive to SFS LogReg (not in CatBoost/LightGBM models)
OFF_MODEL_BIN = SFS_LOGREG - ALL_MODEL_FEATURES

FEATURE_META = {
    # ── Chuva ──
    "taquari_mucum_chuva":       {"desc": "Chuva em Muçum (mm)", "interp": "Chuva que vai propagar pelo rio Taquari até o Guaíba em ~3 dias", "group": "Chuva", "type": "rain"},
    "jacui_rp_chuva_roll7":      {"desc": "Chuva acumulada em 7 dias no Jacuí — Rio Pardo (mm)", "interp": "Chuva persistente no Jacuí — maior bacia que deságua no Guaíba", "group": "Chuva", "type": "rain"},
    "campo_bom_precip_mm":       {"desc": "Chuva em Campo Bom (mm)", "interp": "Chuva na região metropolitana — contribuição rápida ao Guaíba", "group": "Chuva", "type": "rain"},
    "chuva_total_acc_30d":       {"desc": "Chuva total acumulada em 30 dias (mm)", "interp": "Quanto choveu no último mês — indica saturação geral do sistema", "group": "Chuva", "type": "rain"},
    "n_bacias_muito_chuvosas":   {"desc": "Nº de bacias com chuva intensa hoje", "interp": "0-7: quantas bacias tiveram >20mm. Mais bacias = risco maior", "group": "Chuva", "type": "rain"},
    "guaiba_chuva_roll3":        {"desc": "Chuva acumulada em 3 dias na bacia do Guaíba (mm)", "interp": "Chuva direta na região de Porto Alegre", "group": "Chuva", "type": "rain"},
    "cai_pm_chuva_roll3":        {"desc": "Chuva acumulada em 3 dias no Caí (mm)", "interp": "Chuva no Caí — contribuição direta ao Guaíba", "group": "Chuva", "type": "rain"},
    "sinos_sl_chuva_roll3":      {"desc": "Chuva acumulada em 3 dias nos Sinos (mm)", "interp": "Chuva persistente na bacia dos Sinos", "group": "Chuva", "type": "rain"},
    "catsul_chuva":              {"desc": "Chuva no Terminal CATSUL (mm)", "interp": "Chuva direta na região do Guaíba a jusante", "group": "Chuva", "type": "rain"},
    "sinos_cb_chuva_roll3":      {"desc": "Chuva acumulada em 3 dias em Campo Bom/Sinos (mm)", "interp": "Chuva na sub-bacia dos Sinos — Campo Bom", "group": "Chuva", "type": "rain"},
    "jacui_rp_chuva_roll3":      {"desc": "Chuva acumulada em 3 dias no Jacuí — Rio Pardo (mm)", "interp": "Chuva recente no Jacuí — contribuição direta ao Guaíba", "group": "Chuva", "type": "rain"},
    "sinos_cb_chuva_roll3":      {"desc": "Chuva acumulada em 3 dias em Campo Bom/Sinos (mm)", "interp": "Chuva na sub-bacia dos Sinos — Campo Bom", "group": "Chuva", "type": "rain"},
    # ── Vento (direção) ──
    "encantado_wind_dir_deg":          {"desc": "Direção do vento em Encantado (°)", "interp": "Rosa dos ventos: N/NW=sobe, S/SE=desce", "group": "Vento (direção)", "type": "wind_dir"},
    "cachoeira_do_sul_wind_dir_deg":   {"desc": "Direção do vento em Cachoeira do Sul (°)", "interp": "Rosa dos ventos: N/NW=sobe, S/SE=desce", "group": "Vento (direção)", "type": "wind_dir"},
    # ── Vento (velocidade) ──
    "feliz_wind_max_kmh":        {"desc": "Rajada máxima de vento em Feliz (km/h)", "interp": "Ventos fortes podem causar represamento no Guaíba", "group": "Vento (velocidade)", "type": "wind_speed"},
    "mostardas_v_wind":          {"desc": "Vento sul em Mostardas — hoje (m/s)", "interp": "Positivo=vento sul → represamento. Negativo=vento norte → drenagem", "group": "Vento (velocidade)", "type": "wind_speed"},
    "mostardas_v_wind_roll2":    {"desc": "Vento sul em Mostardas — média 2 dias (m/s)", "interp": "Positivo=vento sul → represamento (empurra Lagoa dos Patos para o Guaíba). Negativo=vento norte → drenagem", "group": "Vento (velocidade)", "type": "wind_speed"},
    "mostardas_v_wind_roll3":    {"desc": "Vento sul em Mostardas — média 3 dias (m/s)", "interp": "Positivo=vento sul persistente → represamento acumulado. Negativo=vento norte → drenagem", "group": "Vento (velocidade)", "type": "wind_speed"},
    "estrela_wind_max_kmh":      {"desc": "Rajada máxima de vento em Estrela (km/h)", "interp": "Ventos fortes no vale do Taquari", "group": "Vento (velocidade)", "type": "wind_speed"},
    "rio_grande_v_wind_roll2":   {"desc": "Vento sul em Rio Grande — média 2 dias (m/s)", "interp": "Positivo=vento sul → represamento na Lagoa dos Patos. Média de 2 dias", "group": "Vento (velocidade)", "type": "wind_speed"},
    "represamento_3d":           {"desc": "Índice de represamento — 3 dias", "interp": "Acumulado de efeito de vento sul sobre o Guaíba em 3 dias", "group": "Vento (velocidade)", "type": "other"},
    # ── Nível ──
    "taquari_mucum_delta3":      {"desc": "Variação do nível em Muçum em 3 dias (m)", "interp": "Se positivo, o rio Taquari está subindo — água a caminho do Guaíba", "group": "Nível", "type": "level"},
    "sinos_sl_nivel_mean":       {"desc": "Nível do rio Sinos em São Leopoldo (m)", "interp": "Nível alto indica contribuição da bacia dos Sinos", "group": "Nível", "type": "level"},
    "catsul_nivel_mean_lag5":    {"desc": "Nível no Terminal CATSUL — 5 dias atrás (m)", "interp": "Nível passado no Guaíba a jusante — indica tendência de longo prazo", "group": "Nível", "type": "level"},
    "sinos_sl_nivel_mean_lag5":  {"desc": "Nível do Sinos em São Leopoldo — 5 dias atrás (m)", "interp": "Nível passado indica propagação lenta da cheia", "group": "Nível", "type": "level"},
    "sinos_sl_nivel_mean_lag3":  {"desc": "Nível do Sinos em São Leopoldo — 3 dias atrás (m)", "interp": "Nível com lag intermediário — captura propagação da cheia", "group": "Nível", "type": "level"},
    "sinos_sl_nivel_mean_lag1":  {"desc": "Nível do Sinos em São Leopoldo — 1 dia atrás (m)", "interp": "Nível recente — reação imediata da bacia dos Sinos", "group": "Nível", "type": "level"},
    "jacui_rp_nivel_mean_lag7":  {"desc": "Nível do Jacuí em Rio Pardo — 7 dias atrás (m)", "interp": "Nível com longo lag — captura a propagação lenta do Jacuí", "group": "Nível", "type": "level"},
    "guaiba_nivel_mean":         {"desc": "Nível atual do Guaíba — T0 (m)", "interp": "O nível medido hoje na régua de Porto Alegre", "group": "Nível", "type": "level"},
    "guaiba_delta1":             {"desc": "Variação do nível do Guaíba em 1 dia (m)", "interp": "Se positivo, o nível subiu desde ontem", "group": "Nível", "type": "level"},
    "guaiba_delta3":             {"desc": "Variação do nível do Guaíba em 3 dias (m)", "interp": "Tendência de 3 dias — mostra se está subindo ou descendo", "group": "Nível", "type": "level"},
    # ── Saturação do solo ──
    "dias_chuvosos_7d":          {"desc": "Dias com chuva nos últimos 7 dias", "interp": "0-7: quantos dias choveu. Solo saturado escoa mais rápido", "group": "Saturação do solo", "type": "other"},
    "dias_chuvosos_14d":         {"desc": "Dias com chuva nos últimos 14 dias", "interp": "0-14: frequência de chuva — proxy de saturação do solo", "group": "Saturação do solo", "type": "other"},
    # ── Contexto (extra — not in models) ──
    "chuva_total_raw":           {"desc": "Chuva total hoje em todas as bacias (mm)", "interp": "Soma da chuva de hoje em todas as 8 estações", "group": "Contexto", "type": "other"},
    # ── Off-model (delta_1d SFS features) ──
    "gravatai_sl_chuva":           {"desc": "Chuva em Gravataí/São Leopoldo (mm)", "interp": "Chuva na bacia dos Sinos — contribuição direta ao Guaíba", "group": "Chuva", "type": "off_model"},
    "encantado_precip_mm":         {"desc": "Precipitação em Encantado (mm)", "interp": "Precipitação no vale do Taquari — propagação em ~3 dias", "group": "Chuva", "type": "off_model"},
    "cachoeira_do_sul_precip_mm":  {"desc": "Precipitação em Cachoeira do Sul (mm)", "interp": "Precipitação na região intermediária do Jacuí", "group": "Chuva", "type": "off_model"},
    "taquari_mucum_chuva_roll3":   {"desc": "Chuva acumulada em 3 dias em Muçum (mm)", "interp": "Chuva persistente no Taquari — proxy de propagação", "group": "Chuva", "type": "off_model"},
    "mucum_wind_dir_deg":          {"desc": "Direção do vento em Muçum (°)", "interp": "Rosa dos ventos: N/NW=sobe, S/SE=desce", "group": "Vento (direção)", "type": "off_model"},
    "campo_bom_wind_dir_deg":      {"desc": "Direção do vento em Campo Bom (°)", "interp": "Rosa dos ventos: N/NW=sobe, S/SE=desce", "group": "Vento (direção)", "type": "off_model"},
    "cachoeira_do_sul_wind_max_kmh": {"desc": "Rajada máxima em Cachoeira do Sul (km/h)", "interp": "Ventos fortes na região intermediária do Jacuí", "group": "Vento (velocidade)", "type": "off_model"},
    "rio_grande_v_wind":           {"desc": "Vento sul em Rio Grande (m/s)", "interp": "Positivo=vento sul → represamento na Lagoa dos Patos", "group": "Vento (velocidade)", "type": "off_model"},
    "gravatai_sl_nivel_max":       {"desc": "Nível máximo em Gravataí/São Leopoldo (m)", "interp": "Pico de nível na bacia dos Sinos — contribuição ao Guaíba", "group": "Nível", "type": "off_model"},
    "catsul_nivel_max":            {"desc": "Nível máximo no Terminal CATSUL (m)", "interp": "Pico de nível no Guaíba a jusante", "group": "Nível", "type": "off_model"},
}


# ── Helpers ──────────────────────────────────────────────────────────────────
MONTH_ABBR = {1:'jan',2:'fev',3:'mar',4:'abr',5:'mai',6:'jun',7:'jul',8:'ago',9:'set',10:'out',11:'nov',12:'dez'}

def fmt_date(d):
    d = pd.Timestamp(d)
    return f"{d.day:02d}{MONTH_ABBR[d.month]}{str(d.year)[2:]}"

def deg_to_compass(d):
    if pd.isna(d): return "N/A"
    d = d % 360
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    idx = int((d + 22.5) / 45) % 8
    return dirs[idx]

def deg_to_compass_pt(d):
    """Return Portuguese compass name for degree value."""
    if pd.isna(d): return "N/A"
    mapping = {"N": "Norte", "NE": "Nordeste", "E": "Leste", "SE": "Sudeste",
               "S": "Sul", "SW": "Sudoeste", "W": "Oeste", "NW": "Noroeste"}
    return mapping.get(deg_to_compass(d), "N/A")

def deg_to_arrow(d):
    """Return arrow emoji for degree value."""
    if pd.isna(d): return "❓"
    mapping = {"N": "⬆️", "NE": "↗️", "E": "➡️", "SE": "↘️",
               "S": "⬇️", "SW": "↙️", "W": "⬅️", "NW": "↖️"}
    return mapping.get(deg_to_compass(d), "❓")

def alert_level(nivel):
    if nivel < 1.0: return "Normal"
    elif nivel < 2.0: return "Atenção"
    elif nivel < 3.0: return "Alerta"
    else: return "INUNDAÇÃO"

def alert_color(level):
    return {"Normal":"#4CAF50","Atenção":"#F9A825","Alerta":"#FF9800","INUNDAÇÃO":"#F44336"}[level]

def alert_emoji(level):
    return {"Normal":"🟢","Atenção":"🟡","Alerta":"🟠","INUNDAÇÃO":"🔴"}[level]

def percentile_class(pct):
    if pct >= 99: return "critical"
    elif pct >= 95: return "extreme"
    elif pct >= 90: return "very_high"
    elif pct >= 75: return "elevated"
    else: return "normal"

def pct_badge_html(cls, pct):
    colors = {"normal":"#4CAF50","elevated":"#F9A825","very_high":"#FF9800","extreme":"#F44336","critical":"#B71C1C"}
    labels = {"normal":"Normal","elevated":"Elevado","very_high":"Muito Alto","extreme":"Extremo","critical":"CRÍTICO"}
    c = colors.get(cls, "#4CAF50")
    l = labels.get(cls, "Normal")
    anim = "animation:pulse 1.5s infinite;" if cls == "critical" else ""
    return f'<span style="background:{c}22;color:{c};padding:3px 8px;border-radius:6px;font-size:0.8em;font-weight:600;{anim}">{pct:.1f}% — {l}</span>'

def rain_color_mm(mm):
    if mm < 10: return "#4CAF50"
    elif mm < 30: return "#F9A825"
    elif mm < 60: return "#FF9800"
    else: return "#F44336"

def risk_level(prob):
    """Return risk classification based on extreme event probability."""
    if prob < 0.01:
        return "Normal", "#4CAF50", "🟢"
    elif prob < 0.05:
        return "Atenção", "#F9A825", "🟡"
    elif prob < 0.20:
        return "Alerta Precoce", "#FF9800", "🟠"
    else:
        return "Risco Crítico", "#F44336", "🔴"


# ── Data loading (cached) ────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Carregando dados históricos...")
def load_historico():
    """Load the unified historical dataset."""
    if not DATASET_PATH.exists():
        return None
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=3600, show_spinner="Carregando dados históricos para estatísticas...")
def load_dev_data():
    """Load historical dataset for statistics (percentiles)."""
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    if "guaiba_delta1" not in df.columns:
        df["guaiba_delta1"] = df["guaiba_nivel_mean"].diff(1)
    if "guaiba_delta3" not in df.columns:
        df["guaiba_delta3"] = df["guaiba_nivel_mean"].diff(3)
    return df


def run_update():
    """Run update_dataset.py to incrementally update the historical dataset."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, str(UPDATE_SCRIPT)],
        capture_output=True, text=True, cwd=str(PROJECT),
    )
    return result.returncode == 0, result.stdout + result.stderr


# ── Main App ─────────────────────────────────────────────────────────────────
def main():
    # Custom CSS
    st.markdown("""
    <style>
    @keyframes pulse {
        0%,100% { box-shadow: 0 0 5px rgba(183,28,28,0.3); }
        50% { box-shadow: 0 0 15px rgba(183,28,28,0.6); }
    }
    @keyframes risk_glow {
        0%,100% { box-shadow: 0 0 5px rgba(244,67,54,0.2); }
        50% { box-shadow: 0 0 15px rgba(244,67,54,0.5); }
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px; padding: 24px; text-align: center;
        border: 2px solid; margin-bottom: 8px;
    }
    .metric-value { font-size: 2.8em; font-weight: 800; line-height: 1.1; }
    .metric-label { font-size: 0.85em; color: #8899aa; text-transform: uppercase; letter-spacing: 1px; }
    .metric-class { font-size: 1.2em; font-weight: 600; margin-top: 8px; }
    .metric-delta { font-size: 0.9em; color: #8899aa; margin-top: 4px; }
    .metric-trend { font-size: 1.4em; margin-top: 4px; }
    div[data-testid="stMetric"] { background: #1a1a2e; border-radius: 12px; padding: 12px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    st.sidebar.title("🌊 Controles")

    # Load data
    historico = load_historico()
    wind_impact = json.loads((PROJECT / "data" / "processed" / "wind_direction_impact.json").read_text())

    # Load models for feature importance
    models_dir = PROJECT / "models"
    importance_map = {}
    try:
        with open(models_dir / "model_metadata.pkl", "rb") as f:
            meta = pkl.load(f)
        features_3d = meta.get("features_3d", FEATURES_3D)
        features_5d = meta.get("features_5d", FEATURES_5D)

        with open(models_dir / "model_delta_3d.pkl", "rb") as f:
            model_3d = pkl.load(f)
        with open(models_dir / "model_delta_5d.pkl", "rb") as f:
            model_5d = pkl.load(f)

        imp_3d = dict(zip(features_3d, model_3d.feature_importances_))
        imp_5d = dict(zip(features_5d, model_5d.get_feature_importance()))

        # Combine: use max importance across models
        all_imp_feats = set(imp_3d.keys()) | set(imp_5d.keys())
        raw_imp = {}
        for feat in all_imp_feats:
            raw_imp[feat] = max(imp_3d.get(feat, 0), imp_5d.get(feat, 0))

        # Normalize to 0-100
        max_imp = max(raw_imp.values()) if raw_imp else 1
        for feat, val in raw_imp.items():
            importance_map[feat] = (val / max_imp) * 100 if max_imp > 0 else 0
    except Exception:
        importance_map = {}
    if historico is None:
        st.sidebar.warning("⚠️ Dataset não encontrado!")
        if st.sidebar.button("🔄 Gerar dataset histórico", type="primary"):
            with st.spinner("Gerando dados históricos... (pode levar vários minutos na primeira vez)"):
                success, output = run_update()
            if success:
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"Erro ao gerar dataset:\n{output}")
        st.stop()

    chart_df = historico.copy()
    last_date = chart_df["date"].max().date()
    today = datetime.now().date()
    days_behind = (today - last_date).days

    st.sidebar.caption(f"📂 Dados: {chart_df['date'].min().date()} → {last_date} ({len(chart_df)} dias)")
    if days_behind > 0:
        st.sidebar.warning(f"⚠️ Dados {days_behind} dia{'s' if days_behind > 1 else ''} atrasado{'s' if days_behind > 1 else ''}")
    else:
        st.sidebar.success("✅ Dados atualizados")

    # ── Binary model: prob_extremo already in dataset ──
    if "prob_extremo" not in chart_df.columns:
        chart_df["prob_extremo"] = 0.0
    binary_model_ok = "prob_extremo" in chart_df.columns and chart_df["prob_extremo"].max() > 0

    # Horizon selector (removed — T+5 only)

    # Date range selector
    st.sidebar.markdown("---")
    min_date = chart_df["date"].min().date()
    max_date = chart_df["date"].max().date()

    # Initialize session state
    if "start_date" not in st.session_state:
        st.session_state.start_date = max(min_date, max_date - timedelta(days=90))
    if "end_date" not in st.session_state:
        st.session_state.end_date = max_date
    if "slider_start" not in st.session_state:
        st.session_state.slider_start = st.session_state.start_date
    if "slider_end" not in st.session_state:
        st.session_state.slider_end = st.session_state.end_date

    # Callbacks for sync
    def on_slider_change():
        st.session_state.start_date = st.session_state.slider_range[0]
        st.session_state.end_date = st.session_state.slider_range[1]

    def on_input_change():
        if "d1" in st.session_state and "d2" in st.session_state:
            st.session_state.start_date = st.session_state.d1
            st.session_state.end_date = st.session_state.d2

    # Date range slider
    st.sidebar.slider(
        "📅 Período (arraste)",
        min_value=min_date,
        max_value=max_date,
        value=(st.session_state.start_date, st.session_state.end_date),
        format="DD/MM/YYYY",
        key="slider_range",
        on_change=on_slider_change,
    )

    # Date input boxes
    col_d1, col_d2 = st.sidebar.columns(2)
    col_d1.date_input("De", value=st.session_state.start_date, min_value=min_date, max_value=max_date, key="d1", format="DD/MM/YYYY", on_change=on_input_change)
    col_d2.date_input("Até", value=st.session_state.end_date, min_value=min_date, max_value=max_date, key="d2", format="DD/MM/YYYY", on_change=on_input_change)

    start_date = st.session_state.start_date
    end_date = st.session_state.end_date

    # Filter chart data
    mask = (chart_df["date"] >= pd.Timestamp(start_date)) & (chart_df["date"] <= pd.Timestamp(end_date))
    view_df = chart_df[mask].copy()

    # Predictions are pre-computed in the dataset by update_dataset.py
    # If missing (shouldn't happen), compute inline (T+5 only)
    if "proj_T5" not in chart_df.columns:
        models_dir = PROJECT / "models"
        with open(models_dir / "model_metadata.pkl", "rb") as f:
            meta = pkl.load(f)
        with open(models_dir / "model_delta_5d.pkl", "rb") as f:
            m5d = pkl.load(f)
        chart_df["pred_delta_5d"] = m5d.predict(chart_df[meta["features_5d"]].values)
        chart_df["proj_T5"] = chart_df["guaiba_nivel_mean"] + chart_df["pred_delta_5d"]

    # Filter view
    mask = (chart_df["date"] >= pd.Timestamp(start_date)) & (chart_df["date"] <= pd.Timestamp(end_date))
    view_df = chart_df[mask].copy()

    # ── Current state (last day of selected range) ──
    if len(view_df) == 0:
        st.warning("Nenhum dado no período selecionado.")
        st.stop()
    last = view_df.iloc[-1]
    last_proj_T5 = float(last["proj_T5"]) if "proj_T5" in last.index and not pd.isna(last.get("proj_T5")) else 0

    current_nivel = float(last["guaiba_nivel_mean"])
    current_alert = alert_level(current_nivel)
    current_prob = float(last.get("prob_extremo", 0))
    # Compute delta_1d from view data
    if len(view_df) >= 2:
        delta_1d = float(view_df.iloc[-1]["guaiba_nivel_mean"] - view_df.iloc[-2]["guaiba_nivel_mean"])
    else:
        delta_1d = 0.0

    # ── Header ──
    st.markdown(f"# 🌊 Monitor de Enchentes — Rio Guaíba")
    st.caption(f"📅 Dados de: {pd.Timestamp(last['date']).strftime('%d/%m/%Y')} · {len(chart_df)} dias de histórico")

    # ── Cards de status ──
    # Compute target dates
    last_date_ts = pd.Timestamp(last["date"])
    date_t0_str = last_date_ts.strftime("%d/%m/%Y")
    date_t5_str = (last_date_ts + timedelta(days=5)).strftime("%d/%m/%Y")

    # Trend
    if delta_1d > 0.05:
        trend_text, trend_color = "↑ subindo", "#F44336"
    elif delta_1d < -0.05:
        trend_text, trend_color = "↓ descendo", "#4CAF50"
    else:
        trend_text, trend_color = "→ estável", "#8899aa"

    # Projection trends
    def proj_trend(proj_val, current_val):
        d = proj_val - current_val
        if d > 0.05: return "↑ subindo", "#F44336"
        elif d < -0.05: return "↓ descendo", "#4CAF50"
        else: return "→ estável", "#8899aa"

    t5_text, t5_color = proj_trend(last_proj_T5, current_nivel)

    # ── Risk level classification ──
    risk_cls, risk_col, risk_emo = risk_level(current_prob)

    col_sema, col_risk, col_t5 = st.columns(3)

    with col_sema:
        ac = alert_color(current_alert)
        st.markdown(f"""
        <div style="background:#1a1a2e;border:2px solid {ac};border-radius:12px;
             padding:16px;text-align:center;">
            <div style="font-size:0.8em;color:#8899aa;">NÍVEL ATUAL · {date_t0_str}</div>
            <div style="font-size:1.8em;font-weight:800;color:{ac}">{current_nivel:.2f}m</div>
            <div style="font-size:0.85em;color:{ac}">{alert_emoji(current_alert)} {current_alert}</div>
            <div style="font-size:0.75em;color:#8899aa;">{trend_text} ({delta_1d:+.3f}m)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_risk:
        prob_pct = current_prob * 100
        risk_anim = "animation:risk_glow 1.5s infinite;" if current_prob >= 0.20 else ""
        st.markdown(f"""
        <div style="background:#1a1a2e;border:2px solid {risk_col};border-radius:12px;
             padding:16px;text-align:center;{risk_anim}">
            <div style="font-size:0.8em;color:#8899aa;">RISCO EXTREMO · {date_t0_str}</div>
            <div style="font-size:1.8em;font-weight:800;color:{risk_col}">{prob_pct:.1f}%</div>
            <div style="font-size:0.85em;color:{risk_col}">{risk_emo} {risk_cls}</div>
            <div style="font-size:0.75em;color:#8899aa;">P(Δ5d > 1m)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t5:
        a5 = alert_level(last_proj_T5)
        st.markdown(f"""
        <div style="background:#1a1a2e;border:2px solid {alert_color(a5)};border-radius:12px;
             padding:16px;text-align:center;">
            <div style="font-size:0.8em;color:#8899aa;">PROJEÇÃO T+5 · {date_t5_str}</div>
            <div style="font-size:1.8em;font-weight:800;color:{alert_color(a5)}">{last_proj_T5:.2f}m</div>
            <div style="font-size:0.85em;color:{alert_color(a5)}">{alert_emoji(a5)} {a5}</div>
            <div style="font-size:0.75em;color:#8899aa;">{t5_text} ({last_proj_T5 - current_nivel:+.3f}m)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Main Chart ──
    # Compute realized levels (actual level at D+N)
    view_df = view_df.copy()
    view_df["realizado_TN"] = view_df["guaiba_nivel_mean"].shift(-5)
    proj_col = "proj_T5"

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Actual level — green area with transparency
    fig.add_trace(go.Scatter(
        x=view_df["date"], y=view_df["guaiba_nivel_mean"],
        mode="lines", name="Nível atual",
        line=dict(color="rgba(76,175,80,0.6)", width=1.5),
        fill="tozeroy", fillcolor="rgba(76,175,80,0.15)",
        hovertemplate="%{x|%d/%m/%Y}: %{y:.3f}m<extra></extra>",
    ), secondary_y=False)

    # Realizado T+N — dark blue solid
    fig.add_trace(go.Scatter(
        x=view_df["date"], y=view_df["realizado_TN"],
        mode="lines+markers", name=f"Realizado T+5",
        line=dict(color="#0D47A1", width=2.5),
        marker=dict(size=5, color="#0D47A1"),
        hovertemplate=f"Real T+5: %{{y:.3f}}m<extra></extra>",
    ), secondary_y=False)

    # Projeção T+N — light blue dashed
    if proj_col in view_df.columns:
        fig.add_trace(go.Scatter(
            x=view_df["date"], y=view_df[proj_col],
            mode="lines", name=f"Projeção T+5",
            line=dict(color="#64B5F6", width=1.5, dash="dot"),
            hovertemplate=f"Proj T+5: %{{y:.3f}}m<extra></extra>",
        ), secondary_y=False)

    # ── Probability bars (secondary Y-axis) ──
    if "prob_extremo" in view_df.columns:
        # Color bars by risk level
        prob_colors = []
        for p in view_df["prob_extremo"] * 100:
            if p >= 20: prob_colors.append("#F44336")
            elif p >= 5: prob_colors.append("#FF9800")
            elif p >= 1: prob_colors.append("#F9A825")
            else: prob_colors.append("rgba(156,39,176,0.3)")
        fig.add_trace(go.Bar(
            x=view_df["date"], y=view_df["prob_extremo"] * 100,
            name="P(extremo)", marker_color=prob_colors, opacity=0.6,
            hovertemplate="P(Δ>1m): %{y:.1f}%<extra></extra>",
        ), secondary_y=True)

        # Threshold reference lines on secondary axis
        prob_max = max(view_df["prob_extremo"].max() * 100 * 1.2, 25)
        for thresh, lbl in [(1, "1%"), (5, "5%"), (20, "20%")]:
            fig.add_hline(y=thresh, line_dash="dot", line_color="rgba(156,39,176,0.3)", line_width=1,
                          annotation_text=lbl, annotation_position="top right",
                          annotation_font_size=10, annotation_font_color="rgba(156,39,176,0.6)",
                          secondary_y=True)
    else:
        prob_max = 25

    # Alert level lines
    for nivel, cor, label in [(1.0, "#4CAF50", "Atenção 1.0m"), (2.0, "#FF9800", "Alerta 2.0m"), (3.0, "#F44336", "INUNDAÇÃO 3.0m")]:
        fig.add_hline(y=nivel, line_dash="dash", line_color=cor, line_width=1,
                      annotation_text=label, annotation_position="top left",
                      annotation_font_size=11, annotation_font_color=cor,
                      secondary_y=False)

    # Colored bands
    ymax = max(view_df["guaiba_nivel_mean"].max(), 4.0) if len(view_df) > 0 else 4.0
    for y0, y1, color in [(0,1,"rgba(76,175,80,0.06)"),(1,2,"rgba(255,235,59,0.04)"),
                           (2,3,"rgba(255,152,0,0.04)"),(3,ymax,"rgba(244,67,54,0.06)")]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, secondary_y=False)

    fig.update_layout(
        height=560,
        title=dict(
            text=f"Nível do Guaíba e Projeção T+5<br><sup style='color:#8899aa'>Comparação entre realizado e previsto pelo modelo · Período: {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}</sup>",
            font=dict(color="#fff", size=16),
            x=0.01, xanchor="left",
        ),
        template="plotly_dark",
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#1a1a2e",
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickformat="%d%b%y"),
        legend=dict(
            orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
            font=dict(color="#ccc", size=11),
            bgcolor="rgba(26,26,46,0.8)",
            bordercolor="#2a2a4a",
            borderwidth=1,
        ),
        margin=dict(l=60, r=60, t=40, b=80),
        hovermode="x unified",
    )

    # Primary Y-axis (left)
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.04)", ticksuffix="m", range=[0, ymax],
        secondary_y=False
    )
    # Secondary Y-axis (right)
    fig.update_yaxes(
        title_text="P(Δ>1m)", title_font=dict(color="#9C27B0"),
        tickfont=dict(color="#9C27B0"), ticksuffix="%",
        range=[0, prob_max], showgrid=False,
        secondary_y=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Rain Summary ──
    if "chuva_total_raw" in last.index:
        chuva_hoje = float(last.get("chuva_total_raw", 0) or 0)
        chuva_3d = float(last.get("chuva_total_acc_3d", 0) or 0)
        chuva_7d = float(last.get("chuva_total_acc_7d", 0) or 0)
    else:
        chuva_hoje = chuva_3d = chuva_7d = 0

    def rain_icon(mm):
        if mm <= 0: return "☀️"
        elif mm <= 5: return "⛅"
        elif mm <= 20: return "🌤️"
        elif mm <= 50: return "🌧️"
        else: return "⛈️"

    st.markdown(f"### 🌧️ Precipitação")
    st.caption(f"Soma da chuva em todas as 8 estações monitoradas · Referência: {date_t0_str}")
    rc1, rc2, rc3 = st.columns(3)
    for col, val, label, sub in [(rc1, chuva_hoje, "Hoje", "Chuva em 24h"), (rc2, chuva_3d, "Acumulado 3d", "Soma últimos 3 dias"), (rc3, chuva_7d, "Acumulado 7d", "Soma últimos 7 dias")]:
        c = rain_color_mm(val)
        icon = rain_icon(val)
        col.markdown(f"""
        <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;
             padding:16px 20px;display:flex;align-items:center;gap:14px;">
            <div style="font-size:2em;">{icon}</div>
            <div>
                <div style="font-size:1.8em;font-weight:700;color:{c}">{val:.1f} mm</div>
                <div style="font-size:0.85em;color:#ccc;">{label}</div>
                <div style="font-size:0.75em;color:#8899aa;">{sub}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature Monitor Table (Diagnostic Panel) ──
    st.markdown("### 📊 Painel de Diagnóstico")
    st.caption(f"Valores medidos em {date_t0_str} e comparados com a distribuição histórica (2019–2025) para identificar anomalias · Percentis: P0-P75 normal, P75-P90 elevado, P90-P95 muito alto, P95-P99 extremo, P99+ crítico")

    dev_df = load_dev_data()

    # Build table rows — grouped by FEATURE_META.group
    GROUP_ORDER = ["Chuva", "Vento (direção)", "Vento (velocidade)", "Nível", "Saturação do solo", "Contexto"]
    GROUP_COLORS = {
        "Chuva": "rgba(33,150,243,0.08)",
        "Vento (direção)": "rgba(156,39,176,0.08)",
        "Vento (velocidade)": "rgba(0,150,136,0.08)",
        "Nível": "rgba(76,175,80,0.08)",
        "Saturação do solo": "rgba(255,152,0,0.08)",
        "Contexto": "rgba(158,158,158,0.08)",
    }

    # Compute contributions for SFS features if binary model is loaded
    feature_contributions = {}  # feat -> (coef, woe, contribution)
    # Load binary model feature list
    try:
        with open(PROJECT / "data" / "processed" / "sfs_results_logreg_optbin.json") as f:
            ob_feats = json.load(f)["features"]
    except Exception:
        ob_feats = []

    feature_contributions = {}
    if binary_model_ok:
        for feat in ob_feats:
            contrib_col = f"contrib_{feat}"
            if contrib_col in last.index:
                feature_contributions[feat] = {"contribution": float(last[contrib_col])}

    # Collect all features with their metadata, bucketed by group
    all_feats = sorted(set(ALL_MODEL_FEATURES) | EXTRA_KEYS | OFF_MODEL_1D)
    # Also include the SFS binary features that may not be in the above sets
    for bf in ob_feats:
        if binary_model_ok and bf not in all_feats:
            all_feats.append(bf)
    grouped = {}
    for feat in all_feats:
        meta = FEATURE_META.get(feat)
        if meta is None:
            continue
        grp = meta.get("group", "Contexto")
        is_extra = feat in EXTRA_KEYS or feat in OFF_MODEL_1D
        imp = importance_map.get(feat, 0) if not is_extra else -1
        contrib = feature_contributions.get(feat, {}).get("contribution", None)
        grouped.setdefault(grp, []).append((feat, imp, is_extra, contrib))

    rows_html = ""
    n_extra = 0
    status_counts = {"Normal": 0, "Elevado": 0, "Muito Alto": 0, "Extremo": 0, "CRÍTICO": 0}
    is_elevated_risk = current_prob >= 0.05

    # Count contributing features
    n_contributing = sum(1 for v in feature_contributions.values() if v["contribution"] > 0)

    # Banner
    if is_elevated_risk:
        banner_bg = "rgba(244,67,54,0.08)"
        banner_border = "#F44336"
        banner_text = f"⚠️ Risco elevado detectado — variáveis-chave identificadas · P(extremo) = {current_prob*100:.1f}%"
    else:
        banner_bg = "rgba(76,175,80,0.08)"
        banner_border = "#4CAF50"
        banner_text = "✅ Cenário normal — nenhuma variável em regime extremo"

    st.markdown(f"""
    <div style="background:{banner_bg};border:1px solid {banner_border};border-radius:10px;
         padding:12px 18px;margin-bottom:12px;font-size:0.9em;color:{banner_border};">
        {banner_text}
    </div>
    """, unsafe_allow_html=True)

    # Determine top 3 contributing features for highlighting
    top3_contrib = set()
    if is_elevated_risk and feature_contributions:
        sorted_contribs = sorted(
            [(f, c["contribution"]) for f, c in feature_contributions.items() if c["contribution"] > 0],
            key=lambda x: -x[1]
        )
        top3_contrib = set(f for f, _ in sorted_contribs[:3])

    for grp in GROUP_ORDER:
        feats = grouped.get(grp, [])
        if not feats:
            continue

        if is_elevated_risk:
            # Sort: positive contribution first (highest), then negative (lowest), then N/A
            def sort_key(x):
                feat, imp, is_extra, contrib = x
                if contrib is not None:
                    if contrib > 0:
                        return (0, -contrib)  # positive first, highest first
                    else:
                        return (1, contrib)   # negative, most negative first
                return (2, 0)  # N/A last
            feats.sort(key=sort_key)
        else:
            # Default sort: model features first by importance desc, then off-model
            feats.sort(key=lambda x: (x[2], -x[1]))

        grp_bg = GROUP_COLORS.get(grp, "rgba(158,158,158,0.08)")
        rows_html += f'<tr><td colspan="5" style="background:{grp_bg};padding:8px 10px;font-weight:700;font-size:0.9em;color:#bbb;letter-spacing:0.5px;border-top:1px solid #2a2a4a;">{grp}</td></tr>'

        for feat, imp_val, is_extra, contrib in feats:
            meta = FEATURE_META[feat]
            ftype = meta.get("type", "other")
            val = float(last[feat]) if feat in last.index and not pd.isna(last[feat]) else 0
            col_dev = dev_df[feat].dropna() if feat in dev_df.columns else pd.Series(dtype=float)
            pct = float((col_dev <= val).mean() * 100) if len(col_dev) > 0 else 0
            cls = percentile_class(pct)

            is_extra = feat in EXTRA_KEYS or feat in OFF_MODEL_1D
            if is_extra: n_extra += 1

            # Highlight top 3 contributing features
            is_top3 = feat in top3_contrib
            if is_top3:
                row_bg = "background:rgba(244,67,54,0.06);animation:risk_glow 2s infinite;"
            elif is_extra:
                row_bg = "background:rgba(255,152,0,0.04);"
            else:
                row_bg = ""
            desc_style = "font-style:italic;color:#aaa;" if is_extra else "color:#ddd;"

            # Importance display
            if not is_extra and feat in importance_map and importance_map[feat] > 0:
                imp_pct = importance_map[feat]
                imp_html = f'<span style="color:#64B5F6;font-size:1em;font-weight:600;">{imp_pct:.0f}%</span>'
            else:
                imp_html = '<span style="color:#555;font-size:0.9em;">—</span>'

            # Contribution display
            if contrib is not None and binary_model_ok:
                if contrib > 0.01:
                    contrib_html = f'<span style="color:#F44336;font-weight:700;font-size:0.95em;">+{contrib:.2f} ↑</span>'
                elif contrib < -0.01:
                    contrib_html = f'<span style="color:#4CAF50;font-weight:700;font-size:0.95em;">{contrib:.2f} ↓</span>'
                else:
                    contrib_html = f'<span style="color:#8899aa;font-size:0.9em;">{contrib:+.2f}</span>'
            else:
                contrib_html = '<span style="color:#555;font-size:0.9em;">—</span>'

            pct_colors = {"normal":"#4CAF50","elevated":"#F9A825","very_high":"#FF9800","extreme":"#F44336","critical":"#B71C1C"}
            pct_labels = {"normal":"Normal","elevated":"Elevado","very_high":"Muito Alto","extreme":"Extremo","critical":"CRÍTICO"}
            pc = pct_colors.get(cls, "#4CAF50")
            pl = pct_labels.get(cls, "Normal")
            pulse = "animation:pulse 1.5s infinite;" if cls == "critical" else ""

            # Wind direction: show arrow + compass + impact classification from JSON
            if ftype == "wind_dir" or (ftype == "off_model" and "wind_dir" in feat):
                arrow = deg_to_arrow(val)
                compass_pt = deg_to_compass_pt(val)
                compass_abbr = deg_to_compass(val)
                # Look up classification from wind_impact JSON
                wi_class = "neutral"
                wi_impact = 0.0
                try:
                    if feat in wind_impact.get("classification", {}) and compass_abbr in wind_impact["classification"][feat]:
                        ci = wind_impact["classification"][feat][compass_abbr]
                        wi_class = ci.get("class", "neutral")
                        wi_impact = ci.get("avg_impact", 0.0)
                except Exception:
                    pass
                wi_color_map = {"better": "#4CAF50", "neutral": "#8899aa", "worse": "#F44336"}
                wi_label_map = {"better": "🟢 Favorável", "neutral": "⚪ Neutro", "worse": "🔴 Desfavorável"}
                wi_color = wi_color_map.get(wi_class, "#8899aa")
                wi_label = wi_label_map.get(wi_class, "⚪ Neutro")
                val_html = f'<span style="color:#fff;font-weight:500;">{arrow} {compass_pt}</span>'
                pct_html = f'<span style="background:{wi_color}22;color:{wi_color};padding:3px 8px;border-radius:6px;font-size:0.8em;font-weight:600;">{wi_label}</span>'
                bar_html = '<span style="color:#666;">—</span>'
                # Count wind direction status for summary
                if wi_class == "better":
                    status_counts["Normal"] += 1
                elif wi_class == "worse":
                    status_counts["Extremo"] += 1
                else:
                    status_counts["Elevado"] += 1
            else:
                val_html = f'{val:.3f}'
                pct_html = f'<span style="background:{pc}22;color:{pc};padding:3px 8px;border-radius:6px;font-size:0.8em;font-weight:600;{pulse}">{pl}</span>'
                status_counts[pl] = status_counts.get(pl, 0) + 1
                bar_html = f'''<div style="width:120px;height:10px;background:#1a1a2e;border-radius:5px;overflow:hidden;border:1px solid #2a2a4a;">
                    <div style="width:{min(pct,100):.0f}%;height:100%;background:{pc};border-radius:5px;"></div>
                </div>'''

            rows_html += f"""
        <tr style="{row_bg}">
            <td>
                <div style="font-weight:500;{desc_style}">{meta['desc']}</div>
                <div style="font-size:0.78em;color:#666;margin-top:2px;">{meta['interp']}</div>
            </td>
            <td style="font-weight:600;">{val_html}</td>
            <td>{contrib_html}</td>
            <td>{imp_html}</td>
            <td>{pct_html}</td>
        </tr>"""

    # Build summary badges
    total_vars = sum(status_counts.values())
    _sum_emoji = {"Normal": "🟢", "Elevado": "🟡", "Muito Alto": "🟠", "Extremo": "🔴", "CRÍTICO": "💀"}
    _sum_color = {"Normal": "#4CAF50", "Elevado": "#F9A825", "Muito Alto": "#FF9800", "Extremo": "#F44336", "CRÍTICO": "#B71C1C"}
    _sum_parts = []
    for _sn in ["Normal", "Elevado", "Muito Alto", "Extremo", "CRÍTICO"]:
        _sc = status_counts.get(_sn, 0)
        _sp = (_sc / total_vars * 100) if total_vars > 0 else 0
        _sum_parts.append(f'<span style="background:{_sum_color[_sn]}22;color:{_sum_color[_sn]};padding:4px 10px;border-radius:8px;font-size:0.85em;font-weight:600;">{_sum_emoji[_sn]} {_sn}: {_sc} ({_sp:.0f}%)</span>')

    # Add risk extremo line if elevated
    if is_elevated_risk and binary_model_ok:
        _risk_line = f'<div style="margin-top:8px;"><span style="background:{risk_col}22;color:{risk_col};padding:4px 12px;border-radius:8px;font-size:0.9em;font-weight:700;">🔴 Risco extremo: {current_prob*100:.1f}% — {n_contributing} variável(is) contribuindo</span></div>'
    else:
        _risk_line = ""

    summary_html = ' <span style="color:#555;">·</span> '.join(_sum_parts) + _risk_line
    st.markdown(f'<div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:14px 18px;margin-bottom:10px;text-align:center;">{summary_html}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:16px;padding:20px;overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:0.88em;">
    <thead>
        <tr style="border-bottom:2px solid #2a2a4a;">
            <th style="text-align:left;padding:10px;color:#8899aa;font-size:0.8em;text-transform:uppercase;letter-spacing:1px;">Variável</th>
            <th style="text-align:left;padding:10px;color:#8899aa;font-size:0.8em;">Valor</th>
            <th style="text-align:left;padding:10px;color:#8899aa;font-size:0.8em;">Contribuição</th>
            <th style="text-align:left;padding:10px;color:#8899aa;font-size:0.8em;">Importância</th>
            <th style="text-align:left;padding:10px;color:#8899aa;font-size:0.8em;">Status</th>
        </tr>
    </thead>
    <tbody>{rows_html}</tbody>
    </table>
    <div style="margin-top:12px;padding:10px 14px;background:rgba(255,255,255,0.03);border-radius:8px;font-size:0.82em;color:#8899aa;border-top:1px solid #2a2a4a;">
        ⚠️ Variáveis em itálico (fundo levemente alaranjado) <b>NÃO fazem parte dos modelos 3d/5d</b> — são incluídas como contexto adicional (delta_1d SFS ou variáveis de referência). A coluna "Importância" mostra "—" para essas variáveis.<br>
        📊 <b>Contribuição</b> = impacto da variável na probabilidade de evento extremo (modelo LogReg+OptBin). Valores positivos ↑ aumentam o risco; valores negativos ↓ reduzem o risco. Apenas as 7 variáveis do modelo binário têm contribuição calculada.
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Glossary ──
    st.markdown("### 📖 Glossário Visual")
    glossary = [
        ("🌊", "Nível", "Quão alto está a água do rio, medido em metros. Quando passa de 3 metros, há risco de inundação."),
        ("📈", "Delta", "Quanto o nível vai subir ou descer nos próximos dias. Se o delta é +1.5m, a água vai subir 1.5 metros."),
        ("🔮", "Projeção", "O que o modelo prevê para o nível nos próximos dias. Linha tracejada azul claro no gráfico."),
        ("✅", "Realizado", "O nível que de fato aconteceu N dias depois. Linha sólida azul escuro no gráfico. Só existe para datas passadas."),
        ("🚦", "Cota", "O limite oficial de segurança. Cada cor do semáforo representa uma cota diferente."),
        ("⚠️", "Alerta", "Quando o nível previsto passa de 2 metros, a Defesa Civil entra em ação para preparar a cidade."),
        ("🌬️", "Represamento", "Quando o vento forte empurra a água da Lagoa dos Patos para dentro do Guaíba, impedindo que ela saia."),
        ("🏞️", "Bacia", "A área de terra onde a chuva cai e escorre para o rio. Cada rio tem sua bacia."),
        ("📊", "Percentil", "Se o valor está no percentil 95, significa que ele é maior que 95% dos valores históricos — bem acima do normal."),
        ("📐", "Z-score", "Quantos desvios padrão o valor está da média. Z=3 significa 3x acima do normal — muito raro."),
    ]

    cols = st.columns(5)
    for i, (icon, title, text) in enumerate(glossary):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:16px;margin-bottom:8px;">
                <div style="font-size:2em;margin-bottom:6px;">{icon}</div>
                <div style="font-size:1em;font-weight:700;color:#fff;margin-bottom:4px;">{title}</div>
                <div style="font-size:0.82em;color:#aaa;line-height:1.5;">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("---")
    model_info = "CatBoost (delta_5d)"
    if binary_model_ok:
        model_info += " · LogReg+OptBin (P extremo)"
    st.caption(f"Sistema de previsão de enchentes do Rio Guaíba · Modelos: {model_info} · Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")


if __name__ == "__main__":
    main()
