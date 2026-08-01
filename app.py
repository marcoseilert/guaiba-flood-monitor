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
    # ── Vento (direção) ──
    "encantado_wind_dir_deg":          {"desc": "Direção do vento em Encantado (°)", "interp": "Rosa dos ventos: N/NW=sobe, S/SE=desce", "group": "Vento (direção)", "type": "wind_dir"},
    "cachoeira_do_sul_wind_dir_deg":   {"desc": "Direção do vento em Cachoeira do Sul (°)", "interp": "Rosa dos ventos: N/NW=sobe, S/SE=desce", "group": "Vento (direção)", "type": "wind_dir"},
    # ── Vento (velocidade) ──
    "feliz_wind_max_kmh":        {"desc": "Rajada máxima de vento em Feliz (km/h)", "interp": "Ventos fortes podem causar represamento no Guaíba", "group": "Vento (velocidade)", "type": "wind_speed"},
    "mostardas_v_wind":          {"desc": "Vento sul em Mostardas — hoje (m/s)", "interp": "Positivo=vento sul → represamento. Negativo=vento norte → drenagem", "group": "Vento (velocidade)", "type": "wind_speed"},
    "mostardas_v_wind_roll2":    {"desc": "Vento sul em Mostardas — média 2 dias (m/s)", "interp": "Positivo=vento sul → represamento (empurra Lagoa dos Patos para o Guaíba). Negativo=vento norte → drenagem", "group": "Vento (velocidade)", "type": "wind_speed"},
    "mostardas_v_wind_roll3":    {"desc": "Vento sul em Mostardas — média 3 dias (m/s)", "interp": "Positivo=vento sul persistente → represamento acumulado. Negativo=vento norte → drenagem", "group": "Vento (velocidade)", "type": "wind_speed"},
    "estrela_wind_max_kmh":      {"desc": "Rajada máxima de vento em Estrela (km/h)", "interp": "Ventos fortes no vale do Taquari", "group": "Vento (velocidade)", "type": "wind_speed"},
    # ── Nível ──
    "taquari_mucum_delta3":      {"desc": "Variação do nível em Muçum em 3 dias (m)", "interp": "Se positivo, o rio Taquari está subindo — água a caminho do Guaíba", "group": "Nível", "type": "level"},
    "sinos_sl_nivel_mean":       {"desc": "Nível do rio Sinos em São Leopoldo (m)", "interp": "Nível alto indica contribuição da bacia dos Sinos", "group": "Nível", "type": "level"},
    "catsul_nivel_mean_lag5":    {"desc": "Nível no Terminal CATSUL — 5 dias atrás (m)", "interp": "Nível passado no Guaíba a jusante — indica tendência de longo prazo", "group": "Nível", "type": "level"},
    "sinos_sl_nivel_mean_lag5":  {"desc": "Nível do Sinos em São Leopoldo — 5 dias atrás (m)", "interp": "Nível passado indica propagação lenta da cheia", "group": "Nível", "type": "level"},
    "jacui_rp_nivel_mean_lag7":  {"desc": "Nível do Jacuí em Rio Pardo — 7 dias atrás (m)", "interp": "Nível com longo lag — captura a propagação lenta do Jacuí", "group": "Nível", "type": "level"},
    "guaiba_nivel_mean":         {"desc": "Nível atual do Guaíba — T0 (m)", "interp": "O nível medido hoje na régua de Porto Alegre", "group": "Nível", "type": "level"},
    "guaiba_delta1":             {"desc": "Variação do nível do Guaíba em 1 dia (m)", "interp": "Se positivo, o nível subiu desde ontem", "group": "Nível", "type": "level"},
    "guaiba_delta3":             {"desc": "Variação do nível do Guaíba em 3 dias (m)", "interp": "Tendência de 3 dias — mostra se está subindo ou descendo", "group": "Nível", "type": "level"},
    # ── Saturação do solo ──
    "dias_chuvosos_7d":          {"desc": "Dias com chuva nos últimos 7 dias", "interp": "0-7: quantos dias choveu. Solo saturado escoa mais rápido", "group": "Saturação do solo", "type": "other"},
    "dias_chuvosos_14d":         {"desc": "Dias com chuva nos últimos 14 dias", "interp": "0-14: frequência de chuva — proxy de saturação do solo", "group": "Saturação do solo", "type": "other"},
    # ── Contexto (extra — not in models) ──
    "chuva_total_raw":           {"desc": "Chuva total hoje em todas as bacias (mm)", "interp": "Soma da chuva de hoje em todas as 8 estações", "group": "Contexto", "type": "other"},
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
    result = subprocess.run(
        ["python", str(UPDATE_SCRIPT)],
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

    # Auto-update if data is behind
    if days_behind > 0:
        with st.spinner(f"📅 Atualizando dados (+{days_behind} dia{'s' if days_behind > 1 else ''})..."):
            success, output = run_update()
        if success:
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(f"Erro ao atualizar:\n{output}")

    st.sidebar.caption(f"📂 Dados: {chart_df['date'].min().date()} → {last_date} ({len(chart_df)} dias)")
    st.sidebar.success("✅ Dados atualizados")

    # Horizon selector
    st.sidebar.markdown("---")
    horizon = st.sidebar.radio("🔮 Horizonte de previsão", ["T+3 (3 dias)", "T+5 (5 dias)"], index=1, horizontal=True)
    horizon_days = 3 if "T+3" in horizon else 5

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
    # If missing (shouldn't happen), compute inline
    if "proj_T3" not in chart_df.columns:
        import pickle as pkl
        models_dir = PROJECT / "models"
        with open(models_dir / "model_metadata.pkl", "rb") as f:
            meta = pkl.load(f)
        with open(models_dir / "model_delta_3d.pkl", "rb") as f:
            m3d = pkl.load(f)
        with open(models_dir / "model_delta_5d.pkl", "rb") as f:
            m5d = pkl.load(f)
        chart_df["pred_delta_3d"] = m3d.predict(chart_df[meta["features_3d"]].values)
        chart_df["pred_delta_5d"] = m5d.predict(chart_df[meta["features_5d"]].values)
        chart_df["proj_T3"] = chart_df["guaiba_nivel_mean"] + chart_df["pred_delta_3d"]
        chart_df["proj_T5"] = chart_df["guaiba_nivel_mean"] + chart_df["pred_delta_5d"]

    # Filter view
    mask = (chart_df["date"] >= pd.Timestamp(start_date)) & (chart_df["date"] <= pd.Timestamp(end_date))
    view_df = chart_df[mask].copy()

    # ── Current state (last day of selected range) ──
    if len(view_df) == 0:
        st.warning("Nenhum dado no período selecionado.")
        st.stop()
    last = view_df.iloc[-1]
    last_proj_T3 = float(last["proj_T3"]) if "proj_T3" in last.index and not pd.isna(last.get("proj_T3")) else 0
    last_proj_T5 = float(last["proj_T5"]) if "proj_T5" in last.index and not pd.isna(last.get("proj_T5")) else 0

    current_nivel = float(last["guaiba_nivel_mean"])
    current_alert = alert_level(current_nivel)
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
    date_t3_str = (last_date_ts + timedelta(days=3)).strftime("%d/%m/%Y")
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

    t3_text, t3_color = proj_trend(last_proj_T3, current_nivel)
    t5_text, t5_color = proj_trend(last_proj_T5, current_nivel)

    col_sema, col_t3, col_t5 = st.columns(3)

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

    with col_t3:
        a3 = alert_level(last_proj_T3)
        st.markdown(f"""
        <div style="background:#1a1a2e;border:2px solid {alert_color(a3)};border-radius:12px;
             padding:16px;text-align:center;">
            <div style="font-size:0.8em;color:#8899aa;">PROJEÇÃO T+3 · {date_t3_str}</div>
            <div style="font-size:1.8em;font-weight:800;color:{alert_color(a3)}">{last_proj_T3:.2f}m</div>
            <div style="font-size:0.85em;color:{alert_color(a3)}">{alert_emoji(a3)} {a3}</div>
            <div style="font-size:0.75em;color:#8899aa;">{t3_text} ({last_proj_T3 - current_nivel:+.3f}m)</div>
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
    view_df["realizado_TN"] = view_df["guaiba_nivel_mean"].shift(-horizon_days)
    proj_col = "proj_T3" if horizon_days == 3 else "proj_T5"

    fig = go.Figure()

    # Actual level — green area with transparency
    fig.add_trace(go.Scatter(
        x=view_df["date"], y=view_df["guaiba_nivel_mean"],
        mode="lines", name="Nível atual",
        line=dict(color="rgba(76,175,80,0.6)", width=1.5),
        fill="tozeroy", fillcolor="rgba(76,175,80,0.15)",
        hovertemplate="%{x|%d/%m/%Y}: %{y:.3f}m<extra></extra>",
    ))

    # Realizado T+N — dark blue solid
    fig.add_trace(go.Scatter(
        x=view_df["date"], y=view_df["realizado_TN"],
        mode="lines+markers", name=f"Realizado T+{horizon_days}",
        line=dict(color="#0D47A1", width=2.5),
        marker=dict(size=5, color="#0D47A1"),
        hovertemplate=f"Real T+{horizon_days}: %{{y:.3f}}m<extra></extra>",
    ))

    # Projeção T+N — light blue dashed
    if proj_col in view_df.columns:
        fig.add_trace(go.Scatter(
            x=view_df["date"], y=view_df[proj_col],
            mode="lines", name=f"Projeção T+{horizon_days}",
            line=dict(color="#64B5F6", width=1.5, dash="dot"),
            hovertemplate=f"Proj T+{horizon_days}: %{{y:.3f}}m<extra></extra>",
        ))

    # Alert level lines
    for nivel, cor, label in [(1.0, "#4CAF50", "Atenção 1.0m"), (2.0, "#FF9800", "Alerta 2.0m"), (3.0, "#F44336", "INUNDAÇÃO 3.0m")]:
        fig.add_hline(y=nivel, line_dash="dash", line_color=cor, line_width=1,
                      annotation_text=label, annotation_position="top left",
                      annotation_font_size=11, annotation_font_color=cor)

    # Colored bands
    ymax = max(view_df["guaiba_nivel_mean"].max(), 4.0) if len(view_df) > 0 else 4.0
    for y0, y1, color in [(0,1,"rgba(76,175,80,0.06)"),(1,2,"rgba(255,235,59,0.04)"),
                           (2,3,"rgba(255,152,0,0.04)"),(3,ymax,"rgba(244,67,54,0.06)")]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0)

    fig.update_layout(
        height=560,
        title=dict(
            text=f"Nível do Guaíba e Projeção T+{horizon_days}<br><sup style='color:#8899aa'>Comparação entre realizado e previsto pelo modelo · Período: {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}</sup>",
            font=dict(color="#fff", size=16),
            x=0.01, xanchor="left",
        ),
        template="plotly_dark",
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#1a1a2e",
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickformat="%d%b%y"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", ticksuffix="m", range=[0, ymax]),
        legend=dict(
            orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
            font=dict(color="#ccc", size=11),
            bgcolor="rgba(26,26,46,0.8)",
            bordercolor="#2a2a4a",
            borderwidth=1,
        ),
        margin=dict(l=60, r=30, t=40, b=80),
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

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

    # ── Feature Monitor Table ──
    st.markdown("### 📊 Monitor de Variáveis")
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

    # Collect all features with their metadata, bucketed by group
    all_feats = sorted(set(ALL_MODEL_FEATURES) | EXTRA_KEYS)
    grouped = {}
    for feat in all_feats:
        meta = FEATURE_META.get(feat)
        if meta is None:
            continue
        grp = meta.get("group", "Contexto")
        grouped.setdefault(grp, []).append(feat)

    rows_html = ""
    n_extra = 0
    status_counts = {"Normal": 0, "Elevado": 0, "Muito Alto": 0, "Extremo": 0, "CRÍTICO": 0}
    for grp in GROUP_ORDER:
        feats = grouped.get(grp, [])
        if not feats:
            continue
        grp_bg = GROUP_COLORS.get(grp, "rgba(158,158,158,0.08)")
        rows_html += f'<tr><td colspan="5" style="background:{grp_bg};padding:8px 10px;font-weight:700;font-size:0.9em;color:#bbb;letter-spacing:0.5px;border-top:1px solid #2a2a4a;">{grp}</td></tr>'

        for feat in feats:
            meta = FEATURE_META[feat]
            ftype = meta.get("type", "other")
            val = float(last[feat]) if feat in last.index and not pd.isna(last[feat]) else 0
            col_dev = dev_df[feat].dropna() if feat in dev_df.columns else pd.Series(dtype=float)
            pct = float((col_dev <= val).mean() * 100) if len(col_dev) > 0 else 0
            cls = percentile_class(pct)

            models = []
            if feat in FEATURES_3D: models.append("3d")
            if feat in FEATURES_5D: models.append("5d")
            model_str = " + ".join(models) if models else "—"

            is_extra = feat in EXTRA_KEYS
            if is_extra: n_extra += 1

            row_bg = "background:rgba(255,152,0,0.04);" if is_extra else ""
            desc_style = "font-style:italic;color:#aaa;" if is_extra else "color:#ddd;"
            model_bg = "background:rgba(255,152,0,0.15);color:#FF9800;" if is_extra else "background:rgba(33,150,243,0.15);color:#2196F3;"

            pct_colors = {"normal":"#4CAF50","elevated":"#F9A825","very_high":"#FF9800","extreme":"#F44336","critical":"#B71C1C"}
            pct_labels = {"normal":"Normal","elevated":"Elevado","very_high":"Muito Alto","extreme":"Extremo","critical":"CRÍTICO"}
            pc = pct_colors.get(cls, "#4CAF50")
            pl = pct_labels.get(cls, "Normal")
            pulse = "animation:pulse 1.5s infinite;" if cls == "critical" else ""

            # Wind direction: show arrow + compass + impact classification from JSON
            if ftype == "wind_dir":
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
            <td><span style="{model_bg}padding:2px 8px;border-radius:4px;font-size:0.8em;">{model_str}</span></td>
            <td style="font-weight:600;">{val_html}</td>
            <td>{pct_html}</td>
            <td>{bar_html}</td>
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
    summary_html = ' <span style="color:#555;">·</span> '.join(_sum_parts)
    st.markdown(f'<div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:14px 18px;margin-bottom:10px;text-align:center;">{summary_html}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:16px;padding:20px;overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:0.88em;">
    <thead>
        <tr style="border-bottom:2px solid #2a2a4a;">
            <th style="text-align:left;padding:10px;color:#8899aa;font-size:0.8em;text-transform:uppercase;letter-spacing:1px;">Variável</th>
            <th style="text-align:left;padding:10px;color:#8899aa;font-size:0.8em;">Modelo</th>
            <th style="text-align:left;padding:10px;color:#8899aa;font-size:0.8em;">Valor</th>
            <th style="text-align:left;padding:10px;color:#8899aa;font-size:0.8em;">Status</th>
            <th style="text-align:left;padding:10px;color:#8899aa;font-size:0.8em;">Barra</th>
        </tr>
    </thead>
    <tbody>{rows_html}</tbody>
    </table>
    <div style="margin-top:12px;padding:10px 14px;background:rgba(255,255,255,0.03);border-radius:8px;font-size:0.82em;color:#8899aa;border-top:1px solid #2a2a4a;">
        ⚠️ As últimas {n_extra} linhas (em itálico, fundo levemente alaranjado) <b>NÃO fazem parte dos modelos</b> — são incluídas apenas como contexto adicional. A coluna "Modelo" mostra "—" para essas variáveis.
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
    st.caption(f"Sistema de previsão de enchentes do Rio Guaíba · Modelos: LightGBM (delta_3d) + CatBoost (delta_5d) · Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")


if __name__ == "__main__":
    main()
