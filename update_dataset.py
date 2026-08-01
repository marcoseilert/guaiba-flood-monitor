# -*- coding: utf-8 -*-
"""
update_dataset.py — Incremental dataset updater for Guaíba Flood Prediction v2
===============================================================================
Run: python update_dataset.py

Reads the existing unified dataset, fetches new data from the last available
date to today, appends it, recomputes targets (backfill T+3/T+5), and saves.
"""

import os
import pickle
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
for d in [DATA_RAW, DATA_PROCESSED]:
    d.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_PROCESSED / "dataset_historico.parquet"

# ── Constants ────────────────────────────────────────────────────────────────
ANA_POST_URL = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx/DadosHidrometeorologicos"
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"

TARGET_STATIONS = ["87450004", "87444000"]
UPSTREAM_STATIONS = ["87242000", "87399000", "87382000", "87380000",
                     "87270000", "86510000", "85900000"]
ALL_HYDRO_STATIONS = TARGET_STATIONS + UPSTREAM_STATIONS

METEO_POINTS = {
    "porto_alegre":     (-30.03, -51.23),
    "rio_grande":       (-32.03, -52.10),
    "mostardas":        (-31.11, -50.92),
    "mucum":            (-29.16, -51.87),
    "encantado":        (-29.23, -51.87),
    "estrela":          (-29.50, -51.96),
    "feliz":            (-29.45, -51.30),
    "campo_bom":        (-29.67, -51.06),
    "sao_leopoldo":     (-29.76, -51.15),
    "cachoeira_do_sul": (-30.03, -52.89),
}

LEVEL_LAGS       = [1, 2, 3, 5, 7]
LEVEL_DELTAS     = [1, 2, 3]
RAIN_WINDOWS     = [3, 7, 14, 30, 60]
WIND_ROLLING     = [1, 2, 3]
FORECAST_HORIZONS = [1, 2, 3, 5]

MAX_RETRIES = 5
RETRY_DELAY = 5

UPSTREAM_NAMES = {
    "87242000": "catsul",
    "87399000": "gravatai_sl",
    "87382000": "sinos_sl",
    "87380000": "sinos_cb",
    "87270000": "cai_pm",
    "86510000": "taquari_mucum",
    "85900000": "jacui_rp",
}


# ══════════════════════════════════════════════════════════════
# FETCH FUNCTIONS (same as build_prd_dataset.py)
# ══════════════════════════════════════════════════════════════
def fetch_ana_station(cod_estacao, data_inicio, data_fim):
    post_data = {"codEstacao": cod_estacao, "dataInicio": data_inicio, "dataFim": data_fim}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(ANA_POST_URL, data=post_data, timeout=120)
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"    [ERROR] {cod_estacao}: {e}")
                return pd.DataFrame()
            time.sleep(RETRY_DELAY * attempt)

    xml_text = resp.text
    if "<Error>" in xml_text or not xml_text.strip():
        return pd.DataFrame()

    rows = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return pd.DataFrame()

    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag == "DadosHidrometereologicos":
            row = {}
            for child in elem:
                ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                row[ctag] = child.text
            if row and "DataHora" in row:
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["DataHora"].str.strip(), errors="coerce")
    for col_src, col_dst, divisor in [("Nivel", "nivel_m", 100.0),
                                       ("Chuva", "chuva_mm", 1.0),
                                       ("Vazao", "vazao_m3s", 1.0)]:
        if col_src in df.columns:
            vals = df[col_src].replace("", np.nan)
            df[col_dst] = pd.to_numeric(vals, errors="coerce") / divisor
        else:
            df[col_dst] = np.nan
    df = df[["datetime", "nivel_m", "chuva_mm", "vazao_m3s"]].dropna(subset=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def aggregate_daily(df):
    if df.empty:
        return df
    df = df.copy()
    df["date"] = df["datetime"].dt.date
    daily = df.groupby("date").agg(
        nivel_mean_m=("nivel_m", "mean"),
        nivel_max_m=("nivel_m", "max"),
        chuva_sum_mm=("chuva_mm", "sum"),
        vazao_mean_m3s=("vazao_m3s", "mean"),
        n_obs=("nivel_m", "count"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def fetch_openmeteo_point(name, lat, lon, start_date, end_date):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": "precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant",
        "timezone": "America/Sao_Paulo",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(OPENMETEO_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                return pd.DataFrame()
            time.sleep(RETRY_DELAY * attempt)

    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        return pd.DataFrame()

    return pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        f"{name}_precip_mm": daily.get("precipitation_sum"),
        f"{name}_wind_max_kmh": daily.get("wind_speed_10m_max"),
        f"{name}_wind_dir_deg": daily.get("wind_direction_10m_dominant"),
    })


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (same as build_prd_dataset.py)
# ══════════════════════════════════════════════════════════════
def build_all_features(df):
    df = df.set_index("date").sort_index()

    for alias in UPSTREAM_NAMES.values():
        col = f"{alias}_nivel_mean"
        if col in df.columns:
            for lag in LEVEL_LAGS:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    for alias in UPSTREAM_NAMES.values():
        col = f"{alias}_nivel_mean"
        if col in df.columns:
            for d in LEVEL_DELTAS:
                df[f"{alias}_delta{d}"] = df[col] - df[col].shift(d)

    rain_cols = [f"{a}_chuva" for a in UPSTREAM_NAMES.values() if f"{a}_chuva" in df.columns]
    if "guaiba_chuva" in df.columns:
        rain_cols.append("guaiba_chuva")
    for rc in rain_cols:
        for w in RAIN_WINDOWS:
            df[f"{rc}_roll{w}"] = df[rc].rolling(w, min_periods=1).sum()

    wind_points = ["porto_alegre", "rio_grande", "mostardas"]
    for pt in wind_points:
        speed_col = f"{pt}_wind_max_kmh"
        dir_col = f"{pt}_wind_dir_deg"
        if speed_col in df.columns and dir_col in df.columns:
            wind_dir_rad = np.deg2rad(df[dir_col])
            speed_ms = df[speed_col] / 3.6
            df[f"{pt}_u_wind"] = -speed_ms * np.sin(wind_dir_rad)
            df[f"{pt}_v_wind"] = -speed_ms * np.cos(wind_dir_rad)
            for w in WIND_ROLLING:
                df[f"{pt}_v_wind_roll{w}"] = df[f"{pt}_v_wind"].rolling(w, min_periods=1).mean()
                df[f"{pt}_u_wind_roll{w}"] = df[f"{pt}_u_wind"].rolling(w, min_periods=1).mean()

    v_cols = [f"{pt}_v_wind" for pt in wind_points if f"{pt}_v_wind" in df.columns]
    u_cols = [f"{pt}_u_wind" for pt in wind_points if f"{pt}_u_wind" in df.columns]
    if v_cols:
        df["v_wind_regional"] = df[v_cols].mean(axis=1)
        df["u_wind_regional"] = df[u_cols].mean(axis=1)
        for w in WIND_ROLLING:
            df[f"v_wind_regional_roll{w}"] = df["v_wind_regional"].rolling(w, min_periods=1).mean()
            df[f"u_wind_regional_roll{w}"] = df["u_wind_regional"].rolling(w, min_periods=1).mean()
        df["represamento_2d"] = (df["v_wind_regional_roll2"] > 1.0).astype(int)
        df["represamento_3d"] = (df["v_wind_regional_roll3"] > 1.0).astype(int)

    df["chuva_total_raw"] = df[rain_cols].sum(axis=1) if rain_cols else 0
    for w in [3, 7, 14, 30]:
        df[f"chuva_total_acc_{w}d"] = df["chuva_total_raw"].rolling(w, min_periods=1).sum()
    df["chuva_media_raw"] = df[rain_cols].mean(axis=1) if rain_cols else 0
    for w in [3, 7, 14]:
        df[f"chuva_media_acc_{w}d"] = df["chuva_media_raw"].rolling(w, min_periods=1).sum()

    if "guaiba_nivel_mean" in df.columns:
        df["inter_chuva_x_nivel"] = df["chuva_total_raw"] * df["guaiba_nivel_mean"]
        df["inter_chuva7d_x_nivel"] = df["chuva_total_acc_7d"] * df["guaiba_nivel_mean"]
        guaiba_delta1 = df["guaiba_nivel_mean"].diff(1)
        guaiba_delta3 = df["guaiba_nivel_mean"].diff(3)
        df["inter_chuva_x_delta1"] = df["chuva_total_raw"] * guaiba_delta1
        df["inter_chuva_x_delta3"] = df["chuva_total_raw"] * guaiba_delta3

    df["chuva_ratio_3d_30d"] = df["chuva_total_acc_3d"] / (df["chuva_total_acc_30d"] + 1e-6)
    df["chuva_ratio_7d_30d"] = df["chuva_total_acc_7d"] / (df["chuva_total_acc_30d"] + 1e-6)
    for w in [7, 14, 30]:
        df[f"dias_chuvosos_{w}d"] = df["chuva_total_raw"].rolling(w, min_periods=1).apply(lambda x: (x > 1).sum())

    if "guaiba_nivel_mean" in df.columns:
        for w in [7, 14, 30]:
            df[f"guaiba_nivel_mean_{w}d"] = df["guaiba_nivel_mean"].rolling(w, min_periods=1).mean()
        df["guaiba_acima_media30d"] = df["guaiba_nivel_mean"] - df["guaiba_nivel_mean_30d"]

    if "guaiba_nivel_mean" in df.columns:
        delta1 = df["guaiba_nivel_mean"].diff(1)
        df["guaiba_aceleracao_1d"] = delta1.diff(1)
        df["guaiba_aceleracao_3d"] = df["guaiba_nivel_mean"].diff(3).diff(3)

    for w in [3, 7, 14]:
        df[f"chuva_max_{w}d"] = df["chuva_total_raw"].rolling(w, min_periods=1).max()
        df[f"chuva_std_{w}d"] = df["chuva_total_raw"].rolling(w, min_periods=1).std()

    bacias_chuva = [c for c in ["gravatai_sl_chuva", "sinos_sl_chuva", "sinos_cb_chuva",
                                 "cai_pm_chuva", "jacui_rp_chuva", "guaiba_chuva", "catsul_chuva"]
                    if c in df.columns]
    if bacias_chuva:
        df["n_bacias_chuvosas"] = (df[bacias_chuva] > 5).sum(axis=1)
        df["n_bacias_muito_chuvosas"] = (df[bacias_chuva] > 20).sum(axis=1)

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Targets (computed on full dataset — backfills T+3/T+5 for last days)
    if "guaiba_nivel_mean" in df.columns:
        for N in FORECAST_HORIZONS:
            df[f"target_delta_{N}d"] = df["guaiba_nivel_mean"].shift(-N) - df["guaiba_nivel_mean"]

    return df.reset_index()


# ══════════════════════════════════════════════════════════════
# INCREMENTAL UPDATE
# ══════════════════════════════════════════════════════════════
def update_dataset():
    """Incremental update: read existing, fetch missing days, append, recompute targets."""
    t0 = time.time()
    today = datetime.now().date()

    # Load existing dataset or start fresh
    if DATASET_PATH.exists():
        existing = pd.read_parquet(DATASET_PATH)
        existing["date"] = pd.to_datetime(existing["date"])
        last_date = existing["date"].max().date()
        print(f"Existing dataset: {existing.shape}, last date: {last_date}")
    else:
        existing = None
        last_date = None
        print("No existing dataset found — will build from scratch.")

    # Determine date range to fetch
    if last_date is not None:
        # Start 60 days before last_date to ensure rolling features are correct
        fetch_start = last_date - timedelta(days=60)
        fetch_end = today
        print(f"Fetching: {fetch_start} to {fetch_end} (incremental + buffer)")
    else:
        fetch_start = datetime(2019, 1, 1).date()
        fetch_end = today
        print(f"Fetching: {fetch_start} to {fetch_end} (full build)")

    start_str = fetch_start.strftime("%d/%m/%Y")
    end_str = fetch_end.strftime("%d/%m/%Y")
    start_iso = fetch_start.strftime("%Y-%m-%d")
    end_iso = fetch_end.strftime("%Y-%m-%d")

    # ── STEP 1: Fetch hydro ──
    print("\n[STEP 1] Fetching hydrological data from ANA ...")
    daily_frames = {}
    for stn in ALL_HYDRO_STATIONS:
        print(f"  Station {stn}:", end=" ")
        raw = fetch_ana_station(stn, start_str, end_str)
        if raw.empty:
            print("empty")
            continue
        daily = aggregate_daily(raw)
        if daily.empty:
            print("no daily")
            continue
        print(f"{len(daily)} days ({daily['date'].min().date()} – {daily['date'].max().date()})")
        daily_frames[stn] = daily

    # Combine target stations
    print("  Building guaiba_target ...")
    if "87450004" in daily_frames:
        target = daily_frames["87450004"].copy()
        if "87444000" in daily_frames:
            fill = daily_frames["87444000"]
            merge = target.merge(fill, on="date", how="outer", suffixes=("", "_fill"))
            merge = merge.sort_values("date")
            for col in ["nivel_mean_m", "nivel_max_m", "chuva_sum_mm", "vazao_mean_m3s"]:
                merge[col] = merge[col].fillna(merge[f"{col}_fill"])
                if f"{col}_fill" in merge.columns:
                    merge.drop(columns=[f"{col}_fill"], inplace=True)
            if "n_obs_fill" in merge.columns:
                merge["n_obs"] = merge["n_obs"].fillna(merge["n_obs_fill"])
                merge.drop(columns=["n_obs_fill"], inplace=True)
            target = merge
        daily_frames["guaiba_target"] = target
        print(f"    guaiba_target: {len(target)} days")
    elif "87444000" in daily_frames:
        daily_frames["guaiba_target"] = daily_frames["87444000"].copy()

    # ── STEP 2: Fetch meteo ──
    print("\n[STEP 2] Fetching meteorological data from Open-Meteo ...")
    all_meteo = []
    for name, (lat, lon) in METEO_POINTS.items():
        print(f"  {name}:", end=" ")
        df_m = fetch_openmeteo_point(name, lat, lon, start_iso, end_iso)
        if df_m.empty:
            print("empty")
        else:
            print(f"{len(df_m)} days")
            all_meteo.append(df_m)
        time.sleep(0.3)

    if all_meteo:
        meteo_df = all_meteo[0]
        for m in all_meteo[1:]:
            meteo_df = meteo_df.merge(m, on="date", how="outer")
        meteo_df = meteo_df.sort_values("date").reset_index(drop=True)
    else:
        meteo_df = pd.DataFrame()

    # ── STEP 3: Merge base ──
    print("\n[STEP 3] Merging base dataframe ...")
    date_range = pd.date_range(start_iso, end_iso, freq="D")
    df = pd.DataFrame({"date": date_range})

    for stn_code, alias in UPSTREAM_NAMES.items():
        if stn_code in daily_frames:
            stn_df = daily_frames[stn_code][["date", "nivel_mean_m", "nivel_max_m", "chuva_sum_mm"]].copy()
            stn_df = stn_df.rename(columns={
                "nivel_mean_m": f"{alias}_nivel_mean",
                "nivel_max_m": f"{alias}_nivel_max",
                "chuva_sum_mm": f"{alias}_chuva",
            })
            df = df.merge(stn_df, on="date", how="left")

    if "guaiba_target" in daily_frames:
        tgt = daily_frames["guaiba_target"][["date", "nivel_mean_m", "nivel_max_m", "chuva_sum_mm"]].copy()
        tgt = tgt.rename(columns={
            "nivel_mean_m": "guaiba_nivel_mean",
            "nivel_max_m": "guaiba_nivel_max",
            "chuva_sum_mm": "guaiba_chuva",
        })
        df = df.merge(tgt, on="date", how="left")

    if not meteo_df.empty:
        df = df.merge(meteo_df, on="date", how="left")

    print(f"  Base shape: {df.shape}")

    # ── STEP 4: Build features ──
    print("\n[STEP 4] Building all features ...")
    df = build_all_features(df)
    print(f"  With features: {df.shape}")

    # ── STEP 5: Merge with existing ──
    if existing is not None:
        print("\n[STEP 5] Merging with existing dataset ...")
        # Remove overlapping dates from existing (recompute from buffer start)
        buffer_start = pd.Timestamp(fetch_start)
        existing_clean = existing[existing["date"] < buffer_start].copy()

        # Concatenate
        combined = pd.concat([existing_clean, df], ignore_index=True)
        combined = combined.drop_duplicates(subset="date", keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)

        # Backfill targets for last 7 days (in case T+3/T+5 were NaN)
        for N in FORECAST_HORIZONS:
            col = f"target_delta_{N}d"
            if col in combined.columns and "guaiba_nivel_mean" in combined.columns:
                combined[col] = combined["guaiba_nivel_mean"].shift(-N) - combined["guaiba_nivel_mean"]

        print(f"  Combined: {combined.shape}")
    else:
        combined = df
        print(f"  New dataset: {combined.shape}")

    # ── STEP 6: Predict with saved models ──
    print("\n[STEP 6] Computing predictions with saved models ...")
    models_dir = PROJECT_ROOT / "models"
    model_3d_path = models_dir / "model_delta_3d.pkl"
    model_5d_path = models_dir / "model_delta_5d.pkl"
    meta_path = models_dir / "model_metadata.pkl"

    if model_3d_path.exists() and model_5d_path.exists() and meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        features_3d = meta["features_3d"]
        features_5d = meta["features_5d"]

        with open(model_3d_path, "rb") as f:
            model_3d = pickle.load(f)
        with open(model_5d_path, "rb") as f:
            model_5d = pickle.load(f)

        # Only predict where features are available
        mask_3d = combined[features_3d].notna().all(axis=1)
        mask_5d = combined[features_5d].notna().all(axis=1)

        combined.loc[mask_3d, "pred_delta_3d"] = model_3d.predict(combined.loc[mask_3d, features_3d].values)
        combined.loc[mask_5d, "pred_delta_5d"] = model_5d.predict(combined.loc[mask_5d, features_5d].values)
        combined["proj_T3"] = combined["guaiba_nivel_mean"] + combined["pred_delta_3d"]
        combined["proj_T5"] = combined["guaiba_nivel_mean"] + combined["pred_delta_5d"]

        print(f"  Predictions: {mask_3d.sum()} T+3, {mask_5d.sum()} T+5")
    else:
        print("  [WARN] Models not found — skipping predictions")

    # ── STEP 7: Save ──
    combined.to_parquet(DATASET_PATH, index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Saved: {DATASET_PATH}")
    print(f"Shape: {combined.shape}")
    print(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    print(f"Features: {len([c for c in combined.columns if not c.startswith('target_') and c != 'date'])}")

    # Check target coverage
    for N in FORECAST_HORIZONS:
        col = f"target_delta_{N}d"
        if col in combined.columns:
            valid = combined[col].notna().sum()
            total = len(combined)
            print(f"  {col}: {valid}/{total} valid ({100*valid/total:.1f}%)")

    print(f"{'='*70}")
    return combined


if __name__ == "__main__":
    update_dataset()
