# -*- coding: utf-8 -*-
"""
heal_dataset.py — Reprocessa um período do dataset com contexto completo.

Contexto: o updater antigo calculava features numa fatia de 60 dias; as linhas
"congelavam" lags NaN ao sair do buffer (bloco observado a partir de 03/06/2026).
Este script refaz o período com contexto estendido (mesma lógica do updater
corrigido), restaurando lags/rolling e as previsões (proj_T5, prob_extremo).

Uso:
    python heal_dataset.py [AAAA-MM-DD início]     # default: 2026-06-03

Backup do parquet atual: backups/dataset_historico_pre_heal_<timestamp>.parquet
"""
import shutil
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

import update_dataset as ud

HEAL_START = (
    datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
    if len(sys.argv) > 1
    else datetime(2026, 6, 3).date()
)
CONTEXT_DAYS = ud.CONTEXT_DAYS  # 60d de contexto p/ lags e rolling features


def main():
    t0 = time.time()
    today = datetime.now().date()

    existing = pd.read_parquet(ud.DATASET_PATH)
    existing["date"] = pd.to_datetime(existing["date"])
    last_date = existing["date"].max().date()
    print(f"Dataset atual: {existing.shape} | último dia: {last_date}")

    fetch_start = HEAL_START - timedelta(days=CONTEXT_DAYS)
    fetch_end = today
    start_str = fetch_start.strftime("%d/%m/%Y")
    end_str = fetch_end.strftime("%d/%m/%Y")
    print(f"Fetch: {fetch_start} → {fetch_end} | regravando de {HEAL_START} em diante\n")

    # ── STEP 1: ANA ──
    print("[STEP 1] ANA ...")
    daily_frames = {}
    for stn in ud.ALL_HYDRO_STATIONS:
        raw = ud.fetch_ana_station(stn, start_str, end_str)
        if raw.empty:
            print(f"  {stn}: empty")
            continue
        daily = ud.aggregate_daily(raw)
        if daily.empty:
            print(f"  {stn}: no daily")
            continue
        print(f"  {stn}: {len(daily)} dias ({daily['date'].min().date()} – {daily['date'].max().date()})")
        daily_frames[stn] = daily

    if "87450004" in daily_frames:
        target = daily_frames["87450004"].copy()
        if "87444000" in daily_frames:
            fill = daily_frames["87444000"]
            merge = target.merge(fill, on="date", how="outer", suffixes=("", "_fill")).sort_values("date")
            for col in ["nivel_mean_m", "nivel_max_m", "chuva_sum_mm", "vazao_mean_m3s"]:
                merge[col] = merge[col].fillna(merge[f"{col}_fill"])
                merge.drop(columns=[f"{col}_fill"], inplace=True, errors="ignore")
            if "n_obs_fill" in merge.columns:
                merge["n_obs"] = merge["n_obs"].fillna(merge["n_obs_fill"])
                merge.drop(columns=["n_obs_fill"], inplace=True)
            target = merge
        daily_frames["guaiba_target"] = target
    elif "87444000" in daily_frames:
        daily_frames["guaiba_target"] = daily_frames["87444000"].copy()

    # ── STEP 2: Open-Meteo ──
    print("\n[STEP 2] Open-Meteo ...")
    all_meteo = []
    for name, (lat, lon) in ud.METEO_POINTS.items():
        df_m = ud.fetch_openmeteo_point(name, lat, lon, fetch_start.strftime("%Y-%m-%d"), fetch_end.strftime("%Y-%m-%d"))
        if not df_m.empty:
            all_meteo.append(df_m)
        time.sleep(0.3)
    meteo_df = all_meteo[0]
    for m in all_meteo[1:]:
        meteo_df = meteo_df.merge(m, on="date", how="outer")
    meteo_df = meteo_df.sort_values("date").reset_index(drop=True)

    # ── STEP 3: base ──
    print("\n[STEP 3] Base ...")
    date_range = pd.date_range(fetch_start.strftime("%Y-%m-%d"), fetch_end.strftime("%Y-%m-%d"), freq="D")
    df = pd.DataFrame({"date": date_range})
    for stn_code, alias in ud.UPSTREAM_NAMES.items():
        if stn_code in daily_frames:
            stn_df = daily_frames[stn_code][["date", "nivel_mean_m", "nivel_max_m", "chuva_sum_mm"]].rename(columns={
                "nivel_mean_m": f"{alias}_nivel_mean",
                "nivel_max_m": f"{alias}_nivel_max",
                "chuva_sum_mm": f"{alias}_chuva",
            })
            df = df.merge(stn_df, on="date", how="left")
    if "guaiba_target" in daily_frames:
        tgt = daily_frames["guaiba_target"][["date", "nivel_mean_m", "nivel_max_m", "chuva_sum_mm"]].rename(columns={
            "nivel_mean_m": "guaiba_nivel_mean",
            "nivel_max_m": "guaiba_nivel_max",
            "chuva_sum_mm": "guaiba_chuva",
        })
        df = df.merge(tgt, on="date", how="left")
    df = df.merge(meteo_df, on="date", how="left")

    # fallback nivelguaiba (preenche só os últimos 3 dias — inofensivo aqui)
    if "guaiba_nivel_mean" in df.columns:
        df, _ = ud.apply_nivelguaiba_fallback(df, calibration=existing)

    # TRIM: nunca além do último dia com nível válido
    if "guaiba_nivel_mean" in df.columns:
        valid_dates = df.loc[df["guaiba_nivel_mean"].notna(), "date"]
        if not valid_dates.empty:
            df = df[df["date"] <= valid_dates.max()]

    # ── STEP 4: features + poda do contexto ──
    print("\n[STEP 4] Features ...")
    df = ud.build_all_features(df)
    df = df[df["date"] >= pd.Timestamp(HEAL_START)].reset_index(drop=True)
    print(f"  regravando {len(df)} dias ({df['date'].min().date()} – {df['date'].max().date()})")

    # ── STEP 5: merge ──
    print("\n[STEP 5] Merge ...")
    existing_clean = existing[existing["date"] < pd.Timestamp(HEAL_START)].copy()
    combined = pd.concat([existing_clean, df], ignore_index=True)
    combined = combined.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
    for N in ud.FORECAST_HORIZONS:
        col = f"target_delta_{N}d"
        if col in combined.columns and "guaiba_nivel_mean" in combined.columns:
            combined[col] = combined["guaiba_nivel_mean"].shift(-N) - combined["guaiba_nivel_mean"]

    # ── STEP 6: previsões ──
    print("\n[STEP 6] Previsões ...")
    combined = ud.compute_predictions(combined)

    # ── backup + save ──
    backup_dir = ud.PROJECT_ROOT / "backups"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"dataset_historico_pre_heal_{stamp}.parquet"
    shutil.copy2(ud.DATASET_PATH, backup_path)
    combined.to_parquet(ud.DATASET_PATH, index=False)
    print(f"\nBackup: {backup_path}")
    print(f"Salvo : {ud.DATASET_PATH} | {combined.shape} | {time.time()-t0:.1f}s")

    # ── verificação ──
    w = combined.copy()
    w["date"] = pd.to_datetime(w["date"])
    w = w.set_index("date").loc[HEAL_START:]
    lag_cols = [c for c in w.columns if "_lag" in c]
    n_lag_nan = int(w[lag_cols].isna().sum().sum())
    print(f"\n[CHECK] NaN em colunas _lag no período curado: {n_lag_nan} (esperado 0)")
    print(f"[CHECK] proj_T5 NaN no período curado: {int(w['proj_T5'].isna().sum())}")
    print(f"[CHECK] prob_extremo c/ valor: {int(w['prob_extremo'].notna().sum())}/{len(w)}")


if __name__ == "__main__":
    main()
