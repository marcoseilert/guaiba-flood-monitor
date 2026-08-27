# -*- coding: utf-8 -*-
"""
Script para atualizar o dataset do Guaiba e enviar para o GitHub.
Pode ser agendado no Windows Task Scheduler para rodar automaticamente.

Uso: python auto_update.py
"""
import subprocess
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

PROJECT = Path(__file__).resolve().parent
LOG_FILE = PROJECT / "auto_update.log"
DATASET_FILE = PROJECT / "data" / "processed" / "dataset_historico.parquet"
NTFY_STATE_FILE = PROJECT / ".ntfy_alert_state.json"
NTFY_SERVER = os.getenv("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "alertas_mosoeilert").strip()
NTFY_THRESHOLD_M = float(os.getenv("NTFY_THRESHOLD_M", "2.5"))
NTFY_ALERT_AFTER_HOUR = 17
BRAZIL_TZ = ZoneInfo("America/Sao_Paulo")

# Validação anti-push-de-NaN: nível de hoje existe E é plausível, OU dataset
# termina ontem (ANA pode levar ~36h; trim garante que nunca vai além do válido).
VALIDATE_LEVEL_COL = "guaiba_nivel_mean"
VALIDATE_MIN_M = 0.05
VALIDATE_MAX_M = 10.0

def validate_dataset_fresh(dataset_path=DATASET_FILE, now=None):
    """Confere frescor do dataset antes do push.

    OK se: linha de hoje com nível plausível, OU última data = ontem
    (ANA pode levar ~36h p/ publicar; trim no update_dataset garante que o
    parquet nunca termina em dia vazio). Bloqueia caso contrário — evita
    reproduzir o bug do 26/08/2026 (linha NaN commitada às 18h).
    """
    now_brt = now or datetime.now(BRAZIL_TZ)
    today = pd.Timestamp(now_brt.date())
    yesterday = today - pd.Timedelta(days=1)
    try:
        df = pd.read_parquet(dataset_path, columns=["date", VALIDATE_LEVEL_COL])
    except Exception as exc:
        return False, f"dataset ilegivel: {exc}"
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    last_date = df["date"].max()

    if last_date >= today:
        row = df.loc[df["date"] == today]
        lvl = pd.to_numeric(row.iloc[0][VALIDATE_LEVEL_COL], errors="coerce")
        if pd.isna(lvl):
            return False, f"{VALIDATE_LEVEL_COL} de hoje e NaN"
        if not (VALIDATE_MIN_M <= lvl <= VALIDATE_MAX_M):
            return False, f"{VALIDATE_LEVEL_COL} de hoje fora de faixa plausivel: {lvl:.2f}m"
        return True, f"nivel de hoje OK: {lvl:.2f}m"

    if last_date == yesterday:
        return True, "dataset termina ontem — ANA ainda nao publicou hoje; push permitido"

    return False, (
        f"dataset atrasado: ultima data {last_date:%d/%m/%Y} "
        f"(esperado hoje ou ontem)"
    )

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_alert_state(state_path=NTFY_STATE_FILE):
    if not state_path.exists():
        return {}
    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError) as exc:
        log(f"AVISO: estado do alerta inválido; será recriado ({exc})")
        return {}


def _write_alert_state(state_path, state):
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def send_projection_alert(dataset_path=DATASET_FILE, state_path=NTFY_STATE_FILE, now=None):
    """Envia alerta ntfy se a projeção T+5 mais recente superar o limite."""
    now_brt = now or datetime.now(BRAZIL_TZ)
    if now_brt.hour <= NTFY_ALERT_AFTER_HOUR:
        log(f"Alerta ntfy ignorado: horário ainda não passou das 17h BRT ({now_brt:%H:%M})")
        return True

    alert_date = now_brt.date().isoformat()
    state = _read_alert_state(state_path)
    if state.get("last_alert_date") == alert_date:
        log(f"Alerta ntfy já enviado hoje ({alert_date}); sem duplicata")
        return True

    try:
        df = pd.read_parquet(dataset_path, columns=["date", "proj_T5"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["proj_T5"] = pd.to_numeric(df["proj_T5"], errors="coerce")
        valid = df.dropna(subset=["date", "proj_T5"]).sort_values("date")
        if valid.empty:
            log("AVISO: dataset não contém projeção T+5 válida")
            return False
        row = valid.iloc[-1]
        projected_m = float(row["proj_T5"])
        projection_date = row["date"].strftime("%d/%m/%Y")
        log(f"Projeção T+5 mais recente: {projected_m:.2f}m (limite: {NTFY_THRESHOLD_M:.2f}m)")
    except Exception as exc:
        log(f"AVISO: não foi possível ler projeção T+5 para alerta: {exc}")
        return False

    if projected_m <= NTFY_THRESHOLD_M:
        return True
    if not NTFY_TOPIC:
        log("AVISO: NTFY_TOPIC vazio; alerta não enviado")
        return False

    lines = [
        "🌊 ALERTA DE PROJEÇÃO DO GUAÍBA",
        f"Nível projetado em 5 dias: {projected_m:.2f} m",
        f"Data da projeção: {projection_date}",
        ]
    lines.append(f"Limite de alerta: {NTFY_THRESHOLD_M:.2f} m")

    try:
        response = requests.post(
            f"{NTFY_SERVER}/{NTFY_TOPIC}",
            data="\n".join(lines).encode("utf-8"),
            headers={
                "Title": "Alerta Guaiba - projecao T+5",
                "Priority": "high",
                "Tags": "warning,water_wave",
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        log(f"AVISO: falha ao enviar alerta ntfy: {exc}")
        return False

    _write_alert_state(state_path, {"last_alert_date": alert_date})
    log(f"Alerta ntfy enviado: projeção T+5 = {projected_m:.2f}m ({projection_date})")
    return True

def run():
    log("=== Iniciando atualizacao ===")
    
    # Step 1: Run update_dataset.py
    log("Rodando update_dataset.py...")
    try:
        result = subprocess.run(
            [sys.executable, str(PROJECT / "update_dataset.py")],
            cwd=str(PROJECT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            log(f"ERRO no update_dataset.py: {result.stderr[:500]}")
            return False
        log("update_dataset.py concluido")
    except subprocess.TimeoutExpired:
        log("ERRO: timeout (300s)")
        return False

    # Step 2: validar dataset ANTES de commitar (bloqueia push com NaN de hoje)
    log("Validando nivel de hoje no dataset...")
    ok, msg = validate_dataset_fresh()
    if not ok:
        log(f"BLOQUEADO: {msg} — commit/push cancelados")
        return False
    log(f"Validacao OK: {msg}")

    # Step 3: verificar alerta após recalcular o dataset.
    log("Verificando alerta ntfy da projeção T+5...")
    send_projection_alert()

    # Step 4: Git add + commit
    log("Commitando...")
    add_result = subprocess.run(
        ["git", "add", "data/processed/dataset_historico.parquet", "models/binary_model.pkl"],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
    )
    if add_result.returncode != 0:
        log(f"ERRO no git add: {add_result.stderr[:500]}")
        return False
    result = subprocess.run(
        ["git", "commit", "-m", f"data: auto-update {datetime.now().strftime('%d/%m/%Y %H:%M')}"],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "nothing to commit" not in result.stdout and "nothing to commit" not in result.stderr:
        log(f"ERRO no commit: {result.stderr[:500]}")
        return False
    if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
        log("Nada para commitar")
    else:
        log(f"Commit: {result.stdout.strip()}")
    
    # Step 3: Git push
    log("Enviando para GitHub...")
    result = subprocess.run(
        ["git", "push"],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"ERRO no push: {result.stderr[:300]}")
        # Try pull --rebase and push again
        log("Tentando pull --rebase...")
        subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=str(PROJECT))
        result = subprocess.run(["git", "push"], cwd=str(PROJECT), capture_output=True, text=True)
        if result.returncode != 0:
            log(f"ERRO persistente no push: {result.stderr[:300]}")
            return False

    log("=== Atualizacao concluida ===")
    return True

if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
