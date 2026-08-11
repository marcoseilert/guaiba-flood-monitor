# -*- coding: utf-8 -*-
"""
Script para atualizar o dataset do Guaiba e enviar para o GitHub.
Pode ser agendado no Windows Task Scheduler para rodar automaticamente.

Uso: python auto_update.py
"""
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
LOG_FILE = PROJECT / "auto_update.log"

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

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
    
    # Step 2: Git add + commit
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
