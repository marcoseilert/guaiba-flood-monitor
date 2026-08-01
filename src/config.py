# -*- coding: utf-8 -*-
"""
Configuração central do projeto v2 de previsão do Guaíba.
Dados diários, fontes: ANA SOAP + Open-Meteo ERA5.
"""
from pathlib import Path

# ============================================================
# DIRETÓRIOS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
for d in [DATA_RAW, DATA_PROCESSED]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# PERÍODOS
# ============================================================
DEV_START = "2019-01-01"
DEV_END = "2025-12-31"
OOT_START = "2026-01-01"
OOT_END = "2026-07-31"

# ============================================================
# ESTAÇÕES HIDROLÓGICAS (ANA SOAP API)
# ============================================================
# Target: Guaíba (duas estações combinadas)
TARGET_STATIONS = {
    "87450004": {"name": "Cais Mauá C6", "rio": "Guaíba", "note": "Principal, parou mai/2024, reativada 2025"},
    "87444000": {"name": "Gasômetro", "rio": "Guaíba", "note": "Substituta, mai/2024 - ago/2025"},
}

# Estações a montante (features)
UPSTREAM_STATIONS = {
    # Guaíba (validação)
    "87242000": {"name": "Terminal CATSUL", "rio": "Guaíba", "bacia": "guaiba"},
    # Gravataí
    "87399000": {"name": "São Leopoldo", "rio": "Gravataí", "bacia": "gravatai"},
    # Sinos
    "87382000": {"name": "São Leopoldo", "rio": "Sinos", "bacia": "sinos"},
    "87380000": {"name": "Campo Bom", "rio": "Sinos", "bacia": "sinos"},
    # Caí
    "87270000": {"name": "Passo Montenegro", "rio": "Caí", "bacia": "cai"},
    # Jacuí
    "86510000": {"name": "Muçum", "rio": "Taquari", "bacia": "taquari"},
    "85900000": {"name": "Rio Pardo", "rio": "Jacuí", "bacia": "jacui"},
}

ALL_STATIONS = {**TARGET_STATIONS, **UPSTREAM_STATIONS}

# ============================================================
# PONTOS METEOROLÓGICOS (Open-Meteo ERA5)
# ============================================================
METEO_POINTS = {
    # Bacia Taquari
    "mucum":           {"lat": -29.16, "lon": -51.87, "bacia": "taquari"},
    "encantado":       {"lat": -29.23, "lon": -51.87, "bacia": "taquari"},
    "estrela":         {"lat": -29.50, "lon": -51.96, "bacia": "taquari"},
    "bento_goncalves": {"lat": -29.17, "lon": -51.52, "bacia": "taquari"},
    # Bacia Jacuí
    "cachoeira_do_sul": {"lat": -30.03, "lon": -52.89, "bacia": "jacui"},
    "rio_pardo":        {"lat": -29.98, "lon": -52.37, "bacia": "jacui"},
    "santa_cruz_sul":   {"lat": -29.72, "lon": -52.43, "bacia": "jacui"},
    # Bacia Caí
    "feliz":            {"lat": -29.45, "lon": -51.30, "bacia": "cai"},
    "sao_seb_cai":      {"lat": -29.59, "lon": -51.38, "bacia": "cai"},
    # Bacia Sinos
    "campo_bom":        {"lat": -29.67, "lon": -51.06, "bacia": "sinos"},
    "sao_leopoldo":     {"lat": -29.76, "lon": -51.15, "bacia": "sinos"},
    # Bacia Guaíba / Lagoa dos Patos (vento)
    "porto_alegre":     {"lat": -30.03, "lon": -51.23, "bacia": "guaiba"},
    "rio_grande":       {"lat": -32.03, "lon": -52.10, "bacia": "lagoa_patos"},
    "mostardas":        {"lat": -31.11, "lon": -50.92, "bacia": "lagoa_patos"},
    "arambare":         {"lat": -30.91, "lon": -51.50, "bacia": "lagoa_patos"},
}

BACIAS = ["taquari", "jacui", "cai", "sinos", "gravatai", "guaiba", "lagoa_patos"]

# ============================================================
# PARÂMETROS DE FEATURES
# ============================================================
RAIN_WINDOWS = [3, 6, 12, 24, 48, 72, 120]  # horas acumuladas
LEVEL_LAGS = [1, 2, 3, 5, 7]  # dias
LEVEL_DELTAS = [1, 2, 3]  # dias
WIND_WINDOWS = [1, 2, 3]  # dias
FORECAST_HORIZONS = [24, 48, 72]  # horas → convertido para dias (1, 2, 3)

# ============================================================
# COTAS DE RISCO
# ============================================================
COTA_ATENCAO = 2.50    # metros
COTA_INUNDACAO = 3.00  # metros

# ============================================================
# URLs
# ============================================================
ANA_SOAP_URL = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx"
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
