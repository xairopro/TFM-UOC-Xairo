"""Configuración central de Sub-Radar."""

import os
from pathlib import Path
from datetime import date, timedelta

# ── Carga opcional de .env (no requiere python-dotenv) ───────────────────────
_ENV_FILE = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_BOE = RAW_DIR / "boe"
RAW_DOG = RAW_DIR / "dog"
RAW_BOP = RAW_DIR / "bop"
RAW_BDNS = RAW_DIR / "bdns"
RAW_EU = RAW_DIR / "eu"

for d in (RAW_BOE, RAW_DOG, RAW_BOP, RAW_BDNS, RAW_EU, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Rango de fechas por defecto (prueba controlada: Dic 2025 - Ene 2026) ─────
# TEMPORAL: Para las pruebas del TFM, limitar a un periodo acotado
DATE_START = date(2025, 12, 1)
DATE_END = date(2026, 1, 31)

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")          # Modelo pesado para extracción (Capa 3)
OLLAMA_MODEL_TRIAGE = os.environ.get("OLLAMA_MODEL_TRIAGE", "qwen2.5:3b")  # Modelo lixeiro para triage SI/NON (Capa 2)

# ── Scraping ───────────────────────────────────────────────────────────────────
MAX_CONCURRENT = 10          # Descargas paralelas por fuente
REQUEST_TIMEOUT = 30         # Segundos por request
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}

# ── Tipos de destinatario ─────────────────────────────────────────────────────
ENTITY_TYPES = {
    "asociacion_cultural": "Asociación cultural sin ánimo de lucro",
    "asociacion_general": "Asociación sin ánimo de lucro (general)",
    "autonomo": "Trabajador autónomo",
    "pyme": "Pequeña y mediana empresa (PYME)",
    "pyme_tech": "PYME tecnológica / Startup",
    "empresa": "Sociedad / Empresa",
    "particular": "Particular / Persona física",
    "fundacion": "Fundación",
    "ong": "ONG",
    "cooperativa": "Cooperativa",
    "administracion": "Administración pública / Ayuntamiento",
    "club_deportivo": "Club deportivo / Entidad deportiva",
    "investigador": "Investigador / Grupo de investigación",
}

# ── Tipos de proyecto ─────────────────────────────────────────────────────────
PROJECT_TYPES = [
    "Cultura", "Sociedad", "Educación", "Industria", "Agricultura",
    "Tecnología", "Medio ambiente", "Deporte", "Turismo", "Salud",
    "Patrimonio", "Empleo", "Investigación", "Vivienda", "Igualdad",
]

# ── Fuentes disponibles ───────────────────────────────────────────────────────
SOURCES = {
    "BOE":  "Boletín Oficial del Estado",
    "DOG":  "Diario Oficial de Galicia",
    "BOP":  "Boletín Oficial de la Provincia de A Coruña",
    "BDNS": "Base de Datos Nacional de Subvenciones (Estado + CCAA + EELL)",
    "EU":   "EU Funding & Tenders Portal (Comisión Europea)",
}
