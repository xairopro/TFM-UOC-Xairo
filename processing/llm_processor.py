"""
Pipeline LLM – Procesamiento de documentos con Ollama local.

Funil de 3 capas para optimización extrema:
  Capa 1 (Regex/Keywords): Descarte inmediato por título e keywords (paralelo CPU)
  Capa 2 (Triage LLM lixeiro): Clasificación binaria SI/NON con modelo 3B (paralelo I/O)
  Capa 3 (Extracción LLM pesado): Análisis detallado con Few-Shot + modelo 14B (secuencial GPU)
"""

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date as _date
from pathlib import Path

import httpx

from config import (
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_MODEL_TRIAGE,
    PROCESSED_DIR, ENTITY_TYPES, BASE_DIR,
)
from database import (
    get_document, save_analysis, get_cached_grants,
    get_analysis_stats, register_document, count_analyzed,
    mark_date_analyzed, save_grant,
)

log = logging.getLogger(__name__)

# Fecha de referencia para que los LLM razonen con plazos correctamente
_MESES_ES = [
    '', 'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
    'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'
]
_today = _date.today()
_TODAY_STR = f"{_today.day} de {_MESES_ES[_today.month]} de {_today.year}"
_DATE_AWARENESS = (
    f"IMPORTANTE – CONCIENCIA TEMPORAL: Hoy es {_TODAY_STR}. "
    "El año en curso es 2026. "
    "Al evaluar plazos y fechas límite, asumir siempre que 2026 ES EL AÑO ACTUAL. "
    "Una fecha límite en 2026 puede estar vigente o próxima; una en 2025 ya ha expirado."
)

# Workers para paralelización CPU (Capa 1 pre-filtro)
_CPU_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# Workers para paralelización I/O (Capa 2 triage LLM)
# Limitado: o modelo lixeiro comparte GPU, pero as chamadas HTTP son I/O-bound
# e Ollama internamente enfileira requests, así que podemos enviar varias á vez
_TRIAGE_WORKERS = 4

# Tamaño de chunks para procesamento en lotes
_CHUNK_SIZE = 500

# ═══════════════════════════════════════════════════════════════════════════════
# CAPA 1: Pre-filtro por palabras clave (Regex/Keywords)
# ═══════════════════════════════════════════════════════════════════════════════

_KW_POSITIVE = [
    "subvención", "subvenciones", "subvenci", "ayuda pública",
    "convocatoria de ayudas", "bases reguladoras", "programa de ayudas",
    "programa de subvenciones", "línea de ayuda", "línea de subvención",
    "axuda", "axudas", "subvención", "subvencións",
    "convocatoria de axudas", "beca", "becas", "premio", "premios",
    "incentivo", "incentivos", "bonificación", "deducción",
    "fomento", "financiación pública", "crédito",
    "concurrencia competitiva", "solicitudes",
]

_KW_EXCLUDE_TITLE = [
    # Nomeamentos e cesamentos
    "nombramiento", "nombra", "cese ", "cesamento", "destino",
    "toma de posesión", "xubilación", "jubilación",
    # Xudicial / Edictos
    "sentencia", "auto del tribunal", "edicto", "notificación procesal",
    "recurso contencioso", "recurso de casación", "citación",
    "requerimiento", "embargo", "ejecutoria",
    # Expropiación / Multas
    "expropiación", "expropiaci", "multa", "sanción", "sancion",
    "infracción", "infraccion", "expediente sancionador",
    # Licitacións e adxudicacións (contratación pública, NON subvenciones)
    "licitación", "licitacion", "adjudicación", "adjudicacion",
    "contrato de servicio", "contrato de obra", "contrato de suministro",
    "pliego de cláusulas", "mesa de contratación",
    # Resolucións xenéricas non-subvención
    "resolución de recursos", "resolución de reclamación",
    # Persoal e oposicións
    "oposición", "oposicion", "concurso de traslado", "concurso-oposición",
    "bolsa de empleo", "bolsa de trabajo", "lista de admitidos",
    "tribunal calificador", "relación de aprobados",
    # Urbanismo e planeamento
    "plan parcial", "plan general", "estudio de detalle",
    "aprobación definitiva del plan", "deslinde",
    # Galego
    "nomeamento", "cesamento", "expropiación", "multa",
    "edicto", "licitación", "adxudicación",
]


def pre_filter(doc: dict) -> bool:
    """Capa 1: Filtro rápido por keywords – descarta documentos claramente irrelevantes.

    BDNS y EU son fuentes ya curadas (todas son subvenciones por definición),
    así que se les concede pase automático a la Capa 2/3 sin filtro de keywords.
    """
    if doc.get("source") in ("BDNS", "EU"):
        return True
    text = (doc.get("titulo", "") + " " + doc.get("text", "")[:2000]).lower()
    if not any(kw in text for kw in _KW_POSITIVE):
        return False
    titulo = doc.get("titulo", "").lower()
    if any(kw in titulo for kw in _KW_EXCLUDE_TITLE):
        return False
    return True


def _pre_filter_chunk(chunk: list[dict]) -> list[dict]:
    """Filtra un chunk de documentos. Función pura para ProcessPoolExecutor."""
    return [d for d in chunk if pre_filter(d)]


def _parallel_pre_filter(docs: list[dict]) -> list[dict]:
    """
    Capa 1 paralelizada: distribúe o pre-filtro regex en múltiples procesos.
    Para listas pequenas (<1000), executa secuencialmente para evitar overhead.
    """
    if len(docs) < 1000:
        return [d for d in docs if pre_filter(d)]

    chunks = [docs[i:i + _CHUNK_SIZE] for i in range(0, len(docs), _CHUNK_SIZE)]
    results = []

    with ProcessPoolExecutor(max_workers=_CPU_WORKERS) as executor:
        futures = [executor.submit(_pre_filter_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as e:
                log.warning("Erro no pre-filtro paralelo: %s", e)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CAPA 2: Triage con LLM lixeiro (clasificación binaria SI/NON)
# ═══════════════════════════════════════════════════════════════════════════════

_TRIAGE_SYSTEM = (
    "Eres un clasificador de documentos de boletines oficiales. "
    "Tu ÚNICA tarea es determinar si el texto describe una CONVOCATORIA "
    "de subvención, ayuda económica, beca o premio público a la que "
    "alguien pueda SOLICITAR dinero. "
    "Responde ÚNICAMENTE con la palabra SI o la palabra NON. "
    "Sin explicaciones, sin puntuación, sin nada más. "
    + _DATE_AWARENESS
)


def _triage_llm(text: str, doc_id: str) -> bool | None:
    """
    Capa 2: Pregunta ao modelo lixeiro se o doc é subvención.
    Retorna True (SI), False (NON), ou None (erro).
    """
    # Enviar só os primeiros 1500 chars (suficiente para clasificar)
    snippet = text[:1500]
    payload = {
        "model": OLLAMA_MODEL_TRIAGE,
        "messages": [
            {"role": "system", "content": _TRIAGE_SYSTEM},
            {"role": "user", "content": snippet},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 8,      # Só necesitamos "SI" ou "NON"
            "num_ctx": 2048,
        },
    }
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat", json=payload, timeout=30.0,
        )
        resp.raise_for_status()
        answer = resp.json().get("message", {}).get("content", "").strip().upper()
        # Aceptar variantes: SI, SÍ, YES → True; NON, NO → False
        if answer.startswith("SI") or answer.startswith("SÍ") or answer.startswith("YES"):
            return True
        if answer.startswith("NO"):
            return False
        log.debug("Triage resposta inesperada para %s: '%s'", doc_id, answer)
        return True  # En caso de dúbida, pasar á Capa 3
    except Exception as e:
        log.warning("Triage erro para %s: %s", doc_id, e)
        return None  # Erro → pasar á Capa 3 igualmente


def _triage_single_doc(doc: dict) -> tuple[dict, bool | None]:
    """Wrapper para ThreadPoolExecutor: triage dun doc, retorna (doc, resultado)."""
    triage_text = f"Título: {doc.get('titulo', 'Sin título')}\n\n"
    triage_text += doc.get("text", "")
    result = _triage_llm(triage_text, doc["id"])
    return doc, result


# ═══════════════════════════════════════════════════════════════════════════════
# CAPA 3: Extracción detallada con LLM pesado + Few-Shot Prompting
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_EXTRACTION = f"""Eres un experto jurídico español especializado en subvenciones, becas, premios y ayudas públicas.
Tu tarea es analizar textos de boletines oficiales (BOE, DOG, BOP) y extraer TODA la información relevante de convocatorias de subvención.

{_DATE_AWARENESS}

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE con un objeto JSON válido. Sin texto adicional.
2. Si el texto NO es sobre una subvención, beca, premio o ayuda pública, devuelve: {{"es_subvencion": false}}
3. Si SÍ es una subvención/ayuda/beca/premio, extrae TODA la información con el máximo detalle.
4. Para fechas, usa formato YYYY-MM-DD cuando sea posible.
5. Sigue EXACTAMENTE la estructura JSON que se muestra en los ejemplos.
6. Para el campo "vigente": true si la fecha_limite es posterior a hoy ({_TODAY_STR}), false si ya expiró.
7. Para el campo "comunidad_autonoma" usa EXACTAMENTE uno de estos valores: "Galicia", "Asturias", "Cantabria", "País Vasco", "Navarra", "La Rioja", "Aragón", "Cataluña", "Comunidad Valenciana", "Murcia", "Andalucía", "Extremadura", "Castilla-La Mancha", "Castilla y León", "Madrid", "Canarias", "Baleares", "Ceuta", "Melilla", "Nacional" (cuando aplique a toda España o sea ministerio nacional), "Unión Europea" (cuando sea Comisión Europea o programa UE), o "Otra/Desconocida" si no es claro.

── EJEMPLO 1 (Subvención vigente) ──────────────────────────────────
ENTRADA:
"RESOLUCIÓN de 15 de enero de 2026 de la Consellería de Cultura, Educación, Formación Profesional e Universidades, por la que se establecen las bases reguladoras y se convocan subvenciones para el fomento de actividades culturales de las asociaciones vecinales y culturales sin ánimo de lucro de Galicia para el año 2026. El plazo de presentación de solicitudes será de un mes contado a partir del día siguiente al de la publicación de esta resolución en el Diario Oficial de Galicia. La cuantía global máxima asciende a 500.000 euros."

SALIDA ESPERADA:
{{
  "es_subvencion": true,
  "titulo": "Subvenciones para el fomento de actividades culturales de asociaciones vecinales y culturales sin ánimo de lucro de Galicia 2026",
  "organismo": "Consellería de Cultura, Educación, Formación Profesional e Universidades (Xunta de Galicia)",
  "fecha_publicacion": "2026-01-15",
  "fecha_limite": "2026-02-15",
  "plazo": "Un mes desde la publicación en el DOG",
  "financiacion": "500.000 euros (cuantía global máxima)",
  "destinatarios": "Asociaciones vecinales y culturales sin ánimo de lucro de Galicia",
  "requisitos": "Ser asociación vecinal o cultural sin ánimo de lucro con sede en Galicia. Estar inscrita en el registro de asociaciones.",
  "resumen": "Convocatoria de subvenciones destinadas a fomentar las actividades culturales realizadas por asociaciones vecinales y culturales sin ánimo de lucro en Galicia durante 2026.",
  "tipo_proyecto": "Cultura",
  "bases_reguladoras": "Resolución de 15 de enero de 2026 de la Consellería de Cultura",
  "documentacion": "",
  "contacto": "",
  "ambito_geografico": "Autonómico (Galicia)",
  "comunidad_autonoma": "Galicia",
  "vigente": true,
  "link": ""
}}
── EJEMPLO 2 (Documento que NO es subvención) ──────────────────────
ENTRADA:
"RESOLUCIÓN de 3 de febrero de 2026, de la Dirección General de la Policía, por la que se convoca proceso selectivo para ingreso en la Escala Básica del Cuerpo Nacional de Policía. Los aspirantes deberán superar las pruebas físicas establecidas en la convocatoria."

SALIDA ESPERADA:
{{"es_subvencion": false}}
── EJEMPLO 3 (Subvención con plazo ya expirado) ────────────────────
ENTRADA:
"ORDEN HFP/123/2025, de 10 de octubre, por la que se convocan subvenciones para digitalización de pymes del sector industrial. Plazo de presentación de solicitudes: del 1 al 31 de octubre de 2025. Dotación total: 2.000.000 euros."

SALIDA ESPERADA:
{{
  "es_subvencion": true,
  "titulo": "Subvenciones para digitalización de pymes del sector industrial",
  "organismo": "Ministerio de Hacienda y Función Pública",
  "fecha_publicacion": "2025-10-10",
  "fecha_limite": "2025-10-31",
  "plazo": "Del 1 al 31 de octubre de 2025",
  "financiacion": "2.000.000 euros",
  "destinatarios": "Pymes del sector industrial",
  "requisitos": "",
  "resumen": "Convocatoria de subvenciones para digitalización de pymes industriales.",
  "tipo_proyecto": "Tecnología",
  "bases_reguladoras": "ORDEN HFP/123/2025",
  "documentacion": "",
  "contacto": "",
  "ambito_geografico": "Nacional",
  "comunidad_autonoma": "Nacional",
  "vigente": false,
  "link": ""
}}
── FIN EJEMPLOS ─────────────────────────────────────────────────────

Ahora analiza el texto que te proporcione el usuario y genera el JSON con la misma estructura.
EXTRAE EL MÁXIMO DE DETALLE. Es mejor incluir información de más que de menos.
Si algún campo no se menciona en el texto, pon cadena vacía.
Recuerda: hoy es {_TODAY_STR}. Evalúa la vigencia correctamente."""


def _call_ollama(text: str, doc_id: str) -> dict | None:
    """Capa 3: Extracción completa co modelo pesado."""
    text_trimmed = text[:10000]
    user_prompt = (
        "Analiza el siguiente texto de un boletín oficial y extrae "
        "TODA la información sobre subvenciones, ayudas, becas o premios en JSON. "
        "Sé lo más DETALLADO posible:\n\n" + text_trimmed
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACTION},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 2048,
            "num_ctx": 16384,
        },
    }

    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=180.0,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        return json.loads(content)
    except json.JSONDecodeError as e:
        log.warning("LLM respuesta no-JSON para %s: %s", doc_id, e)
    except httpx.HTTPError as e:
        log.warning("Error HTTP Ollama para %s: %s", doc_id, e)
    except Exception as e:
        log.warning("Error procesando %s: %s", doc_id, e)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Filtro por perfil de usuario
# ═══════════════════════════════════════════════════════════════════════════════

_ENTITY_KEYWORDS = {
    "asociacion_cultural": ["asociaci", "entidad sin ánimo", "entidades sin ánimo",
                            "personas jurídicas", "cultural", "sin ánimo de lucro",
                            "sen ánimo de lucro"],
    "asociacion_general": ["asociaci", "entidad sin ánimo", "entidades sin ánimo",
                           "personas jurídicas", "sin ánimo de lucro", "sen ánimo de lucro"],
    "autonomo": ["autónom", "persona física", "personas físicas",
                 "trabajador", "profesional"],
    "pyme": ["pyme", "pequeña", "mediana", "empresa", "microempresa",
             "persona jurídica"],
    "pyme_tech": ["pyme", "empresa", "tecnológi", "innovación", "startup",
                  "digital", "i+d", "emprendedor", "base tecnológica"],
    "empresa": ["empresa", "sociedad", "persona jurídica", "mercantil"],
    "particular": ["persona física", "personas físicas", "particular",
                   "ciudadan"],
    "fundacion": ["fundaci", "entidad sin ánimo", "sin ánimo de lucro"],
    "ong": ["ong", "organización no gubernamental", "entidad sin ánimo",
            "sin ánimo de lucro", "cooperación"],
    "cooperativa": ["cooperativ", "economía social"],
    "administracion": ["administraci", "entidad local", "ayuntamiento",
                       "diputaci", "concello", "municipio"],
    "club_deportivo": ["club deportivo", "entidad deportiva", "federación deportiva",
                       "deporte", "deportiv"],
    "investigador": ["investigador", "investigación", "universidad", "grupo de investigación",
                     "i+d", "científic", "académic"],
}


def _score_entity(grant_data: dict, entity_type: str) -> int:
    """
    Puntúa la compatibilidad de la subvención con el tipo de entidad (0-40).
    Reglas y pesos dinámicos para matching avanzado.
    """
    destinatarios = (grant_data.get("destinatarios", "") or "").lower()
    requisitos = (grant_data.get("requisitos", "") or "").lower()
    combined = destinatarios + " " + requisitos

    if not combined.strip():
        return 20  # Sin info → puntuación neutra (no descartar ni premiar)

    keywords = _ENTITY_KEYWORDS.get(entity_type, [])
    if not keywords:
        return 20

    matches = sum(1 for kw in keywords if kw in combined)
    if matches == 0:
        return 0
    # Escalar: 1 match = 15, 2 = 25, 3+ = 40
    if matches >= 3:
        return 40
    if matches >= 2:
        return 25
    return 15


def _score_project(grant_data: dict, project_types: list[str]) -> int:
    """
    Puntúa la compatibilidad con los tipos de proyecto del usuario (0-40).
    """
    if not project_types:
        return 40  # Sin filtro de proyecto → puntuación máxima

    tipo = (grant_data.get("tipo_proyecto", "") or "").lower()
    resumen = (grant_data.get("resumen", "") or "").lower()
    titulo = (grant_data.get("titulo", "") or "").lower()
    combined = tipo + " " + resumen + " " + titulo

    matches = sum(1 for pt in project_types if pt.lower() in combined)
    if matches == 0:
        return 0
    if matches >= 2:
        return 40
    return 25


def _score_deadline(grant_data: dict) -> int:
    """
    Puntúa la urgencia por proximidad de fecha límite (0-20).
    Más puntos si el plazo está próximo (incentiva subvenciones útiles ahora).
    """
    from datetime import date as dt_date

    fecha_limite = (grant_data.get("fecha_limite", "") or "").strip()
    if not fecha_limite:
        return 10  # Sin fecha → puntuación neutra

    try:
        deadline = dt_date.fromisoformat(fecha_limite)
    except (ValueError, TypeError):
        return 10

    today = dt_date.today()
    days_left = (deadline - today).days

    if days_left < 0:
        return 0   # Plazo vencido
    if days_left <= 15:
        return 20  # Muy urgente
    if days_left <= 30:
        return 18  # Urgente
    if days_left <= 60:
        return 15  # Próximo
    if days_left <= 90:
        return 12  # Medio plazo
    return 8       # Plazo lejano


def compute_match_score(grant_data: dict, entity_type: str,
                        project_types: list[str] | None = None) -> int:
    """
    Calcula el score total de matching (0-100) para una subvención vs un perfil.
    Componentes:
      - Entidad (0-40): compatibilidad del destinatario
      - Proyecto (0-40): compatibilidad del tipo de proyecto
      - Urgencia (0-20): proximidad de la fecha límite
    """
    s_entity = _score_entity(grant_data, entity_type)
    s_project = _score_project(grant_data, project_types or [])
    s_deadline = _score_deadline(grant_data)
    return s_entity + s_project + s_deadline


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline principal (funil de 3 capas)
# ═══════════════════════════════════════════════════════════════════════════════

def process_documents(docs: list[dict],
                      entity_type: str = "asociacion_cultural",
                      project_types: list[str] | None = None,
                      callback=None) -> list[dict]:
    """
    Pipeline con funil de 3 capas + cache:
      Capa 1: Pre-filtro regex/keywords
      Cache:  Reutilizar análisis previos da BD
      Capa 2: Triage con LLM lixeiro (SI/NON)
      Capa 3: Extracción detallada con LLM pesado + Few-Shot
      Filtro: Perfil de usuario (entity_type, project_types)
    """
    log.info("=" * 60)
    log.info("PIPELINE LLM (funil de 3 capas)")
    log.info("  CPU workers (Capa 1): %d", _CPU_WORKERS)
    log.info("  I/O workers (Capa 2): %d", _TRIAGE_WORKERS)
    log.info("=" * 60)

    # ── Capa 1: Pre-filtro por keywords (paralelo CPU) ─────────────────────
    candidates = _parallel_pre_filter(docs)
    discarded_kw = len(docs) - len(candidates)
    log.info("Capa 1 (Keywords): %d/%d pasan (%d descartados)",
             len(candidates), len(docs), discarded_kw)
    if callback:
        callback(
            f"[Capa 1] Keywords: {len(candidates)}/{len(docs)} pasan "
            f"({discarded_kw} descartados)"
        )

    # ── Cache: separar analizados vs novos ────────────────────────────────
    cached_grants = []
    to_triage = []

    for doc in candidates:
        db_doc = get_document(doc["source"], doc["id"])
        if db_doc:
            from database import _get_conn
            conn = _get_conn()
            row = conn.execute(
                """SELECT da.es_subvencion, da.grant_data FROM document_analysis da
                   WHERE da.document_id = ?""",
                (db_doc["id"],)
            ).fetchone()
            if row:
                if row["es_subvencion"]:
                    try:
                        gdata = json.loads(row["grant_data"])
                        gdata["source"] = doc["source"]
                        gdata["doc_id"] = doc["id"]
                        gdata["filepath"] = doc.get("filepath", "")
                        if not gdata.get("link"):
                            gdata["link"] = doc.get("url", "")
                        cached_grants.append(gdata)
                    except (json.JSONDecodeError, TypeError):
                        pass
                continue  # xa analizado

        to_triage.append(doc)

    log.info("Cache: %d xa analizados (%d subvencions), %d novos para triage",
             len(candidates) - len(to_triage), len(cached_grants), len(to_triage))
    if callback:
        callback(
            f"[Cache] {len(candidates) - len(to_triage)} xa analizados, "
            f"{len(to_triage)} novos para triage"
        )

    # ── Capa 2: Triage con LLM lixeiro (paralelo I/O) ───────────────────
    to_extract = []
    triage_rejected = 0

    if to_triage:
        log.info("Capa 2 (Triage %s): clasificando %d documentos con %d threads...",
                 OLLAMA_MODEL_TRIAGE, len(to_triage), _TRIAGE_WORKERS)
        if callback:
            callback(
                f"[Capa 2] Triage con {OLLAMA_MODEL_TRIAGE}: "
                f"clasificando {len(to_triage)} documentos ({_TRIAGE_WORKERS} threads)..."
            )

        processed_count = 0
        with ThreadPoolExecutor(max_workers=_TRIAGE_WORKERS) as executor:
            futures = {executor.submit(_triage_single_doc, doc): doc
                       for doc in to_triage}

            for future in as_completed(futures):
                try:
                    doc, result = future.result()
                except Exception as e:
                    log.warning("Triage thread erro: %s", e)
                    continue

                processed_count += 1

                if result is False:
                    # NON → rexistrar na BD como non-subvención e saltar
                    triage_rejected += 1
                    db_doc = get_document(doc["source"], doc["id"])
                    if not db_doc:
                        doc_db_id = register_document(
                            source=doc["source"],
                            doc_id=doc["id"],
                            titulo=doc.get("titulo", ""),
                            filepath=doc.get("filepath", ""),
                            url=doc.get("url", ""),
                            date_published=doc.get("date_published", ""),
                        )
                    else:
                        doc_db_id = db_doc["id"]
                    if doc_db_id:
                        save_analysis(doc_db_id, False, {})
                    # Marcar fecha
                    date_pub = doc.get("date_published", "")
                    if date_pub:
                        mark_date_analyzed(doc["source"], date_pub)
                else:
                    # SI ou erro → pasar á Capa 3
                    to_extract.append(doc)

                if processed_count % 20 == 0:
                    log.info("Triage: %d/%d (descartados: %d)",
                             processed_count, len(to_triage), triage_rejected)

        log.info("Capa 2 (Triage): %d pasan, %d descartados",
                 len(to_extract), triage_rejected)
        if callback:
            callback(
                f"[Capa 2] Triage: {len(to_extract)} pasan, "
                f"{triage_rejected} descartados"
            )

    # ── Capa 3: Extracción detallada con LLM pesado ──────────────────────
    new_grants = []
    start = time.time()

    if to_extract:
        log.info("Capa 3 (Extracción %s): procesando %d documentos...",
                 OLLAMA_MODEL, len(to_extract))
        if callback:
            callback(
                f"[Capa 3] Extracción con {OLLAMA_MODEL}: "
                f"procesando {len(to_extract)} documentos..."
            )

    for i, doc in enumerate(to_extract):
        text_for_llm = f"Título: {doc.get('titulo', 'Sin título')}\n"
        text_for_llm += f"Fuente: {doc.get('source', '')}\n"
        if doc.get("date_published"):
            text_for_llm += f"Fecha de publicación: {doc['date_published']}\n"
        text_for_llm += f"\n{doc.get('text', '')}"

        llm_result = _call_ollama(text_for_llm, doc["id"])

        # Registrar documento en DB si no existe
        db_doc = get_document(doc["source"], doc["id"])
        if not db_doc:
            doc_db_id = register_document(
                source=doc["source"],
                doc_id=doc["id"],
                titulo=doc.get("titulo", ""),
                filepath=doc.get("filepath", ""),
                url=doc.get("url", ""),
                date_published=doc.get("date_published", ""),
            )
        else:
            doc_db_id = db_doc["id"]

        # Guardar análisis en cache
        if doc_db_id and llm_result:
            es_sub = bool(llm_result.get("es_subvencion"))
            save_analysis(doc_db_id, es_sub, llm_result if es_sub else {})

            if es_sub:
                llm_result["source"] = doc["source"]
                llm_result["doc_id"] = doc["id"]
                llm_result["filepath"] = doc.get("filepath", "")
                if not llm_result.get("link"):
                    llm_result["link"] = doc.get("url", "")
                if not llm_result.get("fecha_publicacion"):
                    llm_result["fecha_publicacion"] = doc.get("date_published", "")

                # Calcular ruta relativa para el filepath
                filepath_abs = doc.get("filepath", "")
                try:
                    filepath_rel = str(Path(filepath_abs).relative_to(BASE_DIR))
                except (ValueError, TypeError):
                    filepath_rel = filepath_abs

                # Guardar en tabla normalizada de grants
                save_grant(
                    document_id=doc_db_id,
                    source=doc["source"],
                    grant_data=llm_result,
                    filepath=filepath_rel,
                    link=llm_result.get("link", doc.get("url", "")),
                )
                new_grants.append(llm_result)
        elif doc_db_id:
            # LLM falló → guardar como no-subvención para no reintentar
            save_analysis(doc_db_id, False, {})

        # Marcar fecha del documento como analizada
        date_pub = doc.get("date_published", "")
        if date_pub:
            mark_date_analyzed(doc["source"], date_pub)

        if (i + 1) % 5 == 0 or (i + 1) == len(to_extract):
            elapsed = time.time() - start
            rate = (i + 1) / max(elapsed, 0.1)
            if callback:
                callback(
                    f"[Capa 3] Extracción: {i+1}/{len(to_extract)} "
                    f"({len(new_grants)} subvencions encontradas)"
                )
            log.info("Capa 3: %d/%d (%.1f docs/s) – %d subvencions",
                     i + 1, len(to_extract), rate, len(new_grants))

    elapsed = time.time() - start
    log.info("Capa 3 procesou %d en %.0fs → %d subvencions",
             len(to_extract), elapsed, len(new_grants))

    # ── Resultados ────────────────────────────────────────────────────────
    all_grants = cached_grants + new_grants
    log.info("Total subvencions: %d (%d cache + %d novas)",
             len(all_grants), len(cached_grants), len(new_grants))

    # Resumo do funil
    log.info("--- RESUMO DO FUNIL ---")
    log.info("  Documentos totais:          %d", len(docs))
    log.info("  Capa 1 (Keywords) pasaron:  %d", len(candidates))
    log.info("  Cache (xa analizados):      %d", len(candidates) - len(to_triage))
    log.info("  Capa 2 (Triage) pasaron:    %d", len(to_extract))
    log.info("  Capa 2 (Triage) descartou:  %d", triage_rejected)
    log.info("  Capa 3 (Extracción) saíron: %d subvencions", len(new_grants))
    log.info("-----------------------")

    # Filtrar y puntuar por perfil de usuario (scoring 0-100)
    SCORE_THRESHOLD = 15  # Puntuación mínima para considerar apta
    scored = []
    for g in all_grants:
        score = compute_match_score(g, entity_type, project_types)
        g["match_score"] = score
        if score >= SCORE_THRESHOLD:
            scored.append(g)

    # Ordenar por score descendente
    scored.sort(key=lambda g: g["match_score"], reverse=True)

    entity_desc = ENTITY_TYPES.get(entity_type, entity_type)
    for g in scored:
        g["apto_solicitante"] = True
        g["motivo_aptitud"] = (
            f"Score {g['match_score']}/100 – "
            f"Destinatarios compatibles con {entity_desc}"
        )
        g["consejos"] = g.get("consejos", "Revisa los requisitos completos en el documento original")

    log.info("Subvencions aptas para %s (score >= %d): %d/%d",
             entity_type, SCORE_THRESHOLD, len(scored), len(all_grants))
    if callback:
        callback(
            f"Analise completada: {len(scored)} subvencions aptas "
            f"de {len(all_grants)} identificadas "
            f"(funil: {len(docs)} -> {len(candidates)} -> {len(to_extract)} -> {len(new_grants)})"
        )

    # Guardar resultados a JSON
    output_all = PROCESSED_DIR / "todas_subvenciones.json"
    output_aptas = PROCESSED_DIR / "subvenciones_aptas.json"

    output_all.write_text(
        json.dumps(all_grants, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    output_aptas.write_text(
        json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log.info("Gardado: %s (%d), %s (%d)",
             output_all, len(all_grants), output_aptas, len(scored))

    return scored
