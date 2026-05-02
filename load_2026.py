"""
load_2026.py – Carga completa de datos YTD 2026 para Sub-Radar.

Ejecuta el pipeline completo desde el 1 de enero de 2026 hasta hoy:
  1. Scraping: BOE (XML API), DOG (HTML), BOP (HTML/PDF)
  2. Vectorización: ChromaDB con paraphrase-multilingual-MiniLM-L12-v2
  3. Análisis LLM: Funil 3 capas (regex → qwen2.5:3b triage → qwen2.5:14b extracción)

Maneja automáticamente:
  - Reanudación (los documentos ya descargados/analizados se omiten)
  - Bloqueos de SQLite ("database is locked") con reintentos exponenciales
  - Estadísticas finales de la operación

Uso:
    cd sub_radar && python load_2026.py [--skip-scraping] [--skip-vectorize] [--skip-llm]
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

# Añadir directorio padre al path para imports relativos
sys.path.insert(0, str(Path(__file__).parent))

from config import RAW_BOE, RAW_DOG, RAW_BOP, RAW_BDNS, RAW_EU, BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(BASE_DIR / "data" / "load_2026.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("load_2026")

# ── Rango de fechas 2026 (dinámico hasta hoy) ──────────────────────────────
DATE_START_2026 = date(2026, 1, 1)
DATE_END_2026   = date.today()   # Siempre hoy

_MESES_ES = [
    '', 'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
    'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre',
]
_TODAY_HUMAN = (
    f"{DATE_END_2026.day} de {_MESES_ES[DATE_END_2026.month]} "
    f"de {DATE_END_2026.year}"
)


def _banner(title: str) -> None:
    log.info("")
    log.info("═" * 65)
    log.info("  %s", title)
    log.info("  Rango: %s → %s", DATE_START_2026, DATE_END_2026)
    log.info("  Fecha de referencia (hoy): %s", _TODAY_HUMAN)
    log.info("═" * 65)


def _elapsed(t0: float) -> str:
    secs = int(time.time() - t0)
    return f"{secs // 3600}h {(secs % 3600) // 60}m {secs % 60}s"


# ── Utilidad SQLite retry ────────────────────────────────────────────────────
def _with_db_retry(fn, max_retries: int = 8, base_delay: float = 1.0):
    """
    Ejecuta fn() con reintentos exponenciales ante errores 'database is locked'.
    Garantiza que la BD WAL no bloquee el proceso principal.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if "locked" in msg or "database is locked" in msg:
                delay = base_delay * (2 ** attempt)
                log.warning(
                    "SQLite bloqueado (intento %d/%d). Reintentando en %.1fs...",
                    attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"SQLite sigue bloqueado después de {max_retries} reintentos")


# ── Fase 1: Scraping ─────────────────────────────────────────────────────────
async def phase_scraping(sources: list[str]) -> dict:
    from run_scrapers import run_all_scrapers

    _banner("FASE 1: SCRAPING DE BOLETINES OFICIALES 2026")
    t0 = time.time()

    def cb(msg: str) -> None:
        log.info("[Scraper] %s", msg)

    try:
        results = await run_all_scrapers(
            sources=sources,
            start=DATE_START_2026,
            end=DATE_END_2026,
            callback=cb,
        )
    except Exception as e:
        log.error("Error durante el scraping: %s", e, exc_info=True)
        results = {}

    total_files = sum(len(v) for v in results.values())
    log.info("Scraping completado en %s — %d archivos nuevos", _elapsed(t0), total_files)
    for src, files in results.items():
        log.info("  %-4s  %d archivos", src.upper(), len(files))
    return results


# ── Fase 2: Vectorización ─────────────────────────────────────────────────────
def phase_vectorize() -> int:
    from processing.text_extractor import build_document_list
    from vector_store import index_document, compute_doc_hash, get_index_stats
    from database import get_document

    _banner("FASE 2: VECTORIZACIÓN EN CHROMADB")
    t0 = time.time()

    log.info("Construyendo lista de documentos desde archivos raw...")
    docs = build_document_list(RAW_BOE, RAW_DOG, RAW_BOP, RAW_BDNS, RAW_EU)
    log.info("Total documentos en disco: %d", len(docs))

    # Filtrar solo los del rango 2026
    range_start = str(DATE_START_2026)
    range_end   = str(DATE_END_2026)
    docs_2026 = [
        d for d in docs
        if range_start <= (d.get("date_published") or "") <= range_end
    ]
    log.info("Documentos en rango 2026: %d", len(docs_2026))

    total_chunks = 0
    indexed = 0
    skipped = 0

    for i, doc in enumerate(docs_2026):
        text = doc.get("text", "")
        if not text:
            skipped += 1
            continue

        text_hash = compute_doc_hash(text)
        db_doc = _with_db_retry(lambda: get_document(doc["source"], doc["id"]))
        if db_doc and db_doc.get("text_hash") == text_hash:
            skipped += 1
            continue

        if db_doc:
            def _upd():
                from database import _get_conn
                conn = _get_conn()
                conn.execute(
                    "UPDATE documents SET text_hash = ? WHERE id = ?",
                    (text_hash, db_doc["id"]),
                )
                conn.commit()
            _with_db_retry(_upd)

        n = index_document(
            source=doc["source"],
            doc_id=doc["id"],
            titulo=doc.get("titulo", ""),
            text=text,
            filepath=doc.get("filepath", ""),
            url=doc.get("url", ""),
            date_published=doc.get("date_published", ""),
        )
        total_chunks += n
        if n > 0:
            indexed += 1

        if (i + 1) % 200 == 0:
            stats = get_index_stats()
            log.info("[%d/%d] Vectorizados: %d nuevos, %d omitidos | ChromaDB total: %d chunks",
                     i + 1, len(docs_2026), indexed, skipped, stats["total_chunks"])

    stats = get_index_stats()
    log.info(
        "Vectorización completada en %s — %d docs indexados (%d chunks nuevos), %d omitidos",
        _elapsed(t0), indexed, total_chunks, skipped,
    )
    log.info("ChromaDB total: %d chunks en BD", stats["total_chunks"])
    return total_chunks


# ── Fase 3: Análisis LLM ──────────────────────────────────────────────────────
def phase_llm() -> dict:
    from processing.text_extractor import build_document_list
    from processing.llm_processor import process_documents

    _banner("FASE 3: ANÁLISIS LLM – FUNIL 3 CAPAS")
    t0 = time.time()

    log.info("Construyendo lista de documentos...")
    docs = build_document_list(RAW_BOE, RAW_DOG, RAW_BOP, RAW_BDNS, RAW_EU)

    # Filtrar rango 2026
    range_start = str(DATE_START_2026)
    range_end   = str(DATE_END_2026)
    docs_2026 = [
        d for d in docs
        if range_start <= (d.get("date_published") or "") <= range_end
    ]
    log.info("Documentos 2026 a procesar: %d", len(docs_2026))

    # Verificar que Ollama está disponible
    import httpx
    from config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_MODEL_TRIAGE
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10.0)
        models = [m["name"] for m in r.json().get("models", [])]
        log.info("Ollama disponible. Modelos cargados: %s", ", ".join(models[:5]) or "(ninguno)")
        missing = [m for m in [OLLAMA_MODEL, OLLAMA_MODEL_TRIAGE] if not any(m in x for x in models)]
        if missing:
            log.warning(
                "ADVERTENCIA: Los siguientes modelos no están en Ollama: %s\n"
                "Ejecuta: ollama pull %s",
                ", ".join(missing), " && ollama pull ".join(missing),
            )
    except Exception as e:
        log.error("No se puede conectar a Ollama en %s: %s", OLLAMA_URL, e)
        log.error("Asegúrate de que Ollama está corriendo: systemctl start ollama")
        return {}

    stats = process_documents(docs_2026)
    elapsed = _elapsed(t0)
    log.info("Análisis LLM completado en %s", elapsed)
    if isinstance(stats, dict):
        for k, v in stats.items():
            log.info("  %-20s %s", k + ":", v)
    return stats or {}


# ── Resumen final ─────────────────────────────────────────────────────────────
def print_final_summary() -> None:
    from database import get_analysis_stats
    from vector_store import get_index_stats

    try:
        db = get_analysis_stats()
        vec = get_index_stats()

        log.info("")
        log.info("╔══════════════════════════════════════════════════════════════╗")
        log.info("║                RESUMEN FINAL – SUB-RADAR 2026               ║")
        log.info("╠══════════════════════════════════════════════════════════════╣")
        log.info("║  Documentos totales en disco:  %6d                        ║", db.get("total_docs", 0))
        log.info("║  Documentos analizados (LLM):  %6d                        ║", db.get("analyzed", 0))
        log.info("║  Subvenciones extraídas:        %6d                        ║", db.get("grants", 0))
        log.info("║  Pendientes de análisis:        %6d                        ║", db.get("pending", 0))
        log.info("║  Chunks en ChromaDB:            %6d                        ║", vec.get("total_chunks", 0))
        log.info("╚══════════════════════════════════════════════════════════════╝")
        log.info("")
        log.info("Para iniciar el servidor web:")
        log.info("  cd sub_radar && python app.py")
        log.info("  → http://localhost:5000  (admin / admin)")
    except Exception as e:
        log.warning("No se pudo obtener el resumen final: %s", e)


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Carga datos YTD 2026 en Sub-Radar (1 ene → {_TODAY_HUMAN})",
    )
    parser.add_argument("--skip-scraping",  action="store_true", help="Omitir fase de descarga")
    parser.add_argument("--skip-vectorize", action="store_true", help="Omitir vectorización ChromaDB")
    parser.add_argument("--skip-llm",       action="store_true", help="Omitir análisis LLM")
    parser.add_argument("--sources",        nargs="+", default=["BOE", "DOG", "BOP", "BDNS", "EU"],
                        choices=["BOE", "DOG", "BOP", "BDNS", "EU"], help="Fuentes a scrapear (default: todas)")
    args = parser.parse_args()

    t_total = time.time()

    # Asegurar BD inicializada
    from database import init_db
    init_db()

    log.info("")
    log.info("████████████████████████████████████████████████████████████████")
    log.info("█                                                              █")
    log.info("█       SUB-RADAR – Carga de datos YTD 2026                   █")
    log.info("█       1 ene 2026 → %-30s         █", f"{_TODAY_HUMAN}  (hoy)")
    log.info("█                                                              █")
    log.info("████████████████████████████████████████████████████████████████")

    try:
        # FASE 1: Scraping
        if not args.skip_scraping:
            asyncio.run(phase_scraping(args.sources))
        else:
            log.info("[SKIP] Fase 1: Scraping omitida por --skip-scraping")

        # FASE 2: Vectorización
        if not args.skip_vectorize:
            phase_vectorize()
        else:
            log.info("[SKIP] Fase 2: Vectorización omitida por --skip-vectorize")

        # FASE 3: Análisis LLM
        if not args.skip_llm:
            phase_llm()
        else:
            log.info("[SKIP] Fase 3: Análisis LLM omitida por --skip-llm")

    except KeyboardInterrupt:
        log.warning("Proceso interrumpido por el usuario (Ctrl+C). El estado se ha guardado.")
        log.warning("Puedes reanudar ejecutando: python load_2026.py --skip-scraping (si ya descargaste)")
    except Exception as e:
        log.error("Error inesperado: %s", e, exc_info=True)
        raise
    finally:
        print_final_summary()
        log.info("Tiempo total: %s", _elapsed(t_total))


if __name__ == "__main__":
    main()
