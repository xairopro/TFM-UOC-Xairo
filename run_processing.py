"""
Pipeline de procesamiento LLM – Lee archivos raw e xera subvencions.
Integra vectorización en ChromaDB para RAG semántico.
"""

import logging

from config import RAW_BOE, RAW_DOG, RAW_BOP, RAW_BDNS, RAW_EU
from processing.text_extractor import build_document_list
from processing.llm_processor import process_documents
from vector_store import index_document, get_index_stats, compute_doc_hash
from database import register_document, get_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("sub_radar")


def vectorize_documents(docs: list[dict]) -> int:
    """
    Indexa los documentos en ChromaDB para búsquedas semánticas.
    Usa el hash del texto para evitar re-procesar documentos ya indexados.
    Retorna el número de chunks nuevos indexados.
    """
    total_chunks = 0
    indexed = 0
    skipped = 0

    for doc in docs:
        text = doc.get("text", "")
        if not text:
            skipped += 1
            continue

        # Guardar hash en la BD SQLite para caché
        text_hash = compute_doc_hash(text)
        db_doc = get_document(doc["source"], doc["id"])
        if db_doc and db_doc.get("text_hash") == text_hash:
            skipped += 1
            continue

        # Actualizar hash en la BD
        if db_doc:
            from database import _get_conn
            conn = _get_conn()
            conn.execute(
                "UPDATE documents SET text_hash = ? WHERE id = ?",
                (text_hash, db_doc["id"]),
            )
            conn.commit()

        # Indexar en ChromaDB
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

    log.info("Vectorización: %d documentos indexados (%d chunks), %d omitidos (ya existentes)",
             indexed, total_chunks, skipped)
    return total_chunks


def main():
    log.info("=" * 60)
    log.info("SUB-RADAR: Pipeline de procesamiento LLM + Vectorización")
    log.info("=" * 60)

    # 1. Construir lista de documentos desde archivos raw
    docs = build_document_list(RAW_BOE, RAW_DOG, RAW_BOP, RAW_BDNS, RAW_EU)

    if not docs:
        log.error("No se encontraron documentos en data/raw/. Ejecuta primero run_scrapers.py")
        return

    # 2. Vectorizar documentos en ChromaDB (para RAG del chatbot)
    log.info("--- Fase de Vectorización (ChromaDB) ---")
    vectorize_documents(docs)
    stats = get_index_stats()
    log.info("ChromaDB: %d chunks totales en la colección", stats["total_chunks"])

    # 3. Procesar con LLM (funil de 3 capas)
    log.info("--- Fase de Análisis LLM ---")
    aptas = process_documents(docs)

    log.info("=" * 60)
    log.info("RESULTADO: %d subvenciones aptas", len(aptas))
    log.info("ChromaDB: %d chunks en el índice vectorial", get_index_stats()["total_chunks"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
