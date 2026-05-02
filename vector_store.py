"""
Vector Store – ChromaDB para RAG semántico.

Almacena chunks de texto de los documentos descargados y permite
búsquedas por similitud semántica para el chatbot.

Usa sentence-transformers con un modelo multilingüe ligero.
"""

import hashlib
import logging
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings

from config import DATA_DIR

log = logging.getLogger(__name__)

# Directorio persistente de ChromaDB
_CHROMA_DIR = DATA_DIR / "chroma_db"
_CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Tamaño de chunk (en caracteres) y solapamiento
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 150

# Nombre de la colección
_COLLECTION_NAME = "subradargrants"

# Cliente ChromaDB persistente
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

# Modelo de embeddings (lazy load)
_embedder = None


def _get_embedder():
    """Carga el modelo de embeddings de forma lazy."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        log.info("Cargando modelo de embeddings (paraphrase-multilingual-MiniLM-L12-v2)...")
        _embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        log.info("Modelo de embeddings cargado.")
    return _embedder


def _get_client() -> chromadb.ClientAPI:
    """Obtiene el cliente ChromaDB persistente."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(_CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_collection() -> chromadb.Collection:
    """Obtiene o crea la colección de ChromaDB."""
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def compute_doc_hash(text: str) -> str:
    """Calcula un hash SHA-256 del texto del documento."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def chunk_text(text: str, chunk_size: int = _CHUNK_SIZE,
               overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """
    Divide un texto en chunks con solapamiento.
    Intenta cortar en saltos de línea o puntos para mantener coherencia.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Buscar un buen punto de corte (salto de línea o punto)
            best_cut = text.rfind("\n", start + chunk_size // 2, end)
            if best_cut == -1:
                best_cut = text.rfind(". ", start + chunk_size // 2, end)
            if best_cut > start:
                end = best_cut + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else len(text)

    return chunks


def is_document_indexed(source: str, doc_id: str) -> bool:
    """Comprueba si un documento ya tiene chunks en ChromaDB.

    Filtra por (source, doc_id) en vez de s\u00f3lo doc_id para evitar falsos
    positivos cuando un mismo identificador aparece en distintas fuentes.
    """
    collection = _get_collection()
    results = collection.get(
        where={"$and": [{"source": source}, {"doc_id": doc_id}]},
        limit=1,
    )
    return len(results["ids"]) > 0


def index_document(source: str, doc_id: str, titulo: str,
                   text: str, filepath: str = "", url: str = "",
                   date_published: str = "") -> int:
    """
    Indexa un documento en ChromaDB.
    - Divide el texto en chunks
    - Genera embeddings con sentence-transformers
    - Almacena en ChromaDB con metadatos

    Retorna el número de chunks indexados.
    """
    if not text or not text.strip():
        return 0

    # Verificar si ya está indexado
    if is_document_indexed(source, doc_id):
        return 0

    embedder = _get_embedder()
    chunks = chunk_text(text)
    if not chunks:
        return 0

    collection = _get_collection()

    # Generar embeddings
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

    # Preparar IDs y metadatos
    ids = []
    documents = []
    metadatas = []
    embs = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{source}_{doc_id}_chunk{i}"
        # Sanitizar el ID para ChromaDB
        chunk_id = re.sub(r'[^a-zA-Z0-9_-]', '_', chunk_id)[:512]

        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({
            "source": source,
            "doc_id": doc_id,
            "titulo": (titulo or "")[:500],
            "filepath": (filepath or "")[:500],
            "url": (url or "")[:500],
            "date_published": date_published or "",
            "chunk_index": i,
            "total_chunks": len(chunks),
        })
        embs.append(emb)

    # Insertar en ChromaDB (en lotes de 500 para evitar límites)
    batch_size = 500
    total = 0
    for batch_start in range(0, len(ids), batch_size):
        batch_end = batch_start + batch_size
        collection.add(
            ids=ids[batch_start:batch_end],
            documents=documents[batch_start:batch_end],
            metadatas=metadatas[batch_start:batch_end],
            embeddings=embs[batch_start:batch_end],
        )
        total += len(ids[batch_start:batch_end])

    log.debug("Indexado %s/%s: %d chunks", source, doc_id, total)
    return total


def semantic_search(query: str, n_results: int = 10,
                    source_filter: str | None = None) -> list[dict]:
    """
    Búsqueda semántica en ChromaDB.

    Retorna una lista de dicts con:
      - text: el chunk de texto
      - source, doc_id, titulo, filepath, url, date_published
      - distance: distancia coseno (menor = más relevante)
    """
    embedder = _get_embedder()
    collection = _get_collection()

    if collection.count() == 0:
        return []

    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()

    where_filter = None
    if source_filter:
        where_filter = {"source": source_filter}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results and results["documents"]:
        for docs, metas, dists in zip(
            results["documents"], results["metadatas"], results["distances"]
        ):
            for doc, meta, dist in zip(docs, metas, dists):
                output.append({
                    "text": doc,
                    "source": meta.get("source", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "titulo": meta.get("titulo", ""),
                    "filepath": meta.get("filepath", ""),
                    "url": meta.get("url", ""),
                    "date_published": meta.get("date_published", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "total_chunks": meta.get("total_chunks", 1),
                    "distance": dist,
                })

    return output


def get_index_stats() -> dict:
    """Estadísticas del índice vectorial."""
    collection = _get_collection()
    count = collection.count()
    return {
        "total_chunks": count,
        "collection_name": _COLLECTION_NAME,
        "chroma_dir": str(_CHROMA_DIR),
    }
