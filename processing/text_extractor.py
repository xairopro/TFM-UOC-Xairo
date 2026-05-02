"""
Extractor de texto – convierte los archivos raw a texto plano procesable.
También registra documentos nuevos en la base de datos.

Usa ProcessPoolExecutor para paralelizar el parsing de HTML/PDF (CPU-bound).
"""

import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from bs4 import BeautifulSoup

from database import register_document, get_document

log = logging.getLogger(__name__)

# Límite de texto para el LLM (15K chars para capturar contenido completo)
_TEXT_LIMIT = 15000

# Workers para paralelización – dejar 1 core libre para el sistema
_MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# Tamaño de chunk para procesar en lotes (evitar overhead de IPC excesivo)
_CHUNK_SIZE = 200


# ═══════════════════════════════════════════════════════════════════════════════
# Funciones puras de extracción (ejecutables en subprocesos)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_html(filepath: Path) -> str:
    """Extrae texto limpio de un HTML del BOE o DOG."""
    try:
        html = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return ""

    soup = BeautifulSoup(html, "lxml")

    # Eliminar scripts, estilos, nav, header, footer
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Colapsar líneas vacías
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text[:_TEXT_LIMIT]


def _extract_html_text_for_path(filepath_str: str) -> str:
    """Wrapper para ProcessPoolExecutor – recibe string, devuelve texto."""
    return extract_text_from_html(Path(filepath_str))


def extract_metadata_from_boe_xml(xml_path: Path) -> list[dict]:
    """Extrae metadatos de un sumario XML del BOE."""
    from xml.etree import ElementTree as ET
    items = []
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return items

    for item_el in root.iter("item"):
        ident = item_el.findtext("identificador", "")
        titulo = item_el.findtext("titulo", "")
        url_html = item_el.findtext("url_html", "")
        if titulo:
            items.append({
                "id": ident,
                "titulo": titulo,
                "url_html": url_html,
                "fuente": "BOE",
            })
    return items


def extract_text_from_bop_meta(filepath: Path) -> dict:
    """Lee un meta JSON del BOP A Coruña."""
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
        return data
    except Exception:
        return {}


def _extract_date_from_boe_id(doc_id: str) -> str:
    """Extrae la fecha de publicación del ID del BOE (ej: BOE-A-2025-25041)."""
    match = re.search(r'BOE-\w-(\d{4})', doc_id)
    if match:
        return match.group(1)
    return ""


def _extract_date_from_sumario_filename(xml_file: Path) -> str:
    """Extrae la fecha del nombre del sumario XML (ej: sumario_20250318.xml)."""
    match = re.search(r'sumario_(\d{4})(\d{2})(\d{2})', xml_file.name)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return ""


def _extract_bop_text(html_path: Path, pdf_path: Path) -> tuple[str, str]:
    """
    Extrae texto de un anuncio BOP (HTML preferido, PDF como fallback).
    Retorna (texto, filepath_usada).
    Función pura – segura para subprocesos.
    """
    # Intentar HTML
    if html_path.exists():
        try:
            html_content = html_path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html_content, "lxml")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            if text:
                return text, str(html_path)
        except Exception:
            pass

    # Fallback: PDF
    if pdf_path.exists():
        try:
            import pdfplumber
            with pdfplumber.open(str(pdf_path)) as pdf:
                pages_text = []
                for page in pdf.pages[:10]:
                    pt = page.extract_text()
                    if pt:
                        pages_text.append(pt)
                text = "\n".join(pages_text)
                if text:
                    return text, str(pdf_path)
        except Exception:
            pass

    return "", ""


def _extract_bop_text_worker(args: tuple) -> tuple[str, str, str]:
    """Worker para BOP: recibe (safe_id, html_str, pdf_str), retorna (safe_id, text, filepath)."""
    safe_id, html_str, pdf_str = args
    text, fpath = _extract_bop_text(Path(html_str), Path(pdf_str))
    return safe_id, text, fpath


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline principal de construcción de documentos (paralelizado)
# ═══════════════════════════════════════════════════════════════════════════════

def build_document_list(raw_boe: Path, raw_dog: Path, raw_bop: Path,
                       raw_bdns: Path | None = None,
                       raw_eu: Path | None = None) -> list[dict]:
    """
    Construye una lista unificada de documentos para procesar.
    Registra cada documento en la DB si es nuevo.
    Cada documento tiene: source, id, titulo, text, filepath, url, date_published.

    Usa ProcessPoolExecutor para paralelizar la extracción de texto (CPU-bound).

    Acepta opcionalmente directorios para BDNS (Base de Datos Nacional de
    Subvenciones) y EU (Funding & Tenders Portal). Para estas fuentes los
    documentos son JSONs estructurados que ya incluyen un campo `text`.
    """
    docs = []

    log.info("Construyendo lista de documentos con %d workers...", _MAX_WORKERS)

    # ── BOE: extraer metadatos de XMLs + texto de HTMLs en paralelo ──────────
    boe_items = []
    for xml_file in sorted(raw_boe.glob("sumario_*.xml")):
        date_published = _extract_date_from_sumario_filename(xml_file)
        items = extract_metadata_from_boe_xml(xml_file)
        for item in items:
            html_path = raw_boe / f"{item['id']}.html"
            if html_path.exists():
                boe_items.append((item, str(html_path), date_published))

    if boe_items:
        log.info("BOE: extrayendo texto de %d HTMLs en paralelo...", len(boe_items))
        # Extraer texto en paralelo con ProcessPoolExecutor
        html_paths = [item[1] for item in boe_items]
        boe_texts = _parallel_extract_html(html_paths)

        for (item, html_path_str, date_published), text in zip(boe_items, boe_texts):
            if text:
                url = item.get("url_html", "")
                doc = {
                    "source": "BOE",
                    "id": item["id"],
                    "titulo": item["titulo"],
                    "text": text,
                    "url": url,
                    "filepath": html_path_str,
                    "date_published": date_published,
                }
                docs.append(doc)
                register_document(
                    source="BOE", doc_id=item["id"],
                    titulo=item["titulo"],
                    filepath=html_path_str, url=url,
                    date_published=date_published,
                )

    # ── DOG: extraer texto de HTMLs en paralelo ──────────────────────────────
    dog_html_files = sorted(raw_dog.glob("*.html"))
    if dog_html_files:
        log.info("DOG: extrayendo texto de %d HTMLs en paralelo...", len(dog_html_files))
        dog_paths = [str(f) for f in dog_html_files]
        dog_texts = _parallel_extract_html(dog_paths)

        for html_file, text in zip(dog_html_files, dog_texts):
            if text:
                lines = text.split("\n")
                titulo = ""
                for line in lines[:10]:
                    if len(line) > 20 and ("orde" in line.lower() or "resolución" in line.lower()
                                           or "decreto" in line.lower() or "convocatoria" in line.lower()
                                           or "subvención" in line.lower() or "axuda" in line.lower()):
                        titulo = line[:300]
                        break
                if not titulo and lines:
                    titulo = lines[0][:300]

                date_match = re.search(r'(\d{4})(\d{2})(\d{2})', html_file.stem)
                date_published = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}" if date_match else ""

                url = f"https://www.xunta.gal/dog/Publicados/{html_file.name.replace('.html', '_gl.html')}"
                doc = {
                    "source": "DOG",
                    "id": html_file.stem,
                    "titulo": titulo,
                    "text": text,
                    "url": url,
                    "filepath": str(html_file),
                    "date_published": date_published,
                }
                docs.append(doc)
                register_document(
                    source="DOG", doc_id=html_file.stem,
                    titulo=titulo,
                    filepath=str(html_file), url=url,
                    date_published=date_published,
                )

    # ── BOP: extraer texto de HTML/PDF en paralelo ───────────────────────────
    bop_meta_files = sorted(raw_bop.glob("*_meta.json"))
    if bop_meta_files:
        # Fase 1: leer metadatos (rápido, secuencial)
        bop_metas = []
        bop_extract_args = []
        for json_file in bop_meta_files:
            data = extract_text_from_bop_meta(json_file)
            if data:
                doc_id = data.get("id", json_file.stem)
                safe_id = re.sub(r'[^\w-]', '_', str(doc_id))
                html_path = raw_bop / f"bop_{safe_id}.html"
                pdf_path = raw_bop / f"bop_{safe_id}.pdf"
                bop_metas.append((data, json_file, doc_id, safe_id))
                bop_extract_args.append((safe_id, str(html_path), str(pdf_path)))

        # Fase 2: extraer texto en paralelo
        log.info("BOP: extrayendo texto de %d documentos en paralelo...", len(bop_extract_args))
        bop_text_map = {}
        with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            for chunk_start in range(0, len(bop_extract_args), _CHUNK_SIZE):
                chunk = bop_extract_args[chunk_start:chunk_start + _CHUNK_SIZE]
                futures = {executor.submit(_extract_bop_text_worker, args): args[0]
                           for args in chunk}
                for future in as_completed(futures):
                    try:
                        safe_id, text, fpath = future.result()
                        bop_text_map[safe_id] = (text, fpath)
                    except Exception:
                        pass

        # Fase 3: construir docs con los textos extraídos
        for data, json_file, doc_id, safe_id in bop_metas:
            titulo = data.get("resumen", "")
            url = data.get("pdf_url", "") or data.get("html_url", "")
            date_published = data.get("fecha", "")

            extra_text, content_filepath = bop_text_map.get(safe_id, ("", ""))
            if not content_filepath:
                content_filepath = str(json_file)

            text_content = (
                f"Organismo: {data.get('organismo', '')}\n"
                f"Resumen: {data.get('resumen', '')}\n"
                f"Fecha: {date_published}\n"
            )
            if extra_text:
                text_content += f"\nContenido del documento:\n{extra_text}"

            doc = {
                "source": "BOP",
                "id": doc_id,
                "titulo": titulo,
                "text": text_content[:_TEXT_LIMIT],
                "url": url,
                "filepath": content_filepath,
                "date_published": date_published,
            }
            docs.append(doc)
            register_document(
                source="BOP", doc_id=doc_id,
                titulo=titulo,
                filepath=content_filepath, url=url,
                date_published=date_published,
            )

    # ── BDNS: cada JSON ya contiene el campo `text` precomputado ─────────────
    if raw_bdns and raw_bdns.exists():
        bdns_files = sorted(raw_bdns.glob("bdns_*.json"))
        if bdns_files:
            log.info("BDNS: cargando %d JSONs estructurados...", len(bdns_files))
        for jf in bdns_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            doc_id = data.get("id") or f"BDNS-{data.get('numero_convocatoria', jf.stem)}"
            titulo = (data.get("descripcion") or "")[:300] or data.get("organismo", "")
            doc = {
                "source": "BDNS",
                "id": doc_id,
                "titulo": titulo,
                "text": (data.get("text") or "")[:_TEXT_LIMIT],
                "url": data.get("url", ""),
                "filepath": str(jf),
                "date_published": data.get("fecha_publicacion", ""),
            }
            if not doc["text"]:
                continue
            docs.append(doc)
            register_document(
                source="BDNS", doc_id=doc_id,
                titulo=titulo, filepath=str(jf),
                url=data.get("url", ""),
                date_published=data.get("fecha_publicacion", ""),
            )

    # ── EU: cada JSON ya contiene el campo `text` precomputado ───────────────
    if raw_eu and raw_eu.exists():
        eu_files = sorted(raw_eu.glob("eu_*.json"))
        if eu_files:
            log.info("EU: cargando %d JSONs estructurados...", len(eu_files))
        for jf in eu_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            doc_id = data.get("id") or f"EU-{data.get('topic_id', jf.stem)}"
            titulo = (data.get("titulo") or "")[:300]
            doc = {
                "source": "EU",
                "id": doc_id,
                "titulo": titulo,
                "text": (data.get("text") or "")[:_TEXT_LIMIT],
                "url": data.get("url", ""),
                "filepath": str(jf),
                "date_published": data.get("fecha_publicacion", ""),
            }
            if not doc["text"]:
                continue
            docs.append(doc)
            register_document(
                source="EU", doc_id=doc_id,
                titulo=titulo, filepath=str(jf),
                url=data.get("url", ""),
                date_published=data.get("fecha_publicacion", ""),
            )

    log.info("Total documentos para procesar: %d (BOE: %d, DOG: %d, BOP: %d, BDNS: %d, EU: %d)",
             len(docs),
             sum(1 for d in docs if d["source"] == "BOE"),
             sum(1 for d in docs if d["source"] == "DOG"),
             sum(1 for d in docs if d["source"] == "BOP"),
             sum(1 for d in docs if d["source"] == "BDNS"),
             sum(1 for d in docs if d["source"] == "EU"))
    return docs


def _parallel_extract_html(paths: list[str]) -> list[str]:
    """
    Extrae texto de múltiples HTMLs en paralelo usando ProcessPoolExecutor.
    Procesa en chunks para controlar el uso de memoria.
    Retorna lista ordenada de textos (misma posición que paths).
    """
    results = [""] * len(paths)

    with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        for chunk_start in range(0, len(paths), _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, len(paths))
            chunk_paths = paths[chunk_start:chunk_end]

            futures = {}
            for i, path in enumerate(chunk_paths, start=chunk_start):
                futures[executor.submit(_extract_html_text_for_path, path)] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = ""

    return results
