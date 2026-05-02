"""
Scraper BOE – Boletín Oficial del Estado.

Usa la API de datos abiertos del BOE:
  https://www.boe.es/datosabiertos/api/boe/sumario/{YYYYMMDD}

Descarga los sumarios diarios XML y luego el HTML de cada disposición
de las secciones relevantes (I, III: leyes y subvenciones).
"""

import asyncio
import logging
from datetime import timedelta
from xml.etree import ElementTree as ET

import aiohttp

from config import (
    DATE_END, DATE_START, HEADERS, MAX_CONCURRENT,
    RAW_BOE, REQUEST_TIMEOUT,
)

log = logging.getLogger(__name__)

# Secciones relevantes para subvenciones
SECCIONES_INTERES = {"1", "3", "3A", "3B", "5A"}  # I, III, III-A/B, V-A


def _dates_in_range(start=None, end=None):
    """Genera cada día laborable (L–V) en el rango."""
    d = start or DATE_START
    end_d = end or DATE_END
    while d <= end_d:
        if d.weekday() < 5:  # Lunes a Viernes
            yield d
        d += timedelta(days=1)


async def _fetch(session: aiohttp.ClientSession, url: str) -> str | None:
    """GET con reintentos y manejo de errores."""
    for attempt in range(3):
        try:
            async with session.get(
                url,
                headers={**HEADERS, "Accept": "application/xml"},
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    return await resp.text()
                if resp.status == 404:
                    return None
                log.warning("BOE %s → HTTP %s (intento %d)", url, resp.status, attempt + 1)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("BOE %s → %s (intento %d)", url, e, attempt + 1)
        await asyncio.sleep(1 * (attempt + 1))
    return None


async def _download_html(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                         item_id: str, url_html: str) -> str | None:
    """Descarga el HTML de una disposición individual."""
    async with sem:
        dest = RAW_BOE / f"{item_id}.html"
        if dest.exists():
            return str(dest)
        html = await _fetch(session, url_html)
        if html:
            dest.write_text(html, encoding="utf-8")
            return str(dest)
    return None


def _parse_sumario(xml_text: str) -> list[dict]:
    """Extrae items con título y URL del sumario XML, filtrando por secciones relevantes."""
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    for seccion_el in root.iter("seccion"):
        # El atributo "codigo" ya viene en formato alfanumérico: "1","3","3A","3B","5A"
        sec_codigo = seccion_el.get("codigo", "").strip().upper()
        if sec_codigo not in SECCIONES_INTERES:
            continue

        for item_el in seccion_el.iter("item"):
            ident = item_el.findtext("identificador", "")
            titulo = item_el.findtext("titulo", "")
            url_html = item_el.findtext("url_html", "")
            if ident and url_html:
                items.append({
                    "id": ident,
                    "titulo": titulo,
                    "url_html": url_html,
                })
    return items


async def scrape_boe(start=None, end=None, callback=None) -> list[str]:
    """
    Pipeline completo: itera los días, descarga sumarios y luego cada
    disposición relevante en paralelo.

    Retorna la lista de rutas de archivos descargados.
    """
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    files: list[str] = []

    async with aiohttp.ClientSession() as session:
        # 1. Descargar sumarios diarios
        all_items: list[dict] = []
        for d in _dates_in_range(start, end):
            url = f"https://www.boe.es/datosabiertos/api/boe/sumario/{d.strftime('%Y%m%d')}"
            xml = await _fetch(session, url)
            if xml:
                sumario_path = RAW_BOE / f"sumario_{d.strftime('%Y%m%d')}.xml"
                sumario_path.write_text(xml, encoding="utf-8")
                items = _parse_sumario(xml)
                all_items.extend(items)
                log.info("BOE sumario %s → %d items", d, len(items))

        log.info("BOE total items a descargar: %d", len(all_items))
        if callback:
            callback(f"BOE: descargando {len(all_items)} disposiciones...")

        # 2. Descargar HTML de cada disposición en paralelo
        tasks = [
            _download_html(session, sem, it["id"], it["url_html"])
            for it in all_items
        ]
        results = await asyncio.gather(*tasks)
        files = [r for r in results if r]

    log.info("BOE descargados: %d archivos HTML", len(files))
    if callback:
        callback(f"BOE: {len(files)} documentos descargados")
    return files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    downloaded = asyncio.run(scrape_boe())
    print(f"BOE: {len(downloaded)} documentos descargados en {RAW_BOE}")
