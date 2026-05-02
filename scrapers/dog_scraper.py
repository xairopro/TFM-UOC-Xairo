"""
Scraper DOG – Diario Oficial de Galicia.

Accede a TODAS las secciones publicadas diariamente:
  https://www.xunta.gal/dog/Publicados/{YYYY}/{YYYYMMDD}/Secciones{N}_gl.html

Cobertura completa (sin descartar nada):
  - Secciones1 (I.   Disposicións xerais)
  - Secciones2 (II.  Autoridades e persoal)
  - Secciones3 (III. Outras disposicións)   ← donde van la mayoría de subvenciones
  - Secciones4 (IV.  Oposicións e concursos)
  - Secciones5 (V.   Administración de Xustiza)
  - Secciones6 (VI.  Anuncios)

Se incluyen también sábados y domingos (DOG publica ediciones extraordinarias).
"""

import asyncio
import logging
import re
from datetime import timedelta

import aiohttp
from bs4 import BeautifulSoup

from config import (
    DATE_END, DATE_START, HEADERS, MAX_CONCURRENT,
    RAW_DOG, REQUEST_TIMEOUT,
)

log = logging.getLogger(__name__)

DOG_BASE = "https://www.xunta.gal"
# TODAS las secciones del DOG — no descartamos ninguna para no perder
# subvenciones (p.ej. Xacobeo, voluntariado de Cultura/Lingua/Xuventude…).
SECTION_IDS = [
    "Secciones1_gl.html",
    "Secciones2_gl.html",
    "Secciones3_gl.html",
    "Secciones4_gl.html",
    "Secciones5_gl.html",
    "Secciones6_gl.html",
]


def _dates_in_range(start=None, end=None):
    """Itera todos los días del rango (incluyendo fines de semana — el DOG
    publica ediciones extraordinarias en festivos)."""
    d = start or DATE_START
    end_d = end or DATE_END
    while d <= end_d:
        yield d
        d += timedelta(days=1)


async def _fetch(session: aiohttp.ClientSession, url: str,
                 binary: bool = False) -> bytes | str | None:
    for attempt in range(3):
        try:
            async with session.get(
                url,
                headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    return await resp.read() if binary else await resp.text()
                if resp.status == 404:
                    return None
                log.warning("DOG %s → HTTP %s (intento %d)", url, resp.status, attempt + 1)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("DOG %s → %s (intento %d)", url, e, attempt + 1)
        await asyncio.sleep(1 * (attempt + 1))
    return None


def _extract_anuncio_links(html: str) -> list[dict]:
    """Extrae enlaces a disposiciones individuales de una sección DOG.

    Selector permisivo: cualquier <a> que apunte a un fichero
    AnuncioXXX_gl.html. Esto evita perder subvenciones cuando el DOG
    cambia el layout (`li.dog-toc-sumario` no siempre está presente).
    """
    soup = BeautifulSoup(html, "lxml")
    links: list[dict] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "Anuncio" not in href or not href.endswith("_gl.html"):
            continue
        if href.startswith("/"):
            href = DOG_BASE + href
        elif not href.startswith("http"):
            continue
        if href in seen:
            continue
        seen.add(href)
        titulo = a.get_text(strip=True) or ""
        match = re.search(r"(Anuncio[A-Z0-9]+-\d+-\d+)", href)
        anuncio_id = (
            match.group(1)
            if match
            else href.split("/")[-1].replace("_gl.html", "")
        )
        links.append({"id": anuncio_id, "titulo": titulo, "url": href})
    return links


async def _download_doc(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                        anuncio: dict) -> str | None:
    """Descarga el HTML de una disposición DOG."""
    async with sem:
        safe_id = re.sub(r'[^\w\-]', '_', anuncio["id"])
        dest = RAW_DOG / f"{safe_id}.html"
        if dest.exists():
            return str(dest)
        html = await _fetch(session, anuncio["url"])
        if html:
            dest.write_text(html, encoding="utf-8")
            return str(dest)
    return None


async def scrape_dog(start=None, end=None, callback=None) -> list[str]:
    """
    Pipeline: por cada día del rango, descarga TODAS las secciones
    en paralelo (Secciones1..Secciones6) y luego cada disposición
    en paralelo. Sin filtros restrictivos: queremos TODO el contenido
    para no perder ninguna subvención.
    """
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    section_sem = asyncio.Semaphore(min(MAX_CONCURRENT, 8))
    all_anuncios: list[dict] = []
    files: list[str] = []

    async with aiohttp.ClientSession() as session:
        # 1. Recopilar enlaces — en paralelo por (día, sección)
        async def _grab_section(year: str, date_str: str, sec: str):
            async with section_sem:
                url = f"{DOG_BASE}/dog/Publicados/{year}/{date_str}/{sec}"
                html = await _fetch(session, url)
            if not html:
                return []
            anuncios = _extract_anuncio_links(html)
            if anuncios:
                log.info(
                    "DOG %s/%s → %d disposiciones",
                    date_str, sec, len(anuncios),
                )
            return anuncios

        tasks = []
        for d in _dates_in_range(start, end):
            date_str = d.strftime("%Y%m%d")
            year = d.strftime("%Y")
            for sec in SECTION_IDS:
                tasks.append(_grab_section(year, date_str, sec))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                all_anuncios.extend(r)
            elif isinstance(r, Exception):
                log.warning("DOG sección: %s", r)

        # Deduplicar por ID
        seen = set()
        unique = []
        for a in all_anuncios:
            if a["id"] not in seen:
                seen.add(a["id"])
                unique.append(a)
        all_anuncios = unique

        log.info("DOG total disposiciones únicas: %d", len(all_anuncios))

        if callback:
            callback(f"DOG: descargando {len(all_anuncios)} disposiciones...")

        # 2. Descargar cada disposición en paralelo
        tasks = [_download_doc(session, sem, a) for a in all_anuncios]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        files = [r for r in results if isinstance(r, str) and r]

    log.info("DOG descargados: %d archivos HTML", len(files))
    if callback:
        callback(f"DOG: {len(files)} documentos descargados")
    return files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    downloaded = asyncio.run(scrape_dog())
    print(f"DOG: {len(downloaded)} documentos descargados en {RAW_DOG}")
