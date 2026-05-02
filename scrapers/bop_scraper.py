"""
Scraper BOP – Boletín Oficial de la Provincia de A Coruña.

Accede a los boletines diarios:
  https://bop.dacoruna.gal/bopportal/cambioBoletin.do?fechaInput={DD}%2F{MM}%2F{YYYY}

Cada boletín contiene anuncios como bloques HTML.
Se descarga la versión HTML de cada anuncio (verAnuncio.do?idAnuncio=XXX)
para evitar los bloqueos de conexión que ocurren con la descarga masiva de PDFs.
Como fallback se intenta también el PDF.

── ANTI-BANEO ──
- Delays aleatorios altos entre peticiones (5-12s)
- Rotación agresiva de User-Agents
- Graceful degradation: tras 3 errores SSL seguidos, pausa 5 min y guarda progreso
- Sesiones cortas con force_close para evitar fingerprinting
"""

import asyncio
import json as _json
import logging
import random
import re
import ssl
from datetime import timedelta
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup

from config import (
    DATE_END, DATE_START,
    RAW_BOP, REQUEST_TIMEOUT,
)

log = logging.getLogger(__name__)

BOP_BASE = "https://bop.dacoruna.gal/bopportal"

# ── Concurrencia BOP: mínima para evitar bloqueos ────────────────────────────
_BOP_MAX_CONCURRENT = 1

# ── Delays anti-baneo extremos ───────────────────────────────────────────────
_BOP_DELAY_MIN = 5.0     # segundos mínimos entre descargas de anuncios
_BOP_DELAY_MAX = 15.0    # segundos máximos
_BOP_PAGE_DELAY_MIN = 3.0   # delay entre páginas de boletín diario
_BOP_PAGE_DELAY_MAX = 7.0

# ── Graceful degradation: pausa longa tras erros SSL consecutivos ─────────────
_SSL_COOLDOWN_SECS = 300         # 5 minutos de pausa
_SSL_CONSECUTIVE_THRESHOLD = 3   # número de erros SSL seguidos para activar pausa
_ssl_consecutive_errors = 0      # contador global (reset ao ter éxito)

# ── Archivo de progreso para reanudar tras pausas ─────────────────────────────
_PROGRESS_FILE = RAW_BOP / ".scraper_progress.json"

# ── Pool de User-Agents para rotación ─────────────────────────────────────────
_USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 OPR/114.0.0.0",
]


def _random_headers() -> dict:
    """Genera headers con User-Agent aleatorio para cada petición."""
    ua = random.choice(_USER_AGENTS)
    return {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9,gl;q=0.8,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
    }


# ── SSL: contexto relajado con ciphers legacy ─────────────────────────────────
_ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE
_ssl_ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
_ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_2


def _connector() -> aiohttp.TCPConnector:
    """Conector TCP con límite de conexiones y keep-alive corto."""
    return aiohttp.TCPConnector(
        limit=_BOP_MAX_CONCURRENT,
        limit_per_host=_BOP_MAX_CONCURRENT,
        ssl=_ssl_ctx,
        enable_cleanup_closed=True,
        force_close=True,          # no reutilizar, evita fingerprinting
    )


def _dates_in_range(start=None, end=None):
    d = start or DATE_START
    end_d = end or DATE_END
    while d <= end_d:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def _is_ssl_error(exc: Exception) -> bool:
    """Detecta si unha excepción é un erro SSL/TLS."""
    if isinstance(exc, (ssl.SSLError, aiohttp.ClientConnectorSSLError,
                        aiohttp.ClientConnectorCertificateError)):
        return True
    if isinstance(exc, aiohttp.ClientConnectorError):
        cause = exc.__cause__
        if cause and isinstance(cause, (ssl.SSLError, OSError)):
            msg = str(cause).lower()
            if "ssl" in msg or "certificate" in msg or "handshake" in msg:
                return True
    return False


def _save_progress(last_date: str, downloaded_ids: list[str]):
    """Guarda el progreso del scraper para poder reanudar tras una pausa."""
    progress = {
        "last_date": last_date,
        "downloaded_ids": downloaded_ids[-500:],  # solo los últimos 500
    }
    _PROGRESS_FILE.write_text(
        _json.dumps(progress, ensure_ascii=False), encoding="utf-8"
    )


def _load_progress() -> dict:
    """Carga el progreso guardado."""
    if _PROGRESS_FILE.exists():
        try:
            return _json.loads(_PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_date": "", "downloaded_ids": []}


def _clear_progress():
    """Limpia el archivo de progreso al completar exitosamente."""
    if _PROGRESS_FILE.exists():
        _PROGRESS_FILE.unlink()


async def _fetch(session: aiohttp.ClientSession, url: str,
                 binary: bool = False, retries: int = 4) -> bytes | str | None:
    """Descarga una URL con reintentos, backoff exponencial + jitter,
    rotación de User-Agent por request,
    e graceful degradation ante erros SSL consecutivos."""
    global _ssl_consecutive_errors

    for attempt in range(retries):
        try:
            async with session.get(
                url,
                headers=_random_headers(),
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    _ssl_consecutive_errors = 0  # reset ao ter éxito
                    return await resp.read() if binary else await resp.text()
                if resp.status == 404:
                    return None
                if resp.status == 403:
                    log.warning("BOP %s -> HTTP 403 Forbidden, esperando antes de reintentar...", url)
                    await asyncio.sleep(random.uniform(15, 30))
                else:
                    log.warning("BOP %s -> HTTP %s (intento %d)", url, resp.status, attempt + 1)
        except (aiohttp.ClientError, asyncio.TimeoutError, ssl.SSLError) as e:
            if _is_ssl_error(e):
                _ssl_consecutive_errors += 1
                log.warning("BOP %s -> erro SSL #%d consecutivo (intento %d): %s",
                            url, _ssl_consecutive_errors, attempt + 1, e)

                if _ssl_consecutive_errors >= _SSL_CONSECUTIVE_THRESHOLD:
                    log.warning(
                        "BOP: %d erros SSL consecutivos -- PAUSA de %d segundos "
                        "(graceful degradation). Guardando progreso...",
                        _ssl_consecutive_errors, _SSL_COOLDOWN_SECS,
                    )
                    await asyncio.sleep(_SSL_COOLDOWN_SECS)
                    _ssl_consecutive_errors = 0  # reset tras a pausa
            else:
                log.warning("BOP %s -> %s (intento %d)", url, e, attempt + 1)

        # Backoff exponencial con jitter alto para evitar detección de patrones
        delay = (2 ** attempt) + random.uniform(1.0, 4.0)
        await asyncio.sleep(delay)
    return None


def _parse_boletin(html: str, fecha_str: str) -> list[dict]:
    """
    Extrae anuncios del boletín diario del BOP.
    Cada bloqueAnuncio tiene ID, títulos, enlace PDF y enlace HTML.
    """
    soup = BeautifulSoup(html, "lxml")
    anuncios = []

    for bloque in soup.select("div.bloqueAnuncio"):
        anuncio_id = bloque.get("id", "")
        if not anuncio_id:
            continue

        titulos = []
        for tag in ("h2", "h3", "h4", "h5"):
            for el in bloque.select(f"{tag}.tituloEdicto"):
                titulos.append(el.get_text(strip=True))

        resumen_el = bloque.select_one("p.resumenSumario")
        resumen = ""
        pdf_href = ""
        html_href = ""
        if resumen_el:
            resumen = resumen_el.get_text(strip=True)
            pdf_link = resumen_el.select_one("a[href$='.pdf']")
            if pdf_link:
                pdf_href = pdf_link["href"]
                if not pdf_href.startswith("http"):
                    pdf_href = pdf_href.lstrip("/")
                    pdf_href = f"{BOP_BASE}/{pdf_href}"
            for a in resumen_el.select("a[href]"):
                href = a.get("href", "")
                if ".pdf" not in href and href:
                    if not href.startswith("http"):
                        href = href.lstrip("/")
                        html_href = f"{BOP_BASE}/{href}"
                    else:
                        html_href = href
                    break

        if not html_href:
            html_href = f"{BOP_BASE}/verAnuncio.do?idAnuncio={anuncio_id}"

        anuncios.append({
            "id": anuncio_id,
            "organismo": " > ".join(titulos),
            "resumen": resumen,
            "pdf_url": pdf_href,
            "html_url": html_href,
            "fecha": fecha_str,
        })

    return anuncios


async def _download_anuncio(session: aiohttp.ClientSession,
                            sem: asyncio.Semaphore,
                            anuncio: dict) -> str | None:
    """
    Descarga el contenido de un anuncio del BOP.
    Prioridad: HTML vía verAnuncio.do -> PDF como fallback.
    Delays altos entre descargas para evitar baneo.
    """
    safe_id = re.sub(r'[^\w\-]', '_', str(anuncio["id"]))

    html_dest = RAW_BOP / f"bop_{safe_id}.html"
    pdf_dest = RAW_BOP / f"bop_{safe_id}.pdf"
    if html_dest.exists():
        return str(html_dest)
    if pdf_dest.exists():
        return str(pdf_dest)

    async with sem:
        # Delay aleatorio ALTO entre descargas (5-12 segundos)
        await asyncio.sleep(random.uniform(_BOP_DELAY_MIN, _BOP_DELAY_MAX))

        # 1) Intentar HTML
        html_url = anuncio.get("html_url", "")
        if html_url:
            html_content = await _fetch(session, html_url)
            if html_content and len(html_content) > 200:
                html_dest.write_text(html_content, encoding="utf-8")
                return str(html_dest)

        # 2) Fallback: intentar PDF (con delay extra)
        pdf_url = anuncio.get("pdf_url", "")
        if pdf_url:
            await asyncio.sleep(random.uniform(3.0, 6.0))
            pdf_data = await _fetch(session, pdf_url, binary=True)
            if pdf_data and isinstance(pdf_data, bytes) and len(pdf_data) > 500:
                pdf_dest.write_bytes(pdf_data)
                return str(pdf_dest)

    return None


async def scrape_bop(start=None, end=None, callback=None) -> list[str]:
    """
    Pipeline BOP: itera días laborables, descarga la página del boletín,
    extrae anuncios y descarga el HTML de cada uno (con bajo paralelismo).

    Implementa guardado de progreso para reanudar tras pausas por SSL.
    """
    sem = asyncio.Semaphore(_BOP_MAX_CONCURRENT)
    all_anuncios: list[dict] = []
    files: list[str] = []
    downloaded_ids: list[str] = []

    # Cargar progreso previo (IDs ya descargados en sesión anterior)
    progress = _load_progress()
    already_downloaded = set(progress.get("downloaded_ids", []))
    if already_downloaded:
        log.info("BOP: retomando con %d IDs ya descargados de sesión anterior",
                 len(already_downloaded))

    connector = _connector()
    async with aiohttp.ClientSession(connector=connector) as session:
        # 1. Recopilar anuncios de cada día
        for d in _dates_in_range(start, end):
            fecha_input = f"{d.strftime('%d')}%2F{d.strftime('%m')}%2F{d.strftime('%Y')}"
            url = f"{BOP_BASE}/cambioBoletin.do?fechaInput={fecha_input}"
            html = await _fetch(session, url)
            if html:
                anuncios = _parse_boletin(html, d.strftime("%Y-%m-%d"))
                all_anuncios.extend(anuncios)
                if anuncios:
                    log.info("BOP %s -> %d anuncios", d, len(anuncios))
            # Delay alto entre páginas de boletín para evitar detección
            await asyncio.sleep(random.uniform(_BOP_PAGE_DELAY_MIN, _BOP_PAGE_DELAY_MAX))

        log.info("BOP total anuncios: %d (con PDF: %d)",
                 len(all_anuncios),
                 sum(1 for a in all_anuncios if a.get("pdf_url")))

        # 2. Guardar metadatos JSON de cada anuncio
        for a in all_anuncios:
            meta_path = RAW_BOP / f"bop_{a['id']}_meta.json"
            if not meta_path.exists():
                meta_path.write_text(_json.dumps(a, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

        # 3. Filtrar anuncios ya descargados
        pending = [a for a in all_anuncios if str(a["id"]) not in already_downloaded]
        log.info("BOP: %d anuncios pendientes de descarga (%d ya descargados)",
                 len(pending), len(all_anuncios) - len(pending))

        # 4. Descargar HTML/PDF uno a uno (secuencial para máxima cautela)
        if callback:
            callback(f"BOP: descargando {len(pending)} anuncios (HTML, modo cauteloso)...")

        for i, anuncio in enumerate(pending):
            result = await _download_anuncio(session, sem, anuncio)
            if result:
                files.append(result)
                downloaded_ids.append(str(anuncio["id"]))

            # Guardar progreso cada 10 descargas
            if (i + 1) % 10 == 0:
                _save_progress(
                    last_date=anuncio.get("fecha", ""),
                    downloaded_ids=list(already_downloaded) + downloaded_ids,
                )

            if callback and (i + 1) % 20 == 0:
                callback(f"BOP: {len(files)} documentos descargados de {len(pending)}...")

    # Limpiar progreso al completar exitosamente
    _clear_progress()

    total_files = len(files) + len(already_downloaded)
    log.info("BOP descargados: %d nuevos (%d total con sesiones previas)", len(files), total_files)
    if callback:
        callback(f"BOP: {len(files)} documentos descargados")
    return files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    downloaded = asyncio.run(scrape_bop())
    print(f"BOP: {len(downloaded)} documentos descargados en {RAW_BOP}")
