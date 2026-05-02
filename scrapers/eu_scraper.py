"""
Scraper EU – EU Funding & Tenders Portal (Comisión Europea).

API SEDIA pública (Search-API):
  POST https://api.tech.ec.europa.eu/search-api/prod/rest/search
       ?apiKey=SEDIA&text=*&pageSize=50&pageNumber=N

Cuerpo: multipart/form-data con campos `query` (JSON) y `languages` (JSON).

Captura "topics" de financiación europea (Horizon Europe, CERV, Erasmus+,
Creative Europe, Digital Europe, Innovation Fund, etc.) cuyo `deadlineDate`
o `plannedOpeningDate`/`startDate` cae en el rango pedido — ESTADO Open o
Forthcoming.

Status codes SEDIA:
  31094501 = Forthcoming
  31094502 = Open
  31094503 = Closed

Diseño:
  - aiohttp, paginación con concurrencia controlada y backoff.
  - Cada topic → JSON en `data/raw/eu/eu_{topic_id}.json`.
  - Cache estricta: si existe el archivo, no se descarga.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import aiohttp

from config import HEADERS, RAW_DIR, REQUEST_TIMEOUT

log = logging.getLogger(__name__)

EU_SEARCH_URL = (
    "https://api.tech.ec.europa.eu/search-api/prod/rest/search"
    "?apiKey=SEDIA&text=*"
)

RAW_EU = RAW_DIR / "eu"
RAW_EU.mkdir(parents=True, exist_ok=True)

_EU_CONCURRENCY = 3
_PAGE_SIZE = 50
_MAX_PAGES = 20            # tope defensivo (≈1000 topics por ventana)
_MAX_RETRIES = 3

# Filtramos por defecto a topics Open (31094502) y Forthcoming (31094501)
_STATUSES_DEFAULT = ["31094501", "31094502"]


def _build_query(start: date, end: date) -> dict:
    """Query SEDIA: SOLO filtrado por tipo + estado.

    El backend de búsqueda no acepta cláusulas `range` sobre los campos
    de fecha (`deadlineDate`, `plannedOpeningDate`, `startDate`) — devuelve
    HTTP 400 "Invalid Query format". Por tanto recuperamos TODOS los topics
    Open/Forthcoming y filtramos por rango de fechas en Python (`_in_window`).
    """
    return {
        "bool": {
            "must": [
                # type 1=Topic, 2=Call, 8=Tender
                {"terms": {"type": ["1", "2", "8"]}},
                {"terms": {"status": _STATUSES_DEFAULT}},
            ],
        },
    }


def _in_window(t: dict, start: date, end: date) -> bool:
    """Filtra topics cuya ventana (apertura, deadline) intersecta el rango."""
    s = start.isoformat()
    e = end.isoformat()
    opening = (t.get("fecha_publicacion") or "")[:10]
    deadline = (t.get("fecha_limite") or "")[:10]
    # Aceptamos si:
    #   - opening está dentro del rango, O
    #   - deadline está dentro del rango, O
    #   - el rango cae dentro de [opening, deadline]
    if opening and s <= opening <= e:
        return True
    if deadline and s <= deadline <= e:
        return True
    if opening and deadline and opening <= s and deadline >= e:
        return True
    # Si no tiene fechas, lo conservamos (Open/Forthcoming sin deadline definido)
    if not opening and not deadline:
        return True
    return False


async def _post_search(
    session: aiohttp.ClientSession,
    page: int,
    start: date,
    end: date,
) -> dict | None:
    """POST multipart al endpoint SEDIA."""
    query_json = json.dumps(_build_query(start, end))
    url = f"{EU_SEARCH_URL}&pageSize={_PAGE_SIZE}&pageNumber={page}"

    for attempt in range(_MAX_RETRIES):
        try:
            data = aiohttp.FormData()
            data.add_field(
                "query", query_json, content_type="application/json",
            )
            data.add_field(
                "languages", json.dumps(["en"]),
                content_type="application/json",
            )
            async with session.post(
                url,
                data=data,
                headers={**HEADERS, "Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    try:
                        return await resp.json(content_type=None)
                    except Exception:
                        return None
                if resp.status == 400:
                    body = (await resp.text())[:200]
                    log.warning("EU page=%d → HTTP 400 (%s)", page, body)
                    return None
                log.warning(
                    "EU page=%d → HTTP %s (intento %d/%d)",
                    page, resp.status, attempt + 1, _MAX_RETRIES,
                )
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning(
                "EU page=%d → %s (intento %d/%d)",
                page, e, attempt + 1, _MAX_RETRIES,
            )
        await asyncio.sleep(2 * (attempt + 1))
    return None


# ── Normalización ────────────────────────────────────────────────────────────

_STATUS_LABELS = {
    "31094501": "Forthcoming",
    "31094502": "Open",
    "31094503": "Closed",
}


def _first(v: Any) -> str:
    if isinstance(v, list):
        return str(v[0]) if v else ""
    if v is None:
        return ""
    return str(v)


def _date_only(s: str) -> str:
    if not s:
        return ""
    return s.split("T")[0]


def _normalize_topic(item: dict) -> dict:
    """Mapea un resultado de SEDIA al esquema interno estable."""
    md = item.get("metadata") or {}

    topic_id = (
        item.get("reference")
        or _first(md.get("identifier"))
        or item.get("id", "")
    )
    title = _first(md.get("title")) or item.get("summary") or item.get("title", "")
    description = (
        _first(md.get("description"))
        or item.get("content")
        or item.get("summary", "")
    )
    organism = (
        _first(md.get("frameworkProgramme"))
        or _first(md.get("callTitle"))
        or "European Commission"
    )
    deadlines = md.get("deadlineDate") or []
    if not isinstance(deadlines, list):
        deadlines = [deadlines]
    deadline_first = _date_only(_first(deadlines))
    deadline_all = ", ".join(_date_only(str(d)) for d in deadlines if d)

    opening = _date_only(
        _first(md.get("plannedOpeningDate")) or _first(md.get("startDate"))
    )
    closing = _date_only(_first(md.get("closingDate"))) or deadline_first

    budget = _first(md.get("budget")) or _first(md.get("budgetOverview"))
    types_action = _first(md.get("typesOfAction"))
    duration = _first(md.get("duration"))
    status_code = _first(md.get("status"))
    status = _STATUS_LABELS.get(status_code, status_code or "")

    url = item.get("url") or _first(md.get("url")) or (
        f"https://ec.europa.eu/info/funding-tenders/opportunities/portal/"
        f"screen/opportunities/topic-details/{topic_id}"
        if topic_id else ""
    )

    return {
        "id": f"EU-{topic_id}",
        "topic_id": topic_id,
        "titulo": title[:500],
        "descripcion": description[:5000],
        "organismo": organism,
        "fecha_publicacion": opening,
        "fecha_limite": closing,
        "deadlines_todas": deadline_all,
        "presupuesto": budget,
        "duracion": duration,
        "tipo_accion": types_action,
        "estado": status,
        "estado_codigo": status_code,
        "url": url,
        "_raw": item,
    }


def _build_text_payload(t: dict) -> str:
    lines = [
        f"EU FUNDING & TENDERS – Topic: {t.get('topic_id', '')}",
        f"Programme/Organism: {t.get('organismo', '')}",
        f"Title: {t.get('titulo', '')}",
        f"Status: {t.get('estado', '')} ({t.get('estado_codigo', '')})",
        f"Type of action: {t.get('tipo_accion', '')}",
        f"Opening date: {t.get('fecha_publicacion', '')}",
        f"Deadline: {t.get('fecha_limite', '')}",
        f"All deadlines: {t.get('deadlines_todas', '')}",
        f"Duration: {t.get('duracion', '')}",
        f"Budget overview: {t.get('presupuesto', '')}",
        "",
        "Description / Scope:",
        t.get("descripcion", ""),
        "",
        f"More info: {t.get('url', '')}",
    ]
    return "\n".join(lines)


async def scrape_eu(
    start: date | None = None,
    end: date | None = None,
    callback=None,
) -> list[str]:
    """Pipeline EU Funding & Tenders.

    Si la API cambia de esquema o falla, deja log y devuelve [] sin romper
    al resto del pipeline.
    """
    if start is None:
        start = date.today() - timedelta(days=30)
    if end is None:
        end = date.today() + timedelta(days=180)

    saved: list[str] = []
    if callback:
        callback(f"EU: descargando convocatorias {start} → {end}...")

    sem = asyncio.Semaphore(_EU_CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        first = await _post_search(session, 1, start, end)
        if not first or not isinstance(first, dict):
            log.warning("EU: la API no respondió o devolvió formato inesperado.")
            if callback:
                callback("EU: la API no respondió (omitiendo).")
            return []

        results = first.get("results") or first.get("hits") or []
        total_results = (
            first.get("totalResults") or first.get("total") or len(results)
        )
        try:
            total_results = int(total_results)
        except (ValueError, TypeError):
            total_results = len(results)

        total_pages = min(
            (total_results + _PAGE_SIZE - 1) // _PAGE_SIZE, _MAX_PAGES,
        )
        log.info(
            "EU: %d resultados estimados, %d páginas a descargar (cap %d)",
            total_results, total_pages, _MAX_PAGES,
        )
        if callback:
            callback(
                f"EU: página 1/{total_pages} (+{len(results)}, "
                f"~{total_results} totales)"
            )

        normalized = [
            _normalize_topic(it) for it in results if isinstance(it, dict)
        ]

        if total_pages > 1:
            done = 1

            async def _wrap(p: int):
                nonlocal done
                async with sem:
                    res = await _post_search(session, p, start, end)
                done += 1
                n = 0
                if isinstance(res, dict):
                    items = res.get("results") or res.get("hits") or []
                    n = len(items) if isinstance(items, list) else 0
                if callback:
                    callback(f"EU: página {done}/{total_pages} (+{n})")
                return res

            tasks = [_wrap(p) for p in range(2, total_pages + 1)]
            for chunk in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(chunk, dict):
                    items = chunk.get("results") or chunk.get("hits") or []
                    normalized.extend(
                        _normalize_topic(it)
                        for it in items if isinstance(it, dict)
                    )
                elif isinstance(chunk, Exception):
                    log.warning("EU página: %s", chunk)

    new_count = 0
    skipped_window = 0
    for t in normalized:
        if not t.get("topic_id"):
            continue
        if not _in_window(t, start, end):
            skipped_window += 1
            continue
        safe_id = "".join(
            ch if ch.isalnum() or ch in "-_" else "_"
            for ch in str(t["topic_id"])
        )[:200]
        dest = RAW_EU / f"eu_{safe_id}.json"
        if dest.exists():
            saved.append(str(dest))
            continue
        payload = {
            **{k: v for k, v in t.items() if k != "_raw"},
            "text": _build_text_payload(t),
            "_raw": t.get("_raw", {}),
        }
        try:
            dest.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved.append(str(dest))
            new_count += 1
        except Exception as e:
            log.warning("EU no se pudo escribir %s: %s", dest, e)

    log.info("EU: %d topics totales (%d nuevos)", len(saved), new_count)
    if callback:
        callback(f"EU: {len(saved)} topics ({new_count} nuevos, {skipped_window} fuera de rango)")
    return saved


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )
    asyncio.run(scrape_eu())
