"""
Scraper BDNS – Base de Datos Nacional de Subvenciones (España).

Fuente oficial (API REST pública SNPSAP):
  Listado:  GET https://www.infosubvenciones.es/bdnstrans/api/convocatorias/busqueda
            ?vpd=GE&pageSize=100&page=N
            &order=fechaRecepcion&direccion=desc
            &fechaDesde=DD/MM/YYYY&fechaHasta=DD/MM/YYYY
  Detalle:  GET https://www.infosubvenciones.es/bdnstrans/api/convocatorias?numConv=N

NOTA: la base es `/bdnstrans/api`, NO `/bdnstrans/GE/es/api/v2.1`. El segundo
path está documentado en el front Angular pero el back actual responde 400.

Cobertura: convocatorias publicadas en el rango (BOE, DOG, BOP, ayuntamientos,
diputaciones, consorcios y fondos europeos canalizados a España).

Diseño:
  - aiohttp asíncrono con concurrencia controlada y backoff exponencial.
  - Cada convocatoria → JSON en data/raw/bdns/bdns_{numConv}.json con el
    detalle completo (campos estructurados que el text_extractor convertirá
    a texto plano para el LLM).
  - Cache estricta: si el archivo existe ya, NO se vuelve a descargar.
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

BDNS_BASE = "https://www.infosubvenciones.es/bdnstrans/api"
LISTADO_URL = f"{BDNS_BASE}/convocatorias/busqueda"
DETALLE_URL = f"{BDNS_BASE}/convocatorias"

RAW_BDNS = RAW_DIR / "bdns"
RAW_BDNS.mkdir(parents=True, exist_ok=True)

# Concurrencia muy conservadora — la API rate-limita agresivamente (HTTP 429).
_LIST_CONCURRENCY = 2
_DETAIL_CONCURRENCY = 2
_PAGE_SIZE = 100
_MAX_PAGES = 50            # tope defensivo (≈5000 convocatorias por ventana)
_MAX_RETRIES = 5
_BASE_BACKOFF = 1.5
_THROTTLE_DELAY = 0.15     # pausa mínima entre peticiones por slot


def _fmt(d: date) -> str:
    return d.strftime("%d/%m/%Y")


async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
    params: dict | None = None,
) -> Any | None:
    """GET con reintentos, backoff exponencial y manejo de HTTP 429."""
    import random
    for attempt in range(_MAX_RETRIES):
        try:
            async with session.get(
                url,
                params=params,
                headers={**HEADERS, "Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    try:
                        return await resp.json(content_type=None)
                    except Exception:
                        text = await resp.text()
                        try:
                            return json.loads(text)
                        except Exception:
                            log.warning("BDNS %s → respuesta no-JSON", url)
                            return None
                if resp.status == 404:
                    return None
                if resp.status == 400:
                    body = (await resp.text())[:200]
                    log.warning("BDNS %s → HTTP 400 (%s)", url, body)
                    return None
                if resp.status == 429:
                    # Rate-limit: respeta Retry-After si lo envía, si no espera más
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        wait = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        wait = 0.0
                    if wait <= 0:
                        wait = (_BASE_BACKOFF * 4) * (2 ** attempt) + random.uniform(0, 2)
                    log.warning(
                        "BDNS %s → HTTP 429 rate-limit, esperando %.1fs (intento %d/%d)",
                        url, wait, attempt + 1, _MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    continue
                log.warning(
                    "BDNS %s → HTTP %s (intento %d/%d)",
                    url, resp.status, attempt + 1, _MAX_RETRIES,
                )
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning(
                "BDNS %s → %s (intento %d/%d)",
                url, e, attempt + 1, _MAX_RETRIES,
            )
        await asyncio.sleep(_BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1))
    return None


# ── Normalización ────────────────────────────────────────────────────────────

def _join_descriptions(items: Any) -> str:
    """Junta descripciones de listas de objetos {descripcion: ...}."""
    if not items:
        return ""
    if isinstance(items, list):
        out = []
        for it in items:
            if isinstance(it, dict):
                v = it.get("descripcion") or it.get("nombre") or ""
                if v:
                    out.append(str(v).strip())
            elif it:
                out.append(str(it).strip())
        return ", ".join(out)
    if isinstance(items, dict):
        return str(items.get("descripcion") or "")
    return str(items)


def _format_amount(v: Any) -> str:
    if v is None or v == "":
        return ""
    try:
        n = float(v)
        s = f"{n:,.2f}"
        # es-ES: separador de miles "." y decimales ","
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{s} €"
    except (TypeError, ValueError):
        return str(v)


def _normalize_listado(raw: dict) -> dict:
    """Mapea un item del listado al esquema interno mínimo."""
    num = str(raw.get("numeroConvocatoria") or raw.get("codigoBDNS") or "").strip()
    organismo = " — ".join(
        p for p in (raw.get("nivel1"), raw.get("nivel2"), raw.get("nivel3")) if p
    )
    return {
        "id_interno": raw.get("id"),
        "numero_convocatoria": num,
        "descripcion": str(raw.get("descripcion") or "").strip(),
        "organismo": organismo[:500],
        "fecha_publicacion": str(raw.get("fechaRecepcion") or "")[:10],
        "mrr": bool(raw.get("mrr", False)),
    }


def _normalize_detalle(raw: dict) -> dict:
    """Mapea el detalle completo al esquema interno serializable."""
    org = raw.get("organo") or {}
    organismo = " — ".join(
        p for p in (org.get("nivel1"), org.get("nivel2"), org.get("nivel3")) if p
    )
    num = str(raw.get("codigoBDNS") or "").strip()
    url_publica = (
        f"https://www.infosubvenciones.es/bdnstrans/GE/es/convocatoria/{num}"
        if num else ""
    )

    return {
        "id": f"BDNS-{num}" if num else f"BDNS-{raw.get('id', '')}",
        "id_interno": raw.get("id"),
        "numero_convocatoria": num,
        "organismo": organismo[:500],
        "sede_electronica": raw.get("sedeElectronica") or "",
        "fecha_publicacion": str(raw.get("fechaRecepcion") or "")[:10],
        "fecha_inicio_solicitud": str(raw.get("fechaInicioSolicitud") or "")[:10],
        "fecha_fin_solicitud": str(raw.get("fechaFinSolicitud") or "")[:10],
        "abierto": bool(raw.get("abierto", False)),
        "mrr": bool(raw.get("mrr", False)),
        "tipo_convocatoria": raw.get("tipoConvocatoria") or "",
        "descripcion": str(raw.get("descripcion") or "").strip(),
        "instrumentos": _join_descriptions(raw.get("instrumentos")),
        "tipos_beneficiarios": _join_descriptions(raw.get("tiposBeneficiarios")),
        "sectores": _join_descriptions(raw.get("sectores")),
        "regiones": _join_descriptions(raw.get("regiones")),
        "finalidad": raw.get("descripcionFinalidad") or "",
        "bases_reguladoras": raw.get("descripcionBasesReguladoras") or "",
        "url_bases_reguladoras": raw.get("urlBasesReguladoras") or "",
        "presupuesto": _format_amount(raw.get("presupuestoTotal")),
        "presupuesto_total_num": raw.get("presupuestoTotal"),
        "ayuda_estado": raw.get("ayudaEstado") or "",
        "url_ayuda_estado": raw.get("urlAyudaEstado") or "",
        "fondos": _join_descriptions(raw.get("fondos")),
        "reglamento": (raw.get("reglamento") or {}).get("descripcion", ""),
        "objetivos": _join_descriptions(raw.get("objetivos")),
        "url": url_publica,
        "_raw": raw,
    }


def _build_text_payload(c: dict) -> str:
    """Texto sintético que el text_extractor pasará al LLM."""
    abierto = "sí" if c.get("abierto") else "no"
    mrr = "sí" if c.get("mrr") else "no"
    lines = [
        f"CONVOCATORIA BDNS Nº {c.get('numero_convocatoria', '')}",
        f"Organismo: {c.get('organismo', '')}",
        f"Fecha de publicación (recepción): {c.get('fecha_publicacion', '')}",
        f"Plazo de solicitud: {c.get('fecha_inicio_solicitud', '')}"
        f" → {c.get('fecha_fin_solicitud', '')} (abierto: {abierto})",
        f"Tipo de convocatoria: {c.get('tipo_convocatoria', '')}",
        f"Presupuesto total: {c.get('presupuesto', '')}",
        f"Instrumentos: {c.get('instrumentos', '')}",
        f"Tipos de beneficiarios: {c.get('tipos_beneficiarios', '')}",
        f"Sectores: {c.get('sectores', '')}",
        f"Regiones: {c.get('regiones', '')}",
        f"Finalidad: {c.get('finalidad', '')}",
        f"Fondos: {c.get('fondos', '')}",
        f"Reglamento: {c.get('reglamento', '')}",
        f"Ayuda de Estado: {c.get('ayuda_estado', '')}",
        f"MRR (Mecanismo de Recuperación y Resiliencia): {mrr}",
        "",
        "Objeto / Descripción:",
        c.get("descripcion", ""),
        "",
        "Bases reguladoras:",
        c.get("bases_reguladoras", ""),
        f"URL bases reguladoras: {c.get('url_bases_reguladoras', '')}",
        "",
        f"Ficha pública: {c.get('url', '')}",
    ]
    return "\n".join(lines)


# ── Pipeline ─────────────────────────────────────────────────────────────────

async def _fetch_listado_page(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    page: int,
    start: date,
    end: date,
) -> tuple[list[dict], int]:
    """Devuelve (items_normalizados, total_pages_detectado)."""
    params = {
        "vpd": "GE",
        "pageSize": _PAGE_SIZE,
        "page": page,
        "order": "fechaRecepcion",
        "direccion": "desc",
        "fechaDesde": _fmt(start),
        "fechaHasta": _fmt(end),
    }
    async with sem:
        data = await _get_json(session, LISTADO_URL, params=params)
        await asyncio.sleep(_THROTTLE_DELAY)
    if not data or not isinstance(data, dict):
        return [], 0
    content = data.get("content") or []
    total_pages = int(data.get("totalPages") or 0)
    items = [_normalize_listado(it) for it in content if isinstance(it, dict)]
    return items, total_pages


async def _fetch_detalle(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    listado_item: dict,
) -> dict | None:
    """Descarga el detalle completo de una convocatoria por numConv."""
    num = listado_item.get("numero_convocatoria")
    if not num:
        return None
    async with sem:
        data = await _get_json(session, DETALLE_URL, params={"numConv": num})
        await asyncio.sleep(_THROTTLE_DELAY)
    if not data or not isinstance(data, dict):
        return None
    return _normalize_detalle(data)


async def scrape_bdns(
    start: date | None = None,
    end: date | None = None,
    callback=None,
) -> list[str]:
    """Pipeline BDNS: listado paginado → detalle → JSON persistido.

    Cache estricta: si `bdns_{num}.json` existe ya, no se vuelve a pedir
    el detalle.
    """
    if start is None:
        start = date.today() - timedelta(days=30)
    if end is None:
        end = date.today()

    saved: list[str] = []
    list_sem = asyncio.Semaphore(_LIST_CONCURRENCY)
    det_sem = asyncio.Semaphore(_DETAIL_CONCURRENCY)

    if callback:
        callback(f"BDNS: descargando convocatorias {start} → {end}...")

    async with aiohttp.ClientSession() as session:
        # Página 0 → totalPages
        first_items, total_pages = await _fetch_listado_page(
            session, list_sem, 0, start, end,
        )
        if not first_items and not total_pages:
            log.warning("BDNS: la API no devolvió datos para %s → %s", start, end)
            if callback:
                callback("BDNS: la API no respondió o sin datos en el rango.")
            return []

        total_pages = min(total_pages or 1, _MAX_PAGES)
        log.info(
            "BDNS: %d páginas detectadas (cap %d, pageSize=%d)",
            total_pages, _MAX_PAGES, _PAGE_SIZE,
        )
        if callback:
            callback(f"BDNS: página 1/{total_pages} (+{len(first_items)})")

        all_items: list[dict] = list(first_items)

        if total_pages > 1:
            done = 1

            async def _wrap(p: int):
                nonlocal done
                items, _ = await _fetch_listado_page(
                    session, list_sem, p, start, end,
                )
                done += 1
                if callback:
                    callback(
                        f"BDNS: página {done}/{total_pages} (+{len(items)})"
                    )
                return items

            results = await asyncio.gather(
                *[_wrap(p) for p in range(1, total_pages)],
                return_exceptions=True,
            )
            for chunk in results:
                if isinstance(chunk, list):
                    all_items.extend(chunk)
                elif isinstance(chunk, Exception):
                    log.warning("BDNS página: %s", chunk)

        log.info("BDNS: %d convocatorias listadas en total", len(all_items))

        # Filtrado cache
        pendientes: list[dict] = []
        cached = 0
        for it in all_items:
            num = it.get("numero_convocatoria")
            if not num:
                continue
            dest = RAW_BDNS / f"bdns_{num}.json"
            if dest.exists():
                saved.append(str(dest))
                cached += 1
            else:
                pendientes.append(it)

        if callback:
            callback(
                f"BDNS: {cached} en caché, {len(pendientes)} a descargar detalle"
            )

        if pendientes:
            done_det = 0
            total_det = len(pendientes)

            async def _det(it: dict):
                nonlocal done_det
                detalle = await _fetch_detalle(session, det_sem, it)
                done_det += 1
                if callback and (done_det % 25 == 0 or done_det == total_det):
                    callback(f"BDNS: detalles {done_det}/{total_det}")

                # Si no hay detalle, persistimos el listado mínimo
                if detalle is None:
                    detalle = {
                        "id": f"BDNS-{it.get('numero_convocatoria', '')}",
                        "numero_convocatoria": it.get("numero_convocatoria"),
                        "organismo": it.get("organismo", ""),
                        "fecha_publicacion": it.get("fecha_publicacion", ""),
                        "descripcion": it.get("descripcion", ""),
                        "url": (
                            f"https://www.infosubvenciones.es/bdnstrans/GE/es/"
                            f"convocatoria/{it.get('numero_convocatoria', '')}"
                        ),
                    }

                num = (
                    detalle.get("numero_convocatoria")
                    or it.get("numero_convocatoria")
                )
                dest = RAW_BDNS / f"bdns_{num}.json"
                payload = {
                    **{k: v for k, v in detalle.items() if k != "_raw"},
                    "text": _build_text_payload(detalle),
                    "_raw": detalle.get("_raw", {}),
                }
                try:
                    dest.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    saved.append(str(dest))
                except Exception as e:
                    log.warning("BDNS no se pudo escribir %s: %s", dest, e)

            await asyncio.gather(
                *[_det(it) for it in pendientes],
                return_exceptions=True,
            )

    new_count = max(len(saved) - cached, 0)
    log.info(
        "BDNS: %d convocatorias totales (%d nuevas, %d en caché)",
        len(saved), new_count, cached,
    )
    if callback:
        callback(
            f"BDNS: {len(saved)} convocatorias ({new_count} nuevas, "
            f"{cached} en caché)"
        )
    return saved


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )
    asyncio.run(scrape_bdns(date.today() - timedelta(days=3), date.today()))
