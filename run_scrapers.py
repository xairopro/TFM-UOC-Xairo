"""
Orquestador de scrapers – ejecuta BOE, DOG y BOP en paralelo.
"""

import asyncio
import logging
import time
from datetime import date

from scrapers.boe_scraper import scrape_boe
from scrapers.dog_scraper import scrape_dog
from scrapers.bop_scraper import scrape_bop
from scrapers.bdns_scraper import scrape_bdns
from scrapers.eu_scraper import scrape_eu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("sub_radar")


async def run_all_scrapers(
    sources: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    callback=None,
) -> dict[str, list[str]]:
    """Ejecuta los scrapers seleccionados concurrentemente.

    Cada scraper tiene su propio timeout duro y el orquestador usa
    `as_completed` para notificar al callback en cuanto termina cada uno,
    evitando que un scraper lento (BDNS / EU) deje la UI congelada.
    Las excepciones se atrapan: un scraper roto NUNCA bloquea al resto.
    """
    if sources is None:
        sources = ["BOE", "DOG", "BOP", "BDNS", "EU"]

    log.info("═" * 60)
    log.info("SUB-RADAR: Iniciando descarga de boletines: %s", ", ".join(sources))
    log.info("═" * 60)
    if callback:
        callback(f"📥 Iniciando descarga de: {', '.join(sources)}")

    start_time = time.time()

    # Timeout duro por scraper (segundos). BOP es muy lento por diseño
    # (delays anti-baneo de 5-15s por anuncio).
    _TIMEOUTS = {
        "BOE":  10 * 60,
        "DOG":  10 * 60,
        "BOP":  30 * 60,
        "BDNS": 15 * 60,
        "EU":   12 * 60,
    }

    SCRAPERS = {
        "BOE":  scrape_boe,
        "DOG":  scrape_dog,
        "BOP":  scrape_bop,
        "BDNS": scrape_bdns,
        "EU":   scrape_eu,
    }

    async def _bounded(name: str):
        """Lanza un scraper con timeout y devuelve (name, files_or_None)."""
        fn = SCRAPERS[name]
        try:
            files = await asyncio.wait_for(
                fn(start, end, callback),
                timeout=_TIMEOUTS.get(name, 10 * 60),
            )
            return name, files
        except asyncio.TimeoutError:
            log.error("%s: TIMEOUT tras %ds. Cancelado.",
                      name, _TIMEOUTS.get(name, 10 * 60))
            if callback:
                callback(f"⏱️ {name}: timeout, omitiendo (continuamos con el resto).")
            return name, []
        except Exception as e:
            log.exception("%s: error: %s", name, e)
            if callback:
                callback(f"⚠️ {name}: error ({e.__class__.__name__}), omitiendo.")
            return name, []

    tasks = [
        asyncio.create_task(_bounded(s), name=f"scraper-{s}")
        for s in sources if s in SCRAPERS
    ]

    results: dict[str, list[str]] = {}
    completed = 0
    total_tasks = len(tasks)

    # Notificar conforme cada scraper termina
    for coro in asyncio.as_completed(tasks):
        name, files = await coro
        results[name.lower()] = files or []
        completed += 1
        if callback:
            callback(
                f"✅ {name}: {len(files or [])} documentos "
                f"({completed}/{total_tasks} fuentes completadas)"
            )

    elapsed = time.time() - start_time
    total = sum(len(v) for v in results.values())

    log.info("═" * 60)
    log.info("SUB-RADAR: Descarga completada en %.1f s", elapsed)
    for name, files in results.items():
        log.info("  %s: %d documentos", name.upper(), len(files))
    log.info("  TOTAL: %d documentos", total)
    log.info("═" * 60)

    if callback:
        callback(f"✅ Descarga completada: {total} documentos en {elapsed:.0f}s")

    return results


if __name__ == "__main__":
    asyncio.run(run_all_scrapers())
