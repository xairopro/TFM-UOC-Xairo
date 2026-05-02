"""
Derivación de Comunidad Autónoma para una subvención.

La CCAA se infiere combinando:
  1. La fuente (DOG → Galicia, BOP A Coruña → Galicia, EU → Unión Europea)
  2. El campo `comunidad_autonoma` que devuelve el LLM (si lo extrajo)
  3. El `ambito_geografico` libre del LLM
  4. El `organismo` y, en BDNS, los campos `regiones` y `organismo`
     (que muchas veces incluyen la provincia o la diputación)

Devuelve uno de los valores en `CCAA_LIST`. Si no se logra inferir,
devuelve "Otra/Desconocida" para que el usuario pueda filtrarla aparte.
"""

from __future__ import annotations

import re
import unicodedata

# ── Listado canónico ─────────────────────────────────────────────────────────
CCAA_LIST: list[str] = [
    "Galicia",
    "Asturias",
    "Cantabria",
    "País Vasco",
    "Navarra",
    "La Rioja",
    "Aragón",
    "Cataluña",
    "Comunidad Valenciana",
    "Murcia",
    "Andalucía",
    "Extremadura",
    "Castilla-La Mancha",
    "Castilla y León",
    "Madrid",
    "Canarias",
    "Baleares",
    "Ceuta",
    "Melilla",
    "Nacional",
    "Unión Europea",
    "Otra/Desconocida",
]

# ── Provincia → CCAA ─────────────────────────────────────────────────────────
PROVINCIA_CCAA: dict[str, str] = {
    # Galicia
    "a coruna": "Galicia", "coruna": "Galicia", "lugo": "Galicia",
    "ourense": "Galicia", "orense": "Galicia", "pontevedra": "Galicia",
    # Asturias / Cantabria / La Rioja / Navarra
    "asturias": "Asturias", "cantabria": "Cantabria",
    "la rioja": "La Rioja", "rioja": "La Rioja", "navarra": "Navarra",
    # País Vasco
    "alava": "País Vasco", "araba": "País Vasco", "vizcaya": "País Vasco",
    "bizkaia": "País Vasco", "guipuzcoa": "País Vasco", "gipuzkoa": "País Vasco",
    # Aragón
    "huesca": "Aragón", "teruel": "Aragón", "zaragoza": "Aragón",
    # Cataluña
    "barcelona": "Cataluña", "girona": "Cataluña", "gerona": "Cataluña",
    "lleida": "Cataluña", "lerida": "Cataluña", "tarragona": "Cataluña",
    # Comunidad Valenciana
    "valencia": "Comunidad Valenciana", "alicante": "Comunidad Valenciana",
    "alacant": "Comunidad Valenciana", "castellon": "Comunidad Valenciana",
    "castello": "Comunidad Valenciana",
    # Murcia
    "murcia": "Murcia",
    # Andalucía
    "almeria": "Andalucía", "cadiz": "Andalucía", "cordoba": "Andalucía",
    "granada": "Andalucía", "huelva": "Andalucía", "jaen": "Andalucía",
    "malaga": "Andalucía", "sevilla": "Andalucía",
    # Extremadura
    "badajoz": "Extremadura", "caceres": "Extremadura",
    # Castilla-La Mancha
    "albacete": "Castilla-La Mancha", "ciudad real": "Castilla-La Mancha",
    "cuenca": "Castilla-La Mancha", "guadalajara": "Castilla-La Mancha",
    "toledo": "Castilla-La Mancha",
    # Castilla y León
    "avila": "Castilla y León", "burgos": "Castilla y León",
    "leon": "Castilla y León", "palencia": "Castilla y León",
    "salamanca": "Castilla y León", "segovia": "Castilla y León",
    "soria": "Castilla y León", "valladolid": "Castilla y León",
    "zamora": "Castilla y León",
    # Madrid
    "madrid": "Madrid",
    # Canarias
    "las palmas": "Canarias", "santa cruz de tenerife": "Canarias",
    "tenerife": "Canarias", "gran canaria": "Canarias", "canarias": "Canarias",
    # Baleares
    "baleares": "Baleares", "balears": "Baleares", "mallorca": "Baleares",
    "menorca": "Baleares", "ibiza": "Baleares", "eivissa": "Baleares",
    "formentera": "Baleares",
    # Ceuta / Melilla
    "ceuta": "Ceuta", "melilla": "Melilla",
    # ── Capitales y ciudades grandes que NO coinciden con su provincia ─
    # (necesarias para mapear "Ayuntamiento de X" cuando X != provincia)
    "oviedo": "Asturias", "gijon": "Asturias", "aviles": "Asturias",
    "langreo": "Asturias", "siero": "Asturias", "mieres": "Asturias",
    "santander": "Cantabria", "torrelavega": "Cantabria", "camargo": "Cantabria",
    "logrono": "La Rioja", "calahorra": "La Rioja",
    "pamplona": "Navarra", "iruna": "Navarra", "tudela": "Navarra",
    "vitoria": "País Vasco", "gasteiz": "País Vasco",
    "bilbao": "País Vasco", "barakaldo": "País Vasco", "getxo": "País Vasco",
    "san sebastian": "País Vasco", "donostia": "País Vasco",
    "irun": "País Vasco", "eibar": "País Vasco",
    "merida": "Extremadura", "plasencia": "Extremadura",
    "santiago": "Galicia", "santiago de compostela": "Galicia",
    "vigo": "Galicia", "ferrol": "Galicia",
    "santa cruz": "Canarias", "la laguna": "Canarias",
    "palma": "Baleares", "palma de mallorca": "Baleares",
    "marbella": "Andalucía", "jerez": "Andalucía",
    "lleida": "Cataluña", "manresa": "Cataluña", "sabadell": "Cataluña",
    "terrassa": "Cataluña", "badalona": "Cataluña", "hospitalet": "Cataluña",
    "elche": "Comunidad Valenciana", "elx": "Comunidad Valenciana",
    "torrent": "Comunidad Valenciana", "gandia": "Comunidad Valenciana",
    "alcorcon": "Madrid", "leganes": "Madrid", "getafe": "Madrid",
    "fuenlabrada": "Madrid", "alcala de henares": "Madrid",
    "mostoles": "Madrid", "parla": "Madrid", "torrejon": "Madrid",
    "valladolid": "Castilla y León",
    "talavera": "Castilla-La Mancha",
    "cartagena": "Murcia", "lorca": "Murcia",
}

# ── Patrones de CCAA (formas largas y abreviaturas) ──────────────────────────
CCAA_PATTERNS: list[tuple[str, str]] = [
    ("galicia", "Galicia"),
    ("xunta", "Galicia"),
    ("asturias", "Asturias"), ("principado de asturias", "Asturias"),
    ("cantabria", "Cantabria"),
    ("pais vasco", "País Vasco"), ("euskadi", "País Vasco"),
    ("eusko jaurlaritza", "País Vasco"),
    ("navarra", "Navarra"), ("nafarroa", "Navarra"),
    ("la rioja", "La Rioja"),
    ("aragon", "Aragón"),
    ("cataluna", "Cataluña"), ("catalunya", "Cataluña"),
    ("generalitat de catalunya", "Cataluña"),
    ("valencia", "Comunidad Valenciana"),
    ("comunidad valenciana", "Comunidad Valenciana"),
    ("comunitat valenciana", "Comunidad Valenciana"),
    ("generalitat valenciana", "Comunidad Valenciana"),
    ("murcia", "Murcia"), ("region de murcia", "Murcia"),
    ("andalucia", "Andalucía"), ("junta de andalucia", "Andalucía"),
    ("extremadura", "Extremadura"), ("junta de extremadura", "Extremadura"),
    ("castilla-la mancha", "Castilla-La Mancha"),
    ("castilla la mancha", "Castilla-La Mancha"),
    ("castilla y leon", "Castilla y León"),
    ("junta de castilla y leon", "Castilla y León"),
    ("madrid", "Madrid"), ("comunidad de madrid", "Madrid"),
    ("canarias", "Canarias"), ("gobierno de canarias", "Canarias"),
    ("baleares", "Baleares"), ("illes balears", "Baleares"),
    ("govern de les illes balears", "Baleares"),
    ("ceuta", "Ceuta"),
    ("melilla", "Melilla"),
    # Nacional / UE
    ("ministerio", "Nacional"),
    ("gobierno de espana", "Nacional"),
    ("administracion general del estado", "Nacional"),
    ("toda espana", "Nacional"),
    ("todo el estado", "Nacional"),
    ("ambito nacional", "Nacional"),
    ("ambito estatal", "Nacional"),
    ("union europea", "Unión Europea"),
    ("comision europea", "Unión Europea"),
    ("european commission", "Unión Europea"),
    ("horizon europe", "Unión Europea"),
]


def _norm(s: str | None) -> str:
    """Normaliza: minúsculas, sin tildes, espacios colapsados."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s


def _from_text(text: str) -> str | None:
    """Detecta CCAA en un texto libre (organismo, ámbito, regiones…)."""
    n = _norm(text)
    if not n:
        return None
    # 1) Buscar provincias (mapeo claro a CCAA)
    for prov, ccaa in PROVINCIA_CCAA.items():
        if re.search(rf"\b{re.escape(prov)}\b", n):
            return ccaa
    # 2) Buscar patrones de CCAA / nivel nacional / UE
    for pat, ccaa in CCAA_PATTERNS:
        if pat in n:
            return ccaa
    return None


def derive_ccaa(
    source: str | None,
    organismo: str | None = "",
    ambito_geografico: str | None = "",
    comunidad_autonoma: str | None = "",
    regiones: str | None = "",
    extra_text: str | None = "",
) -> str:
    """Devuelve la CCAA canónica de una convocatoria.

    Estrategia:
      1. Si la fuente fija la CCAA (DOG, BOP A Coruña, EU), úsala.
      2. Si el LLM ya devolvió `comunidad_autonoma` y es válida, úsala.
      3. Si no, intenta inferir desde `regiones` (BDNS), `organismo`,
         `ambito_geografico`, en ese orden.
      4. Fallback: "Otra/Desconocida".
    """
    src = (source or "").upper()

    # Reglas duras por fuente
    if src == "DOG":
        return "Galicia"
    if src == "BOP":
        # Nuestro scraper solo cubre el BOP da Coruña
        return "Galicia"
    if src == "EU":
        return "Unión Europea"

    # Si el LLM ya nos dio una CCAA canónica, usarla
    if comunidad_autonoma:
        n = _norm(comunidad_autonoma)
        for canon in CCAA_LIST:
            if _norm(canon) == n or _norm(canon) in n or n in _norm(canon):
                return canon

    # BDNS: el campo `regiones` es lo más fiable
    for blob in (regiones, organismo, ambito_geografico, extra_text):
        if not blob:
            continue
        # Si menciona varias regiones, la primera ya basta
        ccaa = _from_text(blob)
        if ccaa:
            return ccaa

    # BOE/BDNS sin pistas → asumir nacional (es lo más probable)
    if src in ("BOE", "BDNS"):
        return "Nacional"

    return "Otra/Desconocida"


__all__ = ["CCAA_LIST", "derive_ccaa"]
