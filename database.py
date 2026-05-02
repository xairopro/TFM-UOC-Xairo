"""
Sub-Radar – Capa de base de datos (SQLite).

Almacena:
- Usuarios (auth simple con hash)
- Catálogo de documentos descargados
- Análisis LLM cacheados por documento (independientes del perfil)
- Tabla de subvenciones con campos buscables
- Control de fechas analizadas por fuente
- Historial de búsquedas
"""

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from config import BASE_DIR

log = logging.getLogger(__name__)

DB_PATH = BASE_DIR / "data" / "sub_radar.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Una conexión por hilo (thread-local)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), timeout=30)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def _hash_password(password: str) -> str:
    """Hash simple con SHA-256 + salt fijo (suficiente para este TFM)."""
    salt = "sub_radar_2026"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  INIT                                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def init_db():
    """Crea las tablas si no existen y el admin por defecto."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    UNIQUE NOT NULL,
            password    TEXT    NOT NULL,
            role        TEXT    NOT NULL DEFAULT 'user',
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS documents (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            source          TEXT    NOT NULL,
            doc_id          TEXT    NOT NULL,
            titulo          TEXT    NOT NULL DEFAULT '',
            filepath        TEXT    NOT NULL,
            url             TEXT    NOT NULL DEFAULT '',
            date_published  TEXT    NOT NULL DEFAULT '',
            downloaded_at   TEXT    NOT NULL DEFAULT (datetime('now')),
            text_hash       TEXT    NOT NULL DEFAULT '',
            UNIQUE(source, doc_id)
        );

        CREATE TABLE IF NOT EXISTS document_analysis (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id     INTEGER NOT NULL REFERENCES documents(id),
            es_subvencion   INTEGER NOT NULL DEFAULT 0,
            grant_data      TEXT    NOT NULL DEFAULT '{}',
            analyzed_at     TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(document_id)
        );

        CREATE TABLE IF NOT EXISTS grants (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id         INTEGER NOT NULL REFERENCES documents(id),
            source              TEXT    NOT NULL,
            titulo              TEXT    NOT NULL DEFAULT '',
            organismo           TEXT    NOT NULL DEFAULT '',
            fecha_publicacion   TEXT    NOT NULL DEFAULT '',
            fecha_limite        TEXT    NOT NULL DEFAULT '',
            financiacion        TEXT    NOT NULL DEFAULT '',
            destinatarios       TEXT    NOT NULL DEFAULT '',
            requisitos          TEXT    NOT NULL DEFAULT '',
            resumen             TEXT    NOT NULL DEFAULT '',
            tipo_proyecto       TEXT    NOT NULL DEFAULT '',
            link                TEXT    NOT NULL DEFAULT '',
            filepath            TEXT    NOT NULL DEFAULT '',
            bases_reguladoras   TEXT    NOT NULL DEFAULT '',
            documentacion       TEXT    NOT NULL DEFAULT '',
            contacto            TEXT    NOT NULL DEFAULT '',
            ambito_geografico   TEXT    NOT NULL DEFAULT '',
            comunidad_autonoma  TEXT    NOT NULL DEFAULT '',
            raw_json            TEXT    NOT NULL DEFAULT '{}',
            created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(document_id)
        );

        CREATE TABLE IF NOT EXISTS analyzed_dates (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT    NOT NULL,
            date_val    TEXT    NOT NULL,
            analyzed_at TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(source, date_val)
        );

        CREATE TABLE IF NOT EXISTS search_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER REFERENCES users(id),
            params      TEXT    NOT NULL DEFAULT '{}',
            results_count INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS favorites (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            grant_id    INTEGER NOT NULL REFERENCES grants(id) ON DELETE CASCADE,
            note        TEXT    NOT NULL DEFAULT '',
            created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(user_id, grant_id)
        );

        CREATE INDEX IF NOT EXISTS idx_docs_source ON documents(source);
        CREATE INDEX IF NOT EXISTS idx_docs_doc_id ON documents(doc_id);
        CREATE INDEX IF NOT EXISTS idx_docs_date ON documents(date_published);
        CREATE INDEX IF NOT EXISTS idx_analysis_doc ON document_analysis(document_id);
        CREATE INDEX IF NOT EXISTS idx_analysis_grant ON document_analysis(es_subvencion);
        CREATE INDEX IF NOT EXISTS idx_grants_source ON grants(source);
        CREATE INDEX IF NOT EXISTS idx_grants_fecha ON grants(fecha_publicacion);
        CREATE INDEX IF NOT EXISTS idx_grants_limite ON grants(fecha_limite);
        CREATE INDEX IF NOT EXISTS idx_grants_tipo ON grants(tipo_proyecto);
        CREATE INDEX IF NOT EXISTS idx_analyzed_dates ON analyzed_dates(source, date_val);
        CREATE INDEX IF NOT EXISTS idx_favorites_user ON favorites(user_id);
    """)
    conn.commit()

    # Migración suave: añadir comunidad_autonoma si la tabla grants existe sin ella
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(grants)").fetchall()}
    if "comunidad_autonoma" not in cols:
        try:
            conn.execute(
                "ALTER TABLE grants ADD COLUMN comunidad_autonoma TEXT NOT NULL DEFAULT ''"
            )
            conn.commit()
            log.info("DB: migración añadida columna grants.comunidad_autonoma")
        except Exception as e:
            log.warning("DB: no se pudo migrar grants.comunidad_autonoma: %s", e)

    # Índice CCAA (después de garantizar que la columna existe)
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_grants_ccaa ON grants(comunidad_autonoma)"
        )
        conn.commit()
    except Exception as e:
        log.warning("DB: no se pudo crear índice idx_grants_ccaa: %s", e)

    # Crear admin si no existe
    existing = conn.execute("SELECT id FROM users WHERE username='admin'").fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ("admin", _hash_password("admin"), "admin"),
        )
        conn.commit()
        log.info("DB: usuario admin creado (contraseña: admin)")

    log.info("DB: inicializada en %s", DB_PATH)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  USERS                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def authenticate(username: str, password: str) -> dict | None:
    """Retorna el user dict si las credenciales son válidas."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, username, role FROM users WHERE username=? AND password=?",
        (username, _hash_password(password)),
    ).fetchone()
    return dict(row) if row else None


def create_user(username: str, password: str, role: str = "user") -> bool:
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, _hash_password(password), role),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def list_users() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT id, username, role, created_at FROM users ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def delete_user(user_id: int) -> bool:
    conn = _get_conn()
    if user_id == 1:
        return False  # No borrar admin
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    return True


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DOCUMENTS                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def register_document(source: str, doc_id: str, titulo: str, filepath: str,
                      url: str = "", date_published: str = "",
                      text_hash: str = "") -> int | None:
    """Registra un documento; retorna el id o None si ya existe."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO documents (source, doc_id, titulo, filepath, url,
               date_published, text_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (source, doc_id, titulo, filepath, url, date_published, text_hash),
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def get_document(source: str, doc_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM documents WHERE source=? AND doc_id=?", (source, doc_id)
    ).fetchone()
    return dict(row) if row else None


def get_unanalyzed_documents(source: str | None = None) -> list[dict]:
    """Documentos que no tienen análisis LLM aún."""
    conn = _get_conn()
    q = """SELECT d.* FROM documents d
           LEFT JOIN document_analysis da ON da.document_id = d.id
           WHERE da.id IS NULL"""
    params = []
    if source:
        q += " AND d.source = ?"
        params.append(source)
    q += " ORDER BY d.id"
    return [dict(r) for r in conn.execute(q, params).fetchall()]


def count_documents(source: str | None = None) -> int:
    conn = _get_conn()
    if source:
        return conn.execute("SELECT COUNT(*) FROM documents WHERE source=?",
                            (source,)).fetchone()[0]
    return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


def count_analyzed() -> int:
    conn = _get_conn()
    return conn.execute("SELECT COUNT(*) FROM document_analysis").fetchone()[0]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ANALYSIS CACHE                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def save_analysis(document_id: int, es_subvencion: bool, grant_data: dict):
    """Guarda el resultado del análisis LLM para un documento."""
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO document_analysis
               (document_id, es_subvencion, grant_data)
               VALUES (?, ?, ?)""",
            (document_id, int(es_subvencion), json.dumps(grant_data, ensure_ascii=False)),
        )
        conn.commit()
    except Exception as e:
        log.warning("Error guardando análisis doc %d: %s", document_id, e)


def get_cached_grants(source: str | None = None,
                      entity_type: str | None = None,
                      project_types: list[str] | None = None) -> list[dict]:
    """
    Retorna subvenciones identificadas del cache.
    El filtro por entity_type/project_types se aplica en Python
    porque el análisis genérico ya está hecho.
    """
    conn = _get_conn()
    q = """SELECT d.source, d.doc_id, d.filepath, d.url, da.grant_data
           FROM document_analysis da
           JOIN documents d ON d.id = da.document_id
           WHERE da.es_subvencion = 1"""
    params = []
    if source:
        q += " AND d.source = ?"
        params.append(source)
    q += " ORDER BY d.id DESC"

    rows = conn.execute(q, params).fetchall()
    results = []
    for row in rows:
        try:
            data = json.loads(row["grant_data"])
        except (json.JSONDecodeError, TypeError):
            continue
        data["source"] = row["source"]
        data["doc_id"] = row["doc_id"]
        data["filepath"] = row["filepath"]
        if not data.get("link"):
            data["link"] = row["url"]
        results.append(data)
    return results


def get_analysis_stats() -> dict:
    conn = _get_conn()
    total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    analyzed = conn.execute("SELECT COUNT(*) FROM document_analysis").fetchone()[0]
    grants = conn.execute("SELECT COUNT(*) FROM document_analysis WHERE es_subvencion=1").fetchone()[0]
    return {
        "total_docs": total_docs,
        "analyzed": analyzed,
        "pending": total_docs - analyzed,
        "grants": grants,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SEARCH HISTORY                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def save_search(user_id: int | None, params: dict, results_count: int):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO search_history (user_id, params, results_count) VALUES (?, ?, ?)",
        (user_id, json.dumps(params, ensure_ascii=False), results_count),
    )
    conn.commit()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  GRANTS (tabla normalizada para búsqueda)                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def save_grant(document_id: int, source: str, grant_data: dict,
               filepath: str = "", link: str = ""):
    """Guarda una subvención en la tabla normalizada de grants.

    Deriva automáticamente la CCAA combinando fuente + organismo +
    ámbito_geográfico + comunidad_autonoma del LLM + regiones (BDNS).
    """
    from processing.ccaa import derive_ccaa

    ccaa = derive_ccaa(
        source=source,
        organismo=grant_data.get("organismo", ""),
        ambito_geografico=grant_data.get("ambito_geografico", ""),
        comunidad_autonoma=grant_data.get("comunidad_autonoma", ""),
        regiones=grant_data.get("regiones", ""),
        extra_text=(
            (grant_data.get("titulo") or "") + " " +
            (grant_data.get("resumen") or "")
        ),
    )

    conn = _get_conn()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO grants
               (document_id, source, titulo, organismo, fecha_publicacion,
                fecha_limite, financiacion, destinatarios, requisitos,
                resumen, tipo_proyecto, link, filepath, bases_reguladoras,
                documentacion, contacto, ambito_geografico,
                comunidad_autonoma, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document_id,
                source,
                grant_data.get("titulo", ""),
                grant_data.get("organismo", ""),
                grant_data.get("fecha_publicacion", ""),
                grant_data.get("fecha_limite", ""),
                grant_data.get("financiacion", ""),
                grant_data.get("destinatarios", ""),
                grant_data.get("requisitos", ""),
                grant_data.get("resumen", ""),
                grant_data.get("tipo_proyecto", ""),
                link or grant_data.get("link", ""),
                filepath,
                grant_data.get("bases_reguladoras", ""),
                grant_data.get("documentacion", ""),
                grant_data.get("contacto", ""),
                grant_data.get("ambito_geografico", ""),
                ccaa,
                json.dumps(grant_data, ensure_ascii=False),
            ),
        )
        conn.commit()
    except Exception as e:
        log.warning("Error guardando grant doc %d: %s", document_id, e)


def search_grants(query: str | None = None,
                  source: str | None = None,
                  entity_type: str | None = None,
                  project_types: list[str] | None = None,
                  ccaa: list[str] | str | None = None,
                  date_start: str | None = None,
                  date_end: str | None = None,
                  limit: int = 100,
                  offset: int = 0) -> tuple[list[dict], int]:
    """Busca subvenciones en la tabla grants con filtros y scoring.

    `ccaa` puede ser un string o lista de CCAAs (ver processing/ccaa.py).
    Si está vacío no aplica filtro geográfico.
    """
    conn = _get_conn()
    conditions = []
    params: list = []

    if query:
        conditions.append(
            "(g.titulo LIKE ? OR g.organismo LIKE ? OR g.destinatarios LIKE ? "
            "OR g.requisitos LIKE ? OR g.resumen LIKE ? OR g.financiacion LIKE ?)"
        )
        like = f"%{query}%"
        params.extend([like] * 6)

    if source:
        conditions.append("g.source = ?")
        params.append(source)

    if project_types:
        pt_conds = ["g.tipo_proyecto LIKE ?" for _ in project_types]
        conditions.append(f"({' OR '.join(pt_conds)})")
        params.extend(f"%{pt}%" for pt in project_types)

    if ccaa:
        ccaa_list = [ccaa] if isinstance(ccaa, str) else list(ccaa)
        ccaa_list = [c for c in ccaa_list if c]
        if ccaa_list:
            placeholders = ",".join("?" for _ in ccaa_list)
            conditions.append(f"g.comunidad_autonoma IN ({placeholders})")
            params.extend(ccaa_list)

    if date_start:
        conditions.append("g.fecha_publicacion >= ?")
        params.append(date_start)

    if date_end:
        conditions.append("g.fecha_publicacion <= ?")
        params.append(date_end)

    where = " AND ".join(conditions) if conditions else "1=1"

    count_q = f"SELECT COUNT(*) FROM grants g WHERE {where}"
    total = conn.execute(count_q, params).fetchone()[0]

    q = f"""SELECT g.* FROM grants g
            WHERE {where}
            ORDER BY g.fecha_publicacion DESC
            LIMIT ? OFFSET ?"""
    rows = conn.execute(q, params + [limit, offset]).fetchall()

    # Calcular match_score al vuelo si hay entity_type
    results = []
    for r in rows:
        d = dict(r)
        if entity_type:
            from processing.llm_processor import compute_match_score
            d["match_score"] = compute_match_score(d, entity_type, project_types)
        results.append(d)

    # Ordenar por score si se calculó
    if entity_type:
        results.sort(key=lambda x: x.get("match_score", 0), reverse=True)

    return results, total


def get_calendar_events(date_start: str | None = None,
                        date_end: str | None = None) -> list[dict]:
    """Obtiene eventos de calendario (plazos de subvenciones)."""
    conn = _get_conn()
    conditions = ["g.fecha_limite != ''", "g.fecha_limite IS NOT NULL"]
    params: list = []

    if date_start:
        conditions.append("g.fecha_limite >= ?")
        params.append(date_start)
    if date_end:
        conditions.append("g.fecha_limite <= ?")
        params.append(date_end)

    where = " AND ".join(conditions)
    q = f"""SELECT g.id, g.titulo, g.organismo, g.source,
                   g.fecha_publicacion, g.fecha_limite, g.financiacion,
                   g.tipo_proyecto, g.link, g.filepath
            FROM grants g WHERE {where}
            ORDER BY g.fecha_limite ASC"""
    rows = conn.execute(q, params).fetchall()
    return [dict(r) for r in rows]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ANALYZED DATES                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def mark_date_analyzed(source: str, date_val: str):
    """Marca una fecha como analizada para una fuente."""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO analyzed_dates (source, date_val) VALUES (?, ?)",
            (source, date_val),
        )
        conn.commit()
    except Exception as e:
        log.warning("Error marcando fecha: %s", e)


def get_analyzed_dates(source: str) -> set[str]:
    """Retorna las fechas ya analizadas para una fuente."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT date_val FROM analyzed_dates WHERE source = ?", (source,)
    ).fetchall()
    return {r["date_val"] for r in rows}


def get_unanalyzed_date_ranges(sources: list[str],
                                date_start: str,
                                date_end: str) -> dict[str, list[str]]:
    """Retorna fechas que NO han sido analizadas para cada fuente."""
    from datetime import date, timedelta
    result = {}
    for src in sources:
        analyzed = get_analyzed_dates(src)
        d = date.fromisoformat(date_start)
        end = date.fromisoformat(date_end)
        missing = []
        while d <= end:
            if d.weekday() < 5:  # Solo laborables
                ds = d.isoformat()
                if ds not in analyzed:
                    missing.append(ds)
            d += timedelta(days=1)
        result[src] = missing
    return result


def all_dates_analyzed(sources: list[str],
                       date_start: str,
                       date_end: str) -> bool:
    """Comprueba si todas las fechas del rango están analizadas."""
    gaps = get_unanalyzed_date_ranges(sources, date_start, date_end)
    return all(len(v) == 0 for v in gaps.values())


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FAVORITES (recordatorios por usuario)                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def add_favorite(user_id: int, grant_id: int, note: str = "") -> bool:
    """Añade un grant a favoritos del usuario. Idempotente."""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO favorites (user_id, grant_id, note) VALUES (?, ?, ?)",
            (user_id, grant_id, note),
        )
        conn.commit()
        return True
    except Exception as e:
        log.warning("Error añadiendo favorito: %s", e)
        return False


def remove_favorite(user_id: int, grant_id: int) -> bool:
    """Elimina un grant de favoritos."""
    conn = _get_conn()
    conn.execute(
        "DELETE FROM favorites WHERE user_id=? AND grant_id=?",
        (user_id, grant_id),
    )
    conn.commit()
    return True


def list_favorites(user_id: int) -> list[dict]:
    """Devuelve los favoritos del usuario, joineados con grants."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT g.*, f.note AS fav_note, f.created_at AS fav_created_at
           FROM favorites f
           JOIN grants g ON g.id = f.grant_id
           WHERE f.user_id = ?
           ORDER BY f.created_at DESC""",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_favorite_ids(user_id: int) -> set[int]:
    """Devuelve el conjunto de IDs de grants favoritos del usuario."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT grant_id FROM favorites WHERE user_id=?", (user_id,),
    ).fetchall()
    return {r["grant_id"] for r in rows}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ANALYTICS (gráficas del dashboard)                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _parse_amount(text: str) -> float:
    """Extrae un importe en € de una cadena heurísticamente.

    Reconoce formatos como "500.000 euros", "2.000.000 €", "1,5 millones",
    "1.250,50 EUR". Devuelve 0.0 si no detecta nada fiable.
    """
    if not text:
        return 0.0
    import re as _re
    s = text.lower().replace("\xa0", " ")

    # Detectar "X millones / millón"
    m = _re.search(r"(\d+(?:[.,]\d+)?)\s*mill[oó]n", s)
    if m:
        try:
            v = float(m.group(1).replace(".", "").replace(",", "."))
            return v * 1_000_000
        except ValueError:
            pass

    # Detectar números con punto de miles y coma decimal (ES)
    candidates = _re.findall(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?", s)
    best = 0.0
    for c in candidates:
        try:
            v = float(c.replace(".", "").replace(",", "."))
            if v > best:
                best = v
        except ValueError:
            continue
    return best


def get_chart_data(date_start: str | None = None,
                   date_end: str | None = None) -> dict:
    """Calcula agregados para los gráficos del dashboard.

    Devuelve:
      - by_month: [{month: 'YYYY-MM', count, total_eur}]
      - by_source: [{source, count, total_eur}]
      - by_sector: [{sector, count}]
      - by_status: {vigente, vencida, sin_fecha}
      - top_organisms: [{organism, count}]
    """
    conn = _get_conn()
    conditions = ["1=1"]
    params: list = []
    if date_start:
        conditions.append("g.fecha_publicacion >= ?")
        params.append(date_start)
    if date_end:
        conditions.append("g.fecha_publicacion <= ?")
        params.append(date_end)
    where = " AND ".join(conditions)

    rows = conn.execute(
        f"""SELECT g.source, g.fecha_publicacion, g.fecha_limite,
                   g.financiacion, g.tipo_proyecto, g.organismo
            FROM grants g WHERE {where}""",
        params,
    ).fetchall()

    from collections import defaultdict
    by_month_count: dict = defaultdict(int)
    by_month_eur: dict = defaultdict(float)
    by_source_count: dict = defaultdict(int)
    by_source_eur: dict = defaultdict(float)
    by_sector: dict = defaultdict(int)
    by_org: dict = defaultdict(int)
    status = {"vigente": 0, "vencida": 0, "sin_fecha": 0}

    today = datetime.now().date().isoformat()
    for r in rows:
        src = r["source"] or "OTRO"
        fp = (r["fecha_publicacion"] or "")[:7]   # YYYY-MM
        amount = _parse_amount(r["financiacion"] or "")
        sector = (r["tipo_proyecto"] or "Sin clasificar").strip() or "Sin clasificar"
        org = (r["organismo"] or "Otros").strip()[:80] or "Otros"

        by_source_count[src] += 1
        by_source_eur[src] += amount
        if fp:
            by_month_count[fp] += 1
            by_month_eur[fp] += amount
        by_sector[sector] += 1
        by_org[org] += 1

        fl = r["fecha_limite"] or ""
        if not fl:
            status["sin_fecha"] += 1
        elif fl >= today:
            status["vigente"] += 1
        else:
            status["vencida"] += 1

    by_month = sorted(
        [
            {"month": m, "count": by_month_count[m],
             "total_eur": round(by_month_eur[m], 2)}
            for m in by_month_count
        ],
        key=lambda x: x["month"],
    )
    by_source = sorted(
        [
            {"source": s, "count": by_source_count[s],
             "total_eur": round(by_source_eur[s], 2)}
            for s in by_source_count
        ],
        key=lambda x: x["count"], reverse=True,
    )
    by_sector_list = sorted(
        [{"sector": s, "count": c} for s, c in by_sector.items()],
        key=lambda x: x["count"], reverse=True,
    )
    top_orgs = sorted(
        [{"organism": o, "count": c} for o, c in by_org.items()],
        key=lambda x: x["count"], reverse=True,
    )[:10]

    return {
        "by_month": by_month,
        "by_source": by_source,
        "by_sector": by_sector_list,
        "by_status": status,
        "top_organisms": top_orgs,
        "total_grants": len(rows),
        "total_eur": round(sum(by_source_eur.values()), 2),
    }

