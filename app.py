"""
Sub-Radar – Interfaz Web (Flask + SocketIO)

Sistema de agentes inteligentes para la identificación, análisis
y búsqueda de convocatorias de ayudas públicas.
"""

import json
import logging
import os
import threading
from datetime import date, timedelta
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, session, redirect, url_for,
)
from flask_socketio import SocketIO, emit

from config import (
    BASE_DIR, RAW_BOE, RAW_DOG, RAW_BOP, RAW_BDNS, RAW_EU, PROCESSED_DIR,
    ENTITY_TYPES, PROJECT_TYPES, SOURCES,
)
from database import (
    init_db, authenticate, create_user, list_users, delete_user,
    get_analysis_stats, search_grants, get_calendar_events,
    all_dates_analyzed, add_favorite, remove_favorite, list_favorites,
    get_favorite_ids, get_chart_data,
)
from agent import SubRadarAgent

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("sub_radar.app")

# ── Flask App ──────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# SECRET_KEY persistente: cargado desde variable de entorno o fichero local.
# Esto evita que todas las sesiones queden invalidadas al reiniciar el servidor.
_SECRET_KEY_FILE = BASE_DIR / "data" / ".secret_key"
_secret_key_env = os.environ.get("SUB_RADAR_SECRET_KEY")
if _secret_key_env:
    app.config["SECRET_KEY"] = _secret_key_env.encode()
elif _SECRET_KEY_FILE.exists():
    app.config["SECRET_KEY"] = _SECRET_KEY_FILE.read_bytes()
else:
    _new_key = os.urandom(32)
    _SECRET_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SECRET_KEY_FILE.write_bytes(_new_key)
    app.config["SECRET_KEY"] = _new_key
    log.info("SECRET_KEY generada y guardada en %s", _SECRET_KEY_FILE)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Inicializar base de datos ──────────────────────────────────────────────────
init_db()

# ── Agente por sesión WebSocket ────────────────────────────────────────────────
agents: dict[str, SubRadarAgent] = {}


def get_agent(sid: str) -> SubRadarAgent:
    if sid not in agents:
        agents[sid] = SubRadarAgent()
    return agents[sid]


def login_required(f):
    """Decorador para rutas que requieren autenticación."""
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Decorador para rutas que requieren rol admin."""
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        user = session.get("user")
        if not user or user.get("role") != "admin":
            return jsonify({"error": "Acceso denegado"}), 403
        return f(*args, **kwargs)
    return decorated


# ── Rutas de autenticación ─────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if session.get("user"):
            return redirect(url_for("index"))
        return render_template("login.html")

    data = request.get_json() if request.is_json else request.form
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        if request.is_json:
            return jsonify({"error": "Usuario y contraseña requeridos"}), 400
        return render_template("login.html", error="Usuario y contraseña requeridos")

    user = authenticate(username, password)
    if not user:
        if request.is_json:
            return jsonify({"error": "Credenciales inválidas"}), 401
        return render_template("login.html", error="Credenciales inválidas")

    session["user"] = user
    if request.is_json:
        return jsonify({"ok": True, "user": user})
    return redirect(url_for("index"))


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() if request.is_json else request.form
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Usuario y contraseña requeridos"}), 400
    if len(username) < 3 or len(password) < 4:
        return jsonify({"error": "Usuario (mín 3 chars) y contraseña (mín 4 chars)"}), 400

    ok = create_user(username, password)
    if not ok:
        return jsonify({"error": "El usuario ya existe"}), 409

    user = authenticate(username, password)
    session["user"] = user
    return jsonify({"ok": True, "user": user})


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/api/user")
def api_user():
    user = session.get("user")
    if user:
        return jsonify(user)
    return jsonify(None)


@app.route("/api/admin/users")
@admin_required
def api_admin_users():
    return jsonify(list_users())


@app.route("/api/admin/users/<int:user_id>", methods=["DELETE"])
@admin_required
def api_admin_delete_user(user_id):
    ok = delete_user(user_id)
    if not ok:
        return jsonify({"error": "No se puede eliminar este usuario"}), 400
    return jsonify({"ok": True})


# ── Rutas HTTP principales ─────────────────────────────────────────────────────

@app.route("/")
@login_required
def index():
    from processing.ccaa import CCAA_LIST
    return render_template("index.html",
                           entity_types=ENTITY_TYPES,
                           project_types=PROJECT_TYPES,
                           sources=SOURCES,
                           ccaa_list=CCAA_LIST,
                           user=session["user"])


@app.route("/api/files")
@login_required
def api_files():
    """Devuelve el árbol de archivos descargados."""
    tree = {}
    for source_name, raw_dir in [
        ("BOE", RAW_BOE), ("DOG", RAW_DOG), ("BOP", RAW_BOP),
        ("BDNS", RAW_BDNS), ("EU", RAW_EU),
    ]:
        files = []
        if raw_dir.exists():
            for f in sorted(raw_dir.iterdir()):
                if f.is_file():
                    files.append({
                        "name": f.name,
                        "size": f.stat().st_size,
                        "ext": f.suffix,
                        "path": str(f.relative_to(BASE_DIR)),
                    })
        tree[source_name] = {
            "count": len(files),
            "files": files,
        }
    return jsonify(tree)


@app.route("/api/files/view/<path:filepath>")
@login_required
def api_file_view(filepath):
    """Devuelve el contenido de un archivo para visualización.

    Defensa en profundidad: sólo se permiten ficheros bajo data/raw/* o
    data/processed/* y con extensiones permitidas. Cualquier intento de
    ruta con `..` o fuera del whitelist devuelve 403.
    """
    full_path = (BASE_DIR / filepath).resolve()

    # Whitelist estricta de directorios accesibles
    allowed_roots = [
        (BASE_DIR / "data" / "raw").resolve(),
        (BASE_DIR / "data" / "processed").resolve(),
    ]
    if not any(
        str(full_path).startswith(str(root) + os.sep) or full_path == root
        for root in allowed_roots
    ):
        return jsonify({"error": "Acceso denegado"}), 403

    # Whitelist de extensiones servibles
    if full_path.suffix.lower() not in (".html", ".xml", ".json", ".pdf", ".txt"):
        return jsonify({"error": "Tipo de archivo no soportado"}), 400

    if not full_path.exists() or not full_path.is_file():
        return jsonify({"error": "Archivo no encontrado"}), 404

    if full_path.suffix.lower() in (".html", ".xml", ".json", ".txt"):
        content = full_path.read_text(encoding="utf-8", errors="replace")
        return jsonify({"content": content, "type": full_path.suffix.lower()})
    if full_path.suffix.lower() == ".pdf":
        return send_from_directory(
            str(full_path.parent), full_path.name,
            mimetype="application/pdf",
        )
    return jsonify({"error": "Tipo de archivo no soportado"}), 400


@app.route("/api/grants/search")
@login_required
def api_grants_search():
    """Búsqueda de subvenciones en la base de datos con filtros."""
    query = request.args.get("q", "").strip()
    source = request.args.get("source", "").strip() or None
    date_start = request.args.get("date_start", "").strip() or None
    date_end = request.args.get("date_end", "").strip() or None
    tipo = request.args.get("tipo_proyecto", "").strip()
    project_types = [t.strip() for t in tipo.split(",") if t.strip()] if tipo else None
    ccaa_raw = request.args.get("ccaa", "").strip()
    ccaa = [c.strip() for c in ccaa_raw.split(",") if c.strip()] if ccaa_raw else None
    entity_type = request.args.get("entity_type", "").strip() or None
    limit = min(int(request.args.get("limit", 50)), 200)
    offset = int(request.args.get("offset", 0))

    results, total = search_grants(
        query=query or None,
        source=source,
        project_types=project_types,
        ccaa=ccaa,
        entity_type=entity_type,
        date_start=date_start,
        date_end=date_end,
        limit=limit,
        offset=offset,
    )

    return jsonify({
        "results": results,
        "total": total,
        "limit": limit,
        "offset": offset,
    })


@app.route("/api/calendar")
@login_required
def api_calendar():
    """Eventos de calendario (plazos de subvenciones)."""
    date_start = request.args.get("start", "").strip() or None
    date_end = request.args.get("end", "").strip() or None
    events = get_calendar_events(date_start, date_end)
    return jsonify(events)


@app.route("/api/results")
@login_required
def api_results():
    """Devuelve los resultados procesados."""
    aptas_file = PROCESSED_DIR / "subvenciones_aptas.json"
    all_file = PROCESSED_DIR / "todas_subvenciones.json"

    # Fallback para archivos con el nombre viejo
    if not aptas_file.exists():
        aptas_file = PROCESSED_DIR / "subvenciones_culturales.json"

    data = {"aptas": [], "todas": [], "has_data": False}

    if aptas_file.exists():
        data["aptas"] = json.loads(aptas_file.read_text(encoding="utf-8"))
        data["has_data"] = True
    if all_file.exists():
        data["todas"] = json.loads(all_file.read_text(encoding="utf-8"))

    return jsonify(data)


@app.route("/api/stats")
@login_required
def api_stats():
    """Estadísticas generales del sistema (archivos + DB)."""
    file_stats = {}
    for name, raw_dir in [
        ("BOE", RAW_BOE), ("DOG", RAW_DOG), ("BOP", RAW_BOP),
        ("BDNS", RAW_BDNS), ("EU", RAW_EU),
    ]:
        count = sum(1 for _ in raw_dir.glob("*")) if raw_dir.exists() else 0
        file_stats[name] = count
    file_stats["total"] = sum(file_stats.values())

    db_stats = get_analysis_stats()

    return jsonify({
        **file_stats,
        "db_docs": db_stats["total_docs"],
        "db_analyzed": db_stats["analyzed"],
        "db_pending": db_stats["pending"],
        "db_grants": db_stats["grants"],
    })


# ── Favoritos ──────────────────────────────────────────────────────────────────

@app.route("/api/favorites", methods=["GET"])
@login_required
def api_favorites_list():
    """Lista los favoritos del usuario actual."""
    user = session["user"]
    favs = list_favorites(user["id"])
    return jsonify({"favorites": favs, "total": len(favs)})


@app.route("/api/favorites/<int:grant_id>", methods=["POST"])
@login_required
def api_favorites_toggle(grant_id):
    """Alterna el favorito (añade si no existe, elimina si existe)."""
    user = session["user"]
    current = get_favorite_ids(user["id"])
    if grant_id in current:
        remove_favorite(user["id"], grant_id)
        return jsonify({"action": "removed", "grant_id": grant_id})
    note = ""
    try:
        if request.is_json:
            note = (request.get_json(silent=True) or {}).get("note", "")
    except Exception:
        note = ""
    add_favorite(user["id"], grant_id, note)
    return jsonify({"action": "added", "grant_id": grant_id})


@app.route("/api/favorites/<int:grant_id>", methods=["DELETE"])
@login_required
def api_favorites_remove(grant_id):
    user = session["user"]
    remove_favorite(user["id"], grant_id)
    return jsonify({"action": "removed", "grant_id": grant_id})


@app.route("/api/favorites/ids")
@login_required
def api_favorites_ids():
    """Devuelve sólo los IDs (eficiente para marcar estrellas en listas)."""
    user = session["user"]
    return jsonify({"ids": sorted(get_favorite_ids(user["id"]))})


# ── Analítica para gráficos del dashboard ──────────────────────────────────────

@app.route("/api/charts/overview")
@login_required
def api_charts_overview():
    """Datos agregados para los gráficos del dashboard premium."""
    date_start = request.args.get("date_start", "").strip() or None
    date_end = request.args.get("date_end", "").strip() or None
    return jsonify(get_chart_data(date_start, date_end))


# ── WebSocket Events ───────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    sid = request.sid
    log.info("Cliente conectado: %s", sid)
    get_agent(sid)
    emit("agent_message", {
        "text": (
            "¡Hola! 👋 Soy el asistente de **Sub-Radar**.\n\n"
            "Te ayudo a encontrar subvenciones y ayudas públicas en los boletines oficiales "
            "españoles y gallegos.\n\n"
            "Puedes decirme:\n"
            "- **Qué tipo de entidad eres** (asociación cultural, autónomo, PYME...)\n"
            "- **En qué boletines quieres buscar** (BOE, DOG, BOP)\n"
            "- **En qué fechas buscar**\n"
            "- **Qué tipo de proyectos te interesan** (cultura, educación, industria...)\n\n"
            "Por ejemplo: *\"Soy una asociación cultural sin ánimo de lucro y quiero buscar "
            "subvenciones en el DOG de febrero de 2026\"*\n\n"
            "¿En qué puedo ayudarte?"
        ),
        "type": "greeting",
    })


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    agents.pop(sid, None)
    log.info("Cliente desconectado: %s", sid)


@socketio.on("user_message")
def on_user_message(data):
    sid = request.sid
    msg = data.get("message", "").strip()
    if not msg:
        return

    agent = get_agent(sid)

    if agent.is_processing:
        emit("agent_message", {
            "text": "⏳ Estoy procesando una búsqueda. Por favor, espera a que termine.",
            "type": "info",
        })
        return

    # Streaming token-a-token (mejor UX). Si el agente devuelve una acción,
    # se ejecuta la búsqueda en background al final.
    emit("agent_typing", {"typing": True})
    full_text = []
    final_action = None
    try:
        for token, action in agent.chat_stream(msg):
            if token:
                full_text.append(token)
                socketio.emit("agent_token", {"token": token}, to=sid)
            if action:
                final_action = action
    except Exception as e:
        log.exception("Error en streaming chat")
        socketio.emit("agent_message", {
            "text": f"⚠️ Error en el modelo: {e}",
            "type": "error",
        }, to=sid)
        emit("agent_typing", {"typing": False})
        return

    emit("agent_typing", {"typing": False})

    # Mensaje final consolidado (limpio del bloque JSON de acción)
    final_clean = agent._clean_response("".join(full_text))
    socketio.emit("agent_message", {
        "text": final_clean,
        "type": "response",
        "final": True,
    }, to=sid)

    if final_action and final_action.get("action") == "search":
        user = session.get("user")
        user_id = user["id"] if user else None
        _run_search_background(sid, agent, final_action, user_id)


@socketio.on("quick_search")
def on_quick_search(data):
    """Búsqueda rápida con parámetros directos desde la UI."""
    sid = request.sid
    agent = get_agent(sid)

    if agent.is_processing:
        emit("agent_message", {
            "text": "⏳ Ya hay una búsqueda en curso. Espera a que termine.",
            "type": "info",
        })
        return

    download_only = data.get("download_only", False)

    action = {
        "action": "search",
        "sources": data.get("sources", ["BOE", "DOG", "BOP"]),
        "date_start": data.get("date_start", (date.today() - timedelta(days=90)).isoformat()),
        "date_end": data.get("date_end", date.today().isoformat()),
        "entity_type": data.get("entity_type", "asociacion_cultural"),
        "project_types": data.get("project_types", []),
        "ccaa": data.get("ccaa", []),
        "download_only": download_only,
    }

    entity_desc = ENTITY_TYPES.get(action["entity_type"], action["entity_type"])
    mode = "Solo descargar boletines" if download_only else "Buscar subvenciones"
    emit("agent_message", {
        "text": (
            f"**{mode}** con estos parámetros:\n\n"
            f"- **Fuentes:** {', '.join(action['sources'])}\n"
            f"- **Fechas:** {action['date_start']} → {action['date_end']}\n"
            f"- **Perfil:** {entity_desc}\n"
            f"- **Sectores:** {', '.join(action['project_types']) if action['project_types'] else 'Todos'}\n"
            f"- **CCAA:** {', '.join(action['ccaa']) if action['ccaa'] else 'Todas'}\n\n"
            f"Dame un momento..."
        ),
        "type": "response",
    })

    user = session.get("user")
    user_id = user["id"] if user else None
    _run_search_background(sid, agent, action, user_id)


@socketio.on("reset_conversation")
def on_reset():
    sid = request.sid
    agent = get_agent(sid)
    if not agent.is_processing:
        agent.reset()
        emit("agent_message", {
            "text": "Conversación reiniciada. ¿En qué puedo ayudarte?",
            "type": "info",
        })


@socketio.on("manual_download")
def on_manual_download(data):
    """Descarga manual + análisis lanzada desde el modal premium del dashboard.

    Reusa el pipeline de búsqueda con `download_only=False` por defecto y
    emite eventos `search_progress` para alimentar la barra de progreso.
    """
    sid = request.sid
    agent = get_agent(sid)
    if agent.is_processing:
        emit("search_progress", {
            "phase": "busy", "percent": 0,
            "text": "Ya hay una operación en curso.",
        })
        return

    action = {
        "action": "search",
        "sources": data.get("sources", ["BOE", "DOG", "BOP", "BDNS", "EU"]),
        "date_start": data.get("date_start", (date.today() - timedelta(days=30)).isoformat()),
        "date_end": data.get("date_end", date.today().isoformat()),
        "entity_type": data.get("entity_type", "asociacion_cultural"),
        "project_types": data.get("project_types", []),
        "download_only": bool(data.get("download_only", False)),
    }

    user = session.get("user")
    user_id = user["id"] if user else None
    _run_search_background(sid, agent, action, user_id)


def _run_search_background(sid, agent, action, user_id=None):
    """Ejecuta la búsqueda en un hilo separado.

    Además del callback textual (`search_status`), parsea heurísticamente
    los mensajes para emitir un evento estructurado `search_progress`
    consumible por la barra de progreso del modal de descarga manual.
    """
    import re as _re_progress

    _PHASES = [
        ("scrape",   ["scraping", "descargando", "boe", "dog", "bop", "bdns", "eu", "scrape"]),
        ("extract",  ["construyendo lista", "extrayendo texto", "indexando"]),
        ("filter",   ["pre-filtro", "capa 1", "filtro keywords"]),
        ("triage",   ["triage", "capa 2"]),
        ("analyze",  ["extracción", "capa 3", "qwen2.5:14b", "analizando"]),
        ("save",     ["guardando", "save_grant", "completado"]),
    ]

    def _detect_phase(text: str) -> str:
        t = text.lower()
        for phase, kws in _PHASES:
            if any(kw in t for kw in kws):
                return phase
        return "working"

    def _detect_progress(text: str) -> tuple[int, int] | None:
        # Patrones tipo "(12/345)", "[42/100]", "12 / 345", "doc 12 de 345"
        m = _re_progress.search(r"[\(\[\s](\d+)\s*[/de]+\s*(\d+)[\)\]\s]", text)
        if m:
            try:
                cur, tot = int(m.group(1)), int(m.group(2))
                if 0 <= cur <= tot and tot > 0:
                    return cur, tot
            except ValueError:
                pass
        return None

    def _worker():
        agent.is_processing = True

        # ── Heartbeat: emite un latido cada 5s con segundos transcurridos
        # para que la UI nunca aparente estar congelada aunque ningún
        # callback se dispare durante un tramo (p.e. scraper lento).
        import time as _t_hb
        _start_ts = _t_hb.time()
        _hb_stop = threading.Event()
        _hb_state = {"last_msg": "", "last_phase": "working"}

        def _heartbeat():
            while not _hb_stop.wait(5.0):
                secs = int(_t_hb.time() - _start_ts)
                socketio.emit("search_progress", {
                    "phase": _hb_state["last_phase"],
                    "text": f"⏳ {_hb_state['last_msg'] or 'Procesando…'} ({secs}s)",
                    "heartbeat": True,
                }, to=sid)

        threading.Thread(target=_heartbeat, daemon=True).start()

        try:
            def status_callback(msg):
                _hb_state["last_msg"] = msg[:200]
                _hb_state["last_phase"] = _detect_phase(msg)
                socketio.emit("search_status", {"text": msg}, to=sid)
                phase = _detect_phase(msg)
                prog = _detect_progress(msg)
                payload = {"phase": phase, "text": msg}
                if prog:
                    cur, tot = prog
                    payload.update({
                        "current": cur, "total": tot,
                        "percent": int(cur * 100 / tot),
                    })
                socketio.emit("search_progress", payload, to=sid)

            socketio.emit("search_progress", {
                "phase": "start", "percent": 0, "text": "Iniciando…",
            }, to=sid)

            results = agent.run_search(
                action, callback=status_callback, user_id=user_id
            )

            socketio.emit("search_progress", {
                "phase": "done", "percent": 100, "text": "Completado",
            }, to=sid)

            if action.get("download_only"):
                socketio.emit("agent_message", {
                    "text": "✅ **Descarga completada.** Los documentos están disponibles en la pestaña de Archivos.",
                    "type": "info",
                }, to=sid)
            else:
                results_msg = agent.format_results_message(results)
                socketio.emit("agent_message", {
                    "text": results_msg,
                    "type": "results",
                }, to=sid)

                socketio.emit("search_results_data", {
                    "results": results[:50],
                    "total": len(results),
                }, to=sid)

        except Exception as e:
            log.exception("Error en búsqueda")
            socketio.emit("agent_message", {
                "text": f"⚠️ Error durante la búsqueda: {e}",
                "type": "error",
            }, to=sid)
            socketio.emit("search_progress", {
                "phase": "error", "percent": 0, "text": str(e),
            }, to=sid)
        finally:
            _hb_stop.set()
            agent.is_processing = False
            socketio.emit("search_complete", {}, to=sid)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Iniciando Sub-Radar en http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)

