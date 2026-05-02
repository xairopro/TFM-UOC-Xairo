"""
Agente conversacional de Sub-Radar.

Gestiona la conversación con el usuario, extrae parámetros de búsqueda,
orquesta el scraping y procesamiento, y presenta resultados.

Integra RAG semántico con ChromaDB para respuestas fundamentadas
con citas del texto original de los boletines oficiales.
"""

import asyncio
import json
import logging
import re
import threading
from datetime import date, timedelta
from pathlib import Path

import httpx

# Conciencia temporal – se inyecta en el system prompt para que el LLM razone
# correctamente sobre plazos y fechas en relación al año actual 2026.
_MESES_ES = [
    '', 'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
    'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'
]
_today = date.today()
_TODAY_STR = f"{_today.day} de {_MESES_ES[_today.month]} de {_today.year}"
_DATE_AWARENESS_BLOCK = (
    f"\n\nCONTEXTO TEMPORAL OBLIGATORIO: Hoy es {_TODAY_STR}. "
    "El año en curso es 2026. "
    "Al hablar de plazos abiertos o vencidos, razona SIEMPRE tomando 2026 como el año actual. "
    "Una subvención con fecha límite en 2026 puede estar vigente; una de 2025 ya ha expirado."
)

from config import (
    OLLAMA_URL, OLLAMA_MODEL, ENTITY_TYPES, PROJECT_TYPES,
    SOURCES, RAW_BOE, RAW_DOG, RAW_BOP, RAW_BDNS, RAW_EU, BASE_DIR,
)
from database import (
    get_analysis_stats, save_search,
    search_grants, all_dates_analyzed, get_unanalyzed_date_ranges,
)
from run_scrapers import run_all_scrapers
from processing.text_extractor import build_document_list
from processing.llm_processor import process_documents

log = logging.getLogger(__name__)

# ── System prompt del agente conversacional ─────────────────────────────────
AGENT_SYSTEM = """Eres el asistente de Sub-Radar, un sistema inteligente de búsqueda y análisis de subvenciones públicas en boletines oficiales españoles, gallegos y europeos.

═══════════════════════════════════════════════════════════════════
  PRINCIPIO ABSOLUTO – LOCAL-FIRST STRICT RAG
═══════════════════════════════════════════════════════════════════
NO TIENES ACCESO A INTERNET. NO TIENES CONOCIMIENTO ENCICLOPÉDICO ACTUALIZADO.
Tu única fuente de verdad son los datos almacenados localmente:
  • ChromaDB (texto literal de los documentos – bloque [CONTEXTO RAG])
  • SQLite (subvenciones extraídas – bloque [RESULTADOS DE BÚSQUEDA ESTRUCTURADOS])

REGLAS INVIOLABLES:
1. JAMÁS inventes hechos sobre subvenciones, organismos, fechas, importes o requisitos.
2. JAMÁS afirmes ni insinúes que has consultado internet, Google, sitios web externos
   o "fuentes oficiales en línea".
3. SOLO puedes citar información que aparezca LITERALMENTE en los bloques de contexto.
4. Si una pregunta del usuario es factual sobre subvenciones específicas y NO HAY
   información relevante en NINGÚN bloque de contexto, debes responder EXACTAMENTE
   con esta frase, sin añadir invenciones:

   «No tengo información sobre eso en mi base de datos actual. Sin embargo, puedo
   iniciar una extracción en los boletines oficiales. Ten en cuenta que este proceso
   puede tardar varios minutos.»

   Y a continuación, si tienes pistas sobre fuentes y rango de fechas razonables,
   propón el bloque JSON de acción de búsqueda (ver más abajo).

═══════════════════════════════════════════════════════════════════
  CAPACIDADES
═══════════════════════════════════════════════════════════════════
Buscar subvenciones en estas fuentes deterministas:
  • BOE  – Boletín Oficial del Estado
  • DOG  – Diario Oficial de Galicia
  • BOP  – Boletín Oficial de la Provincia de A Coruña
  • BDNS – Base de Datos Nacional de Subvenciones (Estado + CCAA + EELL)
  • EU   – EU Funding & Tenders Portal (Comisión Europea)

Filtrar por tipo de entidad y tipo de proyecto.

═══════════════════════════════════════════════════════════════════
  FLUJO DE TRABAJO
═══════════════════════════════════════════════════════════════════
Cuando el usuario quiera lanzar una búsqueda, recopila:
  1. Fuentes (si no se especifica → todas)
  2. Rango de fechas (si no se especifica → últimos 3 meses)
  3. Tipo de entidad
  4. Tipo de proyecto / sector (opcional)

Cuando tengas información suficiente, añade al final de tu mensaje un bloque JSON:
```json
{"action": "search", "sources": ["BOE","DOG","BOP","BDNS","EU"], "date_start": "YYYY-MM-DD", "date_end": "YYYY-MM-DD", "entity_type": "clave_tipo", "project_types": ["tipo1"], "ccaa": ["Galicia","Nacional"]}
```

Claves válidas para entity_type: asociacion_cultural, asociacion_general, autonomo, pyme, pyme_tech, empresa, particular, fundacion, ong, cooperativa, administracion, club_deportivo, investigador
Tipos de proyecto válidos: Cultura, Sociedad, Educación, Industria, Agricultura, Tecnología, Medio ambiente, Deporte, Turismo, Salud, Patrimonio, Empleo, Investigación, Vivienda, Igualdad
Valores válidos para ccaa (filtro opcional, lista vacía = todas): "Galicia","Asturias","Cantabria","País Vasco","Navarra","La Rioja","Aragón","Cataluña","Comunidad Valenciana","Murcia","Andalucía","Extremadura","Castilla-La Mancha","Castilla y León","Madrid","Canarias","Baleares","Ceuta","Melilla","Nacional","Unión Europea","Otra/Desconocida"

═══════════════════════════════════════════════════════════════════
  FORMATO DE RESPUESTAS FUNDAMENTADAS CON CITAS
═══════════════════════════════════════════════════════════════════
Para cada afirmación basada en un fragmento [CONTEXTO RAG], cita literalmente:
  «[texto exacto del fragmento]» [Fuente: BOE/DOG/BOP/BDNS/EU, Doc: ID]

Si un dato no aparece en el contexto, di: "Este dato no aparece en los documentos consultados."

Si el usuario saluda o hace preguntas conversacionales generales, responde con
naturalidad y guíale hacia la búsqueda – en ese caso no es necesario aplicar la
frase de fallback.

Responde SIEMPRE en español. Sé conciso pero informativo.""" + _DATE_AWARENESS_BLOCK


def _build_rag_context(query: str, n_results: int = 8) -> str:
    """
    Busca fragmentos semánticamente relevantes en ChromaDB
    y construye un bloque de contexto para el LLM.

    Para evitar latencia inútil, se omite cuando la consulta es muy corta
    (p.ej. saludos, agradecimientos o muletillas).
    """
    q = (query or "").strip().lower()
    if len(q) < 12:
        return ""
    _SHORT_PATTERNS = (
        "hola", "buenas", "buenos días", "buenas tardes", "buenas noches",
        "gracias", "ok", "vale", "perfecto", "adiós", "hasta luego",
        "quién eres", "qué puedes", "qué haces", "ayúdame",
    )
    if any(p in q for p in _SHORT_PATTERNS) and len(q) < 60:
        return ""

    try:
        from vector_store import semantic_search, get_index_stats
        stats = get_index_stats()
        if stats["total_chunks"] == 0:
            return ""

        results = semantic_search(query, n_results=n_results)
        if not results:
            return ""

        context = "\n[CONTEXTO RAG – Fragmentos del texto original de los boletines oficiales]\n"
        context += "Estos son extractos literales de los documentos. Úsalos para fundamentar tus respuestas con citas.\n\n"

        for i, r in enumerate(results, 1):
            confidence = "Alta" if r["distance"] < 0.5 else "Media" if r["distance"] < 0.8 else "Baja"
            context += (
                f"--- Fragmento {i} (Relevancia: {confidence}) ---\n"
                f"Fuente: {r['source']} | Doc: {r['doc_id']}\n"
                f"Título: {r['titulo']}\n"
                f"Fecha publicación: {r['date_published']}\n"
            )
            if r.get("url"):
                context += f"URL original: {r['url']}\n"
            if r.get("filepath"):
                context += f"Archivo local: {r['filepath']}\n"
            context += f"Texto:\n{r['text']}\n\n"

        return context
    except Exception as e:
        log.warning("Error en RAG context: %s", e)
        return ""


class SubRadarAgent:
    """Agente conversacional que orquesta búsqueda de subvenciones."""

    def __init__(self):
        self.history: list[dict] = []
        self.search_results: list[dict] = []
        self.is_processing = False
        self._lock = threading.Lock()

    def reset(self):
        self.history.clear()
        self.search_results.clear()
        self.is_processing = False

    def _build_messages(self, user_msg: str) -> list[dict]:
        """Construye el historial de mensajes para Ollama con contexto RAG."""
        msgs = [{"role": "system", "content": AGENT_SYSTEM}]

        # RAG: buscar fragmentos relevantes en ChromaDB
        rag_context = _build_rag_context(user_msg)
        if rag_context:
            msgs.append({"role": "system", "content": rag_context})
        else:
            # Local-First Strict: si la consulta parece factual y no hay contexto
            # ni resultados estructurados previos, recordamos al modelo la
            # política de fallback EXACTA.
            q = (user_msg or "").strip().lower()
            _GREETING_HINTS = (
                "hola", "buenas", "gracias", "ok", "vale", "perfecto",
                "adiós", "hasta luego", "qué puedes", "quién eres",
                "cómo funcionas", "ayúdame", "ayuda",
            )
            looks_factual = (
                len(q) >= 12
                and not any(g in q for g in _GREETING_HINTS)
                and not self.search_results
            )
            if looks_factual:
                msgs.append({
                    "role": "system",
                    "content": (
                        "[ESTADO DEL CONTEXTO LOCAL]\n"
                        "La búsqueda en ChromaDB y SQLite no ha devuelto "
                        "información relevante para esta consulta. "
                        "Aplica la política Local-First Strict: responde "
                        "EXACTAMENTE con la frase de fallback definida en "
                        "el system prompt y, si procede, ofrece un bloque "
                        "JSON de acción 'search' para iniciar la extracción."
                    ),
                })

        # Agregar contexto de resultados estructurados si hay
        if self.search_results:
            context = (
                "\n[RESULTADOS DE BÚSQUEDA ESTRUCTURADOS]\n"
                "Estos son metadatos extraídos por IA. Complementa con las citas del CONTEXTO RAG.\n\n"
            )
            for i, r in enumerate(self.search_results[:20], 1):
                filepath = r.get("filepath", "")
                url = r.get("link", r.get("url", ""))
                context += (
                    f"\n--- SUBVENCIÓN {i} ---\n"
                    f"Título: {r.get('titulo', 'Sin título')}\n"
                    f"Organismo: {r.get('organismo', 'N/A')}\n"
                    f"Fuente: {r.get('source', 'N/A')}\n"
                    f"Fecha publicación: {r.get('fecha_publicacion', 'N/A')}\n"
                    f"Fecha límite/Plazo: {r.get('fecha_limite', r.get('plazo', 'N/A'))}\n"
                    f"Financiación: {r.get('financiacion', 'N/A')}\n"
                    f"Destinatarios: {r.get('destinatarios', 'N/A')}\n"
                    f"Requisitos: {r.get('requisitos', 'N/A')}\n"
                    f"Resumen: {r.get('resumen', 'N/A')}\n"
                    f"Bases reguladoras: {r.get('bases_reguladoras', 'N/A')}\n"
                    f"Documentación requerida: {r.get('documentacion', 'N/A')}\n"
                    f"Contacto: {r.get('contacto', 'N/A')}\n"
                    f"Ámbito geográfico: {r.get('ambito_geografico', 'N/A')}\n"
                    f"Tipo proyecto: {r.get('tipo_proyecto', 'N/A')}\n"
                    f"Score de matching: {r.get('match_score', 'N/A')}/100\n"
                )
                if url:
                    context += f"Link: {url}\n"
                if filepath:
                    context += f"Archivo: {filepath}\n"
            msgs.append({"role": "system", "content": context})

        # Agregar historial de conversación (últimos 10 mensajes para evitar
        # explotar el contexto cuando hay RAG + resultados estructurados).
        for msg in self.history[-10:]:
            msgs.append(msg)

        msgs.append({"role": "user", "content": user_msg})
        return msgs

    def _extract_action(self, text: str) -> dict | None:
        """Extrae la acción JSON del texto del agente si existe.

        Tolerante a JSON con corchetes anidados (arrays de project_types).
        """
        # Captura no-greedy del bloque entero entre ```json ... ```
        pattern = r'```json\s*(\{.*?\})\s*```'
        for match in re.finditer(pattern, text, re.DOTALL):
            raw = match.group(1)
            try:
                action = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(action, dict) and action.get("action") == "search":
                return action
        return None

    def _clean_response(self, text: str) -> str:
        """Elimina el bloque JSON de acción de la respuesta visible."""
        return re.sub(r'```json\s*\{.*?\}\s*```', '', text, flags=re.DOTALL).strip()

    def chat(self, user_msg: str) -> tuple[str, dict | None]:
        """
        Procesa un mensaje del usuario. Retorna (respuesta, acción_o_None).
        """
        messages = self._build_messages(user_msg)

        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 2048,
            },
        }

        try:
            resp = httpx.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
        except Exception as e:
            log.error("Error en Ollama chat: %s", e)
            return f"Error al conectar con el modelo de lenguaje: {e}", None

        # Guardar en historial
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": content})

        # Extraer acción si existe
        action = self._extract_action(content)
        clean = self._clean_response(content)

        return clean, action

    def chat_stream(self, user_msg: str):
        """
        Versión streaming del chat. Genera tokens uno a uno.
        Retorna un generador de (token, acción_final).
        """
        messages = self._build_messages(user_msg)

        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.3,
                "num_predict": 2048,
            },
        }

        full_response = ""
        try:
            with httpx.stream(
                "POST",
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=120.0,
            ) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        full_response += token
                        if token:
                            yield token, None
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log.error("Error en Ollama stream: %s", e)
            yield f"\nError de conexión: {e}", None
            return

        # Guardar en historial
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": full_response})

        # Extraer acción
        action = self._extract_action(full_response)
        if action:
            yield "", action

    def run_search(self, action: dict, callback=None, user_id: int | None = None):
        """
        Ejecuta la búsqueda basada en la acción del agente.
        Usa cache inteligente: si las fechas ya están analizadas, busca en DB.
        Si no, descarga y analiza solo las fechas faltantes.
        Vectoriza documentos en ChromaDB para RAG.
        Retorna lista de subvenciones aptas.
        """
        sources = action.get("sources", ["BOE", "DOG", "BOP"])
        entity_type = action.get("entity_type", "asociacion_cultural")
        project_types = action.get("project_types", [])
        ccaa_filter = action.get("ccaa") or []
        if isinstance(ccaa_filter, str):
            ccaa_filter = [c.strip() for c in ccaa_filter.split(",") if c.strip()]
        download_only = action.get("download_only", False)

        try:
            d_start = date.fromisoformat(action.get("date_start", ""))
        except (ValueError, TypeError):
            d_start = date.today() - timedelta(days=90)
        try:
            d_end = date.fromisoformat(action.get("date_end", ""))
        except (ValueError, TypeError):
            d_end = date.today()

        if callback:
            entity_desc = ENTITY_TYPES.get(entity_type, entity_type)
            mode = "Solo descarga" if download_only else "Búsqueda completa"
            callback(
                f"{mode}...\n"
                f"Fuentes: {', '.join(sources)}\n"
                f"Fechas: {d_start} a {d_end}\n"
                f"Perfil: {entity_desc}\n"
                f"Sectores: {', '.join(project_types) if project_types else 'Todos'}"
            )

        # Comprobar si el rango ya está analizado en DB
        if not download_only and all_dates_analyzed(
            sources, d_start.isoformat(), d_end.isoformat()
        ):
            if callback:
                callback("Rango ya analizado. Buscando en la base de datos...")

            # Buscar directamente en la DB
            db_results, total = search_grants(
                source=None,
                entity_type=entity_type,
                project_types=project_types,
                ccaa=ccaa_filter or None,
                date_start=d_start.isoformat(),
                date_end=d_end.isoformat(),
                limit=200,
            )

            # Filtrar por fuentes seleccionadas
            aptas = [r for r in db_results if r.get("source") in sources]

            entity_desc = ENTITY_TYPES.get(entity_type, entity_type)
            for g in aptas:
                g["apto_solicitante"] = True
                g["motivo_aptitud"] = f"Destinatarios compatibles con {entity_desc}"

            save_search(
                user_id=user_id,
                params={
                    "sources": sources,
                    "date_start": d_start.isoformat(),
                    "date_end": d_end.isoformat(),
                    "entity_type": entity_type,
                    "project_types": project_types,
                    "from_cache": True,
                },
                results_count=len(aptas),
            )

            self.search_results = aptas
            if callback:
                callback(
                    f"Busqueda completada desde cache: {len(aptas)} subvenciones aptas "
                    f"de {total} en base de datos"
                )
            return aptas

        # Si hay fechas sin analizar, descargar y analizar solo esas
        if not download_only:
            gaps = get_unanalyzed_date_ranges(
                sources, d_start.isoformat(), d_end.isoformat()
            )
            total_gaps = sum(len(v) for v in gaps.values())
            if callback and total_gaps > 0:
                callback(f"{total_gaps} días pendientes de analizar")

        # 1. Scraping
        if callback:
            callback("Descargando boletines oficiales...")
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                run_all_scrapers(sources, d_start, d_end, callback)
            )
        finally:
            loop.close()

        if download_only:
            if callback:
                callback("Descarga completada. Los documentos están disponibles en el explorador de archivos.")
            return []

        # 2. Construir lista de documentos + registrar en DB
        if callback:
            callback("Indexando documentos...")
        docs = build_document_list(RAW_BOE, RAW_DOG, RAW_BOP, RAW_BDNS, RAW_EU)

        if not docs:
            if callback:
                callback("No se encontraron documentos en el rango indicado.")
            return []

        # 3. Vectorizar en ChromaDB para RAG
        if callback:
            callback("Vectorizando documentos para búsqueda semántica...")
        try:
            from run_processing import vectorize_documents
            vectorize_documents(docs)
        except Exception as e:
            log.warning("Error en vectorización: %s", e)

        # 4. Procesar con LLM (usa cache de DB)
        if callback:
            stats = get_analysis_stats()
            callback(
                f"Analizando documentos con IA... "
                f"({stats['analyzed']} ya catalogados, "
                f"{stats['pending']} pendientes)"
            )
        aptas = process_documents(docs, entity_type, project_types, callback)

        # 5. Guardar en historial de búsqueda
        save_search(
            user_id=user_id,
            params={
                "sources": sources,
                "date_start": d_start.isoformat(),
                "date_end": d_end.isoformat(),
                "entity_type": entity_type,
                "project_types": project_types,
            },
            results_count=len(aptas),
        )

        self.search_results = aptas
        return aptas

    def format_results_message(self, results: list[dict]) -> str:
        """Formatea los resultados como mensaje conversacional con enlaces a fuentes."""
        if not results:
            return (
                "No he encontrado subvenciones que se ajusten a tu perfil "
                "en el rango de fechas indicado. Quieres probar con otras fechas "
                "o en otros boletines?"
            )

        msg = f"He encontrado **{len(results)} subvenciones** que se ajustan a tu perfil:\n\n"

        for i, r in enumerate(results[:20], 1):
            filepath = r.get("filepath", "")
            url = r.get("link", r.get("url", ""))
            link_md = ""

            if filepath:
                try:
                    rel = str(Path(filepath).relative_to(BASE_DIR))
                except ValueError:
                    rel = filepath
                fname = Path(filepath).name
                link_md = (
                    f' <a href="#" onclick="viewFile(\'{rel}\', \'{fname}\'); return false;">'
                    f'Ver documento</a>'
                )
            if url:
                link_md += f" | [Ver en web original]({url})"

            score = r.get("match_score", "")
            score_str = f" (Score: {score}/100)" if score else ""

            msg += f"### {i}. {r.get('titulo', 'Sin título')[:120]}{score_str}\n"
            msg += f"- **Organismo:** {r.get('organismo', 'N/A')}\n"
            msg += f"- **Fuente:** {r.get('source', 'N/A')}\n"
            msg += f"- **Fecha publicación:** {r.get('fecha_publicacion', 'N/A')}\n"
            msg += f"- **Fecha límite:** {r.get('fecha_limite', r.get('plazo', 'N/A'))}\n"
            msg += f"- **Financiación:** {r.get('financiacion', 'N/A')}\n"
            msg += f"- **Destinatarios:** {r.get('destinatarios', 'N/A')}\n"
            msg += f"- **Requisitos:** {r.get('requisitos', 'N/A')}\n"
            msg += f"- **Ámbito:** {r.get('ambito_geografico', 'N/A')}\n"
            msg += f"- **Resumen:** {r.get('resumen', 'N/A')}\n"
            if r.get('documentacion'):
                msg += f"- **Documentación:** {r.get('documentacion')}\n"
            if r.get('contacto'):
                msg += f"- **Contacto:** {r.get('contacto')}\n"
            if link_md:
                msg += f"- {link_md}\n"
            msg += "\n"

        if len(results) > 20:
            msg += f"\n... y {len(results) - 20} más. Puedes preguntarme sobre cualquiera de ellas.\n"

        msg += "\nQuieres que te amplíe información sobre alguna de estas subvenciones? Puedo buscar en el texto original del boletín."
        return msg
