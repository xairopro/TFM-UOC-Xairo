#!/usr/bin/env bash
# ── Sub-Radar – Script de inicio ──────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/venv"
PYTHON="$VENV/bin/python3"

# ── Colores ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "  ███████╗██╗   ██╗██████╗       ██████╗  █████╗ ██████╗  █████╗ ██████╗ "
echo "  ██╔════╝██║   ██║██╔══██╗      ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗"
echo "  ███████╗██║   ██║██████╔╝█████╗██████╔╝███████║██║  ██║███████║██████╔╝"
echo "  ╚════██║██║   ██║██╔══██╗╚════╝██╔══██╗██╔══██║██║  ██║██╔══██║██╔══██╗"
echo "  ███████║╚██████╔╝██████╔╝      ██║  ██║██║  ██║██████╔╝██║  ██║██║  ██║"
echo "  ╚══════╝ ╚═════╝ ╚═════╝       ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝"
echo -e "${NC}"
echo -e "${CYAN}  Inteligencia de Subvenciones – TFM UOC${NC}"
echo ""

# ── 1. Comprobar entorno virtual ───────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
    echo -e "${YELLOW}[!] Entorno virtual no encontrado. Creando...${NC}"
    python3 -m venv "$VENV"
    echo -e "${GREEN}[✓] Entorno virtual creado.${NC}"
fi

source "$VENV/bin/activate"

# ── 2. Instalar / actualizar dependencias ─────────────────────────────────────
echo -e "${CYAN}[→] Comprobando dependencias...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}[✓] Dependencias listas.${NC}"

# ── 3. Comprobar que Ollama está en marcha ─────────────────────────────────────
echo -e "${CYAN}[→] Comprobando Ollama (LLM local)...${NC}"
if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}[✓] Ollama disponible.${NC}"
else
    echo -e "${YELLOW}[!] Ollama no responde en localhost:11434.${NC}"
    echo -e "${YELLOW}    El sistema funcionará pero el análisis LLM no estará disponible.${NC}"
    echo -e "${YELLOW}    Inicia Ollama con: ollama serve${NC}"
fi

# ── 4. Comprobar modelo Qwen instalado ────────────────────────────────────────
if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    if curl -sf http://localhost:11434/api/tags | grep -q "qwen2.5:14b"; then
        echo -e "${GREEN}[✓] Modelo qwen2.5:14b disponible.${NC}"
    else
        echo -e "${YELLOW}[!] Modelo qwen2.5:14b no encontrado.${NC}"
        echo -e "${YELLOW}    Descárgalo con: ollama pull qwen2.5:14b${NC}"
    fi
fi

# ── 5. Inicializar base de datos ───────────────────────────────────────────────
echo -e "${CYAN}[→] Inicializando base de datos...${NC}"
"$PYTHON" -c "from database import init_db; init_db(); print('[✓] Base de datos lista.')"
echo -e "${GREEN}[✓] Base de datos lista.${NC}"

# ── 6. Arrancar la aplicación ──────────────────────────────────────────────────
PORT=5000
URL="http://localhost:$PORT"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Sub-Radar arrancando en ${CYAN}${URL}${GREEN}  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${CYAN}Ctrl+C para detener el servidor${NC}"
echo ""

# Abrir el navegador automáticamente (si hay entorno gráfico)
if command -v xdg-open &> /dev/null && [ -n "$DISPLAY" ]; then
    (sleep 2 && xdg-open "$URL") &
elif command -v open &> /dev/null; then
    (sleep 2 && open "$URL") &
fi

"$PYTHON" app.py
