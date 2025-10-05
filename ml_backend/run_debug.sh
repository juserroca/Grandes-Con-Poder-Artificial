#!/bin/bash

echo "========================================"
echo "    INICIANDO DJANGO DEBUG MODE"
echo "========================================"
echo

# Verificar si existe el entorno virtual
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ ERROR: No se encontró el entorno virtual"
    echo "💡 Ejecuta: python -m venv venv"
    echo "💡 Luego: source venv/bin/activate"
    echo "💡 Finalmente: pip install -r requirements.txt"
    exit 1
fi

echo "🔧 Activando entorno virtual..."
source venv/bin/activate

echo
echo "✅ Entorno virtual activado"
echo "📁 Directorio: $(pwd)"
echo "🐍 Python: $VIRTUAL_ENV/bin/python"
echo

# Verificar que las dependencias estén instaladas
echo "🔍 Verificando dependencias..."
python -c "import django; print('✅ Django:', django.get_version())" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Django no está instalado"
    echo "💡 Ejecuta: pip install -r requirements.txt"
    exit 1
fi

python -c "import pandas; print('✅ Pandas:', pandas.__version__)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Pandas no está instalado"
    echo "💡 Ejecuta: pip install -r requirements.txt"
    exit 1
fi

echo
echo "🚀 Iniciando servidor Django en modo DEBUG..."
echo "📊 Logs se guardarán en: debug.log"
echo "🌐 Servidor disponible en: http://localhost:8000"
echo
echo "💡 Para detener el servidor presiona Ctrl+C"
echo "========================================"
echo

# Ejecutar el servidor de debugging
python debug_server.py
