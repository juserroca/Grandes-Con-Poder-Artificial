#!/bin/bash

# Script de inicio con logs para Linux/Mac
echo "🚀 ML BACKEND - SERVIDOR DJANGO CON LOGS"
echo "================================================"
echo "Iniciando servidor Django con logging detallado..."
echo "Los logs aparecerán en tiempo real en esta consola"
echo "================================================"
echo

# Verificar que Python esté instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 no está instalado"
    echo "   Por favor instala Python 3.8+"
    exit 1
fi

# Verificar que pip esté instalado
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 no está instalado"
    echo "   Por favor instala pip3"
    exit 1
fi

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install -r requirements.txt

# Configurar Django si es necesario
if [ ! -f "db.sqlite3" ]; then
    echo "⚙️ Configurando Django..."
    python setup.py
fi

echo
echo "🌐 El servidor estará disponible en: http://localhost:8000"
echo "📊 API disponible en: http://localhost:8000/api/"
echo "🔧 Panel de administración: http://localhost:8000/admin/"
echo "📝 Logs guardados en: ml_backend.log"
echo
echo "💡 Presiona Ctrl+C para detener el servidor"
echo "================================================"
echo

# Ejecutar Django con logs
echo "🔧 Iniciando servidor Django..."
python manage.py runserver

