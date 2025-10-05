#!/bin/bash

# Script de inicio rápido para ML Backend
echo "🚀 Iniciando ML Backend Django..."

# Verificar si Python está instalado
if ! command -v python &> /dev/null; then
    echo "❌ Python no está instalado. Por favor instala Python 3.8+"
    exit 1
fi

# Verificar si pip está instalado
if ! command -v pip &> /dev/null; then
    echo "❌ pip no está instalado. Por favor instala pip"
    exit 1
fi

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python -m venv venv
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate 2>/dev/null || venv\Scripts\activate 2>/dev/null

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install -r requirements.txt

# Configurar Django
echo "⚙️ Configurando Django..."
python setup.py

# Iniciar servidor
echo "🌐 Iniciando servidor Django..."
echo "   URL: http://localhost:8000"
echo "   Admin: http://localhost:8000/admin"
echo "   API: http://localhost:8000/api/"
echo ""
echo "Presiona Ctrl+C para detener el servidor"
python manage.py runserver
