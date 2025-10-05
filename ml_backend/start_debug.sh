#!/bin/bash
echo "Iniciando Django en modo DEBUG con breakpoints..."
echo

# Activar entorno virtual
source venv/bin/activate

# Ejecutar con configuración de debugging
python debug_with_breakpoints.py
