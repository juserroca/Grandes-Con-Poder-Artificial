#!/usr/bin/env python
"""
Script de debugging para Django con breakpoints y logging detallado
"""
import os
import sys
import django
from django.conf import settings
from django.core.management import execute_from_command_line

# Verificar que estamos en el entorno virtual correcto
def check_venv():
    """Verificar que el entorno virtual esté activo"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Entorno virtual detectado correctamente")
        print(f"📁 Python ejecutable: {sys.executable}")
        print(f"📁 Directorio de trabajo: {os.getcwd()}")
    else:
        print("⚠️  ADVERTENCIA: No se detectó entorno virtual activo")
        print("💡 Ejecuta: venv\\Scripts\\activate (Windows) o source venv/bin/activate (Linux/Mac)")
        print(f"📁 Python ejecutable: {sys.executable}")

# Verificar entorno virtual
check_venv()

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_backend.settings')
django.setup()

# Configurar logging para debugging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug.log')
    ]
)

# Logger específico para debugging
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)

def debug_print(message, data=None):
    """Función helper para debugging con datos estructurados"""
    debug_logger.debug(f"🐛 DEBUG: {message}")
    if data is not None:
        debug_logger.debug(f"📊 DATA: {data}")

if __name__ == '__main__':
    debug_print("Iniciando servidor Django en modo DEBUG")
    debug_print("Configuración de Django cargada", {
        'DEBUG': settings.DEBUG,
        'DATABASES': list(settings.DATABASES.keys()),
        'INSTALLED_APPS': settings.INSTALLED_APPS
    })
    
    # Ejecutar servidor con debugging habilitado
    execute_from_command_line(['manage.py', 'runserver', '0.0.0.0:8000'])
