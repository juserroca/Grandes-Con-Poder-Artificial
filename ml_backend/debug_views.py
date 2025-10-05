#!/usr/bin/env python
"""
Script para debugging específico de las vistas de ML
"""
import os
import sys
import django
from django.conf import settings
from django.core.management import execute_from_command_line

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_backend.debug_settings')
django.setup()

# Importar las vistas para debugging
from ml_models.views import train_model, predict, list_models
from ml_models.debug_middleware import DebugMiddleware

# Configurar logging específico para debugging de vistas
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_views.log')
    ]
)

debug_logger = logging.getLogger('debug_views')
debug_logger.setLevel(logging.DEBUG)

def test_train_model():
    """Función para probar el entrenamiento de modelos con debugging"""
    debug_logger.debug("🧪 Iniciando test de train_model")
    
    # Datos de prueba
    test_data = {
        'analysis_type': 'app-data',
        'model': 'random-forest',
        'hyperparameters': {'n_estimators': 100, 'max_depth': 10},
        'dataset_name': 'NASA Kepler Dataset'
    }
    
    debug_logger.debug(f"🧪 Datos de prueba: {test_data}")
    
    # Simular request
    class MockRequest:
        def __init__(self, data):
            self.data = data
            self.method = 'POST'
            self.path = '/api/train-model/'
            self.headers = {}
    
    request = MockRequest(test_data)
    
    try:
        # Llamar a la función con debugging
        debug_logger.debug("🧪 Llamando a train_model...")
        result = train_model(request)
        debug_logger.debug(f"🧪 Resultado: {result}")
    except Exception as e:
        debug_logger.error(f"🧪 Error en test: {e}")
        import traceback
        debug_logger.error(f"🧪 Traceback: {traceback.format_exc()}")

if __name__ == '__main__':
    debug_logger.debug("🚀 Iniciando debugging de vistas ML")
    test_train_model()
