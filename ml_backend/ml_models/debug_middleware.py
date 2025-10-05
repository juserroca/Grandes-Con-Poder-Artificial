"""
Middleware personalizado para debugging de Django
"""
import json
import time
import logging
from django.utils.deprecation import MiddlewareMixin
from django.http import JsonResponse

logger = logging.getLogger('debug')

class DebugMiddleware(MiddlewareMixin):
    """
    Middleware para debugging de peticiones HTTP
    """
    
    def process_request(self, request):
        """Procesar petición entrante"""
        request.start_time = time.time()
        
        # Log de la petición
        logger.debug(f"🚀 REQUEST: {request.method} {request.path}")
        logger.debug(f"📋 Headers: {dict(request.headers)}")
        
        # Log del body si es POST/PUT
        if request.method in ['POST', 'PUT', 'PATCH']:
            try:
                if hasattr(request, 'body') and request.body:
                    body_data = request.body.decode('utf-8')
                    logger.debug(f"📦 Body: {body_data[:1000]}...")  # Primeros 1000 caracteres
            except Exception as e:
                logger.debug(f"❌ Error leyendo body: {e}")
    
    def process_response(self, request, response):
        """Procesar respuesta saliente"""
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            logger.debug(f"⏱️  RESPONSE: {response.status_code} en {duration:.3f}s")
        
        # Log de la respuesta si es JSON
        if isinstance(response, JsonResponse):
            try:
                response_data = json.loads(response.content.decode('utf-8'))
                logger.debug(f"📤 Response data: {response_data}")
            except Exception as e:
                logger.debug(f"❌ Error leyendo response: {e}")
        
        return response
    
    def process_exception(self, request, exception):
        """Procesar excepciones"""
        logger.error(f"💥 EXCEPTION: {type(exception).__name__}: {str(exception)}")
        logger.error(f"📍 Path: {request.path}")
        logger.error(f"🔍 Method: {request.method}")
        
        # Log del traceback completo
        import traceback
        logger.error(f"📚 Traceback: {traceback.format_exc()}")
        
        return None
