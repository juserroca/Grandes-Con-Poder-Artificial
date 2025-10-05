"""
Configuración específica para debugging
"""
from .settings import *

# Configuración adicional para debugging
DEBUG = True

# Logging más detallado para debugging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
        'debug': {
            'format': '🐛 {levelname} {asctime} {name} - {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'debug',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'debug_detailed.log',
            'formatter': 'verbose',
        },
        'debug_file': {
            'class': 'logging.FileHandler',
            'filename': 'debug_breakpoints.log',
            'formatter': 'debug',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'ml_models': {
            'handlers': ['console', 'file', 'debug_file'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'debug': {
            'handlers': ['console', 'debug_file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'debug_breakpoints': {
            'handlers': ['console', 'debug_file'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}

# Configuración de base de datos para debugging (usar SQLite en memoria)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
}

# Configuración de CORS más permisiva para debugging
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

# Configuración de archivos estáticos para debugging
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Configuración de media para debugging
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
