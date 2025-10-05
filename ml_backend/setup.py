#!/usr/bin/env python
"""
Script de configuración inicial para el proyecto Django ML Backend
"""
import os
import sys
import django
from django.core.management import execute_from_command_line

def setup_project():
    """Configura el proyecto Django inicial"""
    
    # Configurar Django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_backend.settings')
    django.setup()
    
    print("🚀 Configurando proyecto Django ML Backend...")
    
    # Crear migraciones
    print("📝 Creando migraciones...")
    execute_from_command_line(['manage.py', 'makemigrations'])
    
    # Aplicar migraciones
    print("🗄️ Aplicando migraciones...")
    execute_from_command_line(['manage.py', 'migrate'])
    
    # Crear superusuario (opcional)
    print("👤 Creando superusuario...")
    try:
        from django.contrib.auth.models import User
        if not User.objects.filter(username='admin').exists():
            User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
            print("✅ Superusuario creado: admin/admin123")
        else:
            print("ℹ️ Superusuario ya existe")
    except Exception as e:
        print(f"⚠️ Error creando superusuario: {e}")
    
    print("✅ Configuración completada!")
    print("\n📋 Para ejecutar el servidor:")
    print("   python manage.py runserver")
    print("\n🌐 El servidor estará disponible en: http://localhost:8000")
    print("📊 Panel de administración: http://localhost:8000/admin")

if __name__ == '__main__':
    setup_project()
