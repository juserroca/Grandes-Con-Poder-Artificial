#!/usr/bin/env python
"""
Script para verificar que el entorno de debugging esté configurado correctamente
"""
import os
import sys
import subprocess

def check_python_version():
    """Verificar versión de Python"""
    print("Verificando version de Python...")
    version = sys.version_info
    print(f"   Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ERROR: Se requiere Python 3.8 o superior")
        return False
    else:
        print("   OK: Version de Python compatible")
        return True

def check_venv():
    """Verificar entorno virtual"""
    print("\nVerificando entorno virtual...")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("   OK: Entorno virtual detectado")
        print(f"   Python ejecutable: {sys.executable}")
        print(f"   Directorio de trabajo: {os.getcwd()}")
        return True
    else:
        print("   ERROR: No se detecto entorno virtual activo")
        print("   Ejecuta: venv\\Scripts\\activate (Windows) o source venv/bin/activate (Linux/Mac)")
        return False

def check_dependencies():
    """Verificar dependencias instaladas"""
    print("\n📦 Verificando dependencias...")
    
    required_packages = [
        'django',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'djangorestframework',
        'django-cors-headers'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'django-cors-headers':
                __import__('corsheaders')
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NO INSTALADO")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   💡 Instala las dependencias faltantes:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_django_setup():
    """Verificar configuración de Django"""
    print("\n⚙️  Verificando configuración de Django...")
    
    try:
        import django
        from django.conf import settings
        
        print(f"   ✅ Django versión: {django.get_version()}")
        print(f"   ✅ DEBUG: {settings.DEBUG}")
        print(f"   ✅ INSTALLED_APPS: {len(settings.INSTALLED_APPS)} aplicaciones")
        
        # Verificar que ml_models esté en INSTALLED_APPS
        if 'ml_models' in settings.INSTALLED_APPS:
            print("   ✅ ml_models en INSTALLED_APPS")
        else:
            print("   ❌ ml_models NO está en INSTALLED_APPS")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR configurando Django: {e}")
        return False

def check_database():
    """Verificar base de datos"""
    print("\n🗄️  Verificando base de datos...")
    
    try:
        import django
        from django.core.management import execute_from_command_line
        
        # Ejecutar migraciones
        result = subprocess.run([
            sys.executable, 'manage.py', 'migrate', '--check'
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("   ✅ Base de datos actualizada")
            return True
        else:
            print("   ⚠️  Base de datos necesita migraciones")
            print("   💡 Ejecuta: python manage.py migrate")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR verificando base de datos: {e}")
        return False

def main():
    """Función principal de verificación"""
    print("=" * 50)
    print("    VERIFICACIÓN DEL ENTORNO DE DEBUGGING")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_venv(),
        check_dependencies(),
        check_django_setup(),
        check_database()
    ]
    
    print("\n" + "=" * 50)
    print("    RESUMEN DE VERIFICACIÓN")
    print("=" * 50)
    
    if all(checks):
        print("✅ TODAS LAS VERIFICACIONES PASARON")
        print("🚀 El entorno está listo para debugging")
        print("\n💡 Comandos disponibles:")
        print("   - run_debug.bat (Windows)")
        print("   - ./run_debug.sh (Linux/Mac)")
        print("   - python debug_server.py")
        print("   - python debug_with_breakpoints.py")
    else:
        print("❌ ALGUNAS VERIFICACIONES FALLARON")
        print("🔧 Corrige los errores antes de continuar")
    
    print("=" * 50)

if __name__ == '__main__':
    main()
