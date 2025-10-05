#!/usr/bin/env python
"""
Script de inicio que ejecuta Django y muestra logs en tiempo real
"""
import subprocess
import sys
import os
import time
import threading
import queue

def print_banner():
    """Imprime el banner de inicio"""
    print("🚀 ML BACKEND - SERVIDOR DJANGO CON LOGS")
    print("=" * 50)
    print("Iniciando servidor Django con logging detallado...")
    print("Los logs aparecerán en tiempo real en esta consola")
    print("=" * 50)
    print()

def run_django_server():
    """Ejecuta el servidor Django"""
    try:
        # Cambiar al directorio del proyecto
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Ejecutar Django
        print("🔧 Iniciando servidor Django...")
        process = subprocess.Popen(
            [sys.executable, "manage.py", "runserver"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Leer output en tiempo real
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[DJANGO] {line.rstrip()}")
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo servidor...")
        if 'process' in locals():
            process.terminate()
        print("✅ Servidor detenido")
    except Exception as e:
        print(f"❌ Error ejecutando Django: {e}")

def check_log_file():
    """Monitorea el archivo de logs en tiempo real"""
    log_file = "ml_backend.log"
    
    if not os.path.exists(log_file):
        print(f"📝 Archivo de logs no encontrado: {log_file}")
        return
    
    print(f"📝 Monitoreando archivo de logs: {log_file}")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            # Ir al final del archivo
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                if line:
                    print(f"[LOG] {line.rstrip()}")
                else:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo monitoreo de logs...")
    except Exception as e:
        print(f"❌ Error monitoreando logs: {e}")

def main():
    """Función principal"""
    print_banner()
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists("manage.py"):
        print("❌ Error: No se encontró manage.py")
        print("   Asegúrate de ejecutar este script desde el directorio ml_backend")
        sys.exit(1)
    
    # Verificar que Django esté instalado
    try:
        import django
        print(f"✅ Django {django.get_version()} encontrado")
    except ImportError:
        print("❌ Error: Django no está instalado")
        print("   Ejecuta: pip install -r requirements.txt")
        sys.exit(1)
    
    print("🌐 El servidor estará disponible en: http://localhost:8000")
    print("📊 API disponible en: http://localhost:8000/api/")
    print("🔧 Panel de administración: http://localhost:8000/admin/")
    print("📝 Logs guardados en: ml_backend.log")
    print()
    print("💡 Presiona Ctrl+C para detener el servidor")
    print("=" * 50)
    print()
    
    try:
        # Ejecutar Django
        run_django_server()
    except KeyboardInterrupt:
        print("\n✅ Servidor detenido por el usuario")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    main()

