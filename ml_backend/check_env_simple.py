#!/usr/bin/env python
"""
Script simple para verificar el entorno de debugging
"""
import os
import sys

def main():
    print("=" * 50)
    print("    VERIFICACION DEL ENTORNO DE DEBUGGING")
    print("=" * 50)
    
    # Verificar Python
    print("\n1. Verificando Python...")
    version = sys.version_info
    print(f"   Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ERROR: Se requiere Python 3.8 o superior")
        return False
    else:
        print("   OK: Version compatible")
    
    # Verificar entorno virtual
    print("\n2. Verificando entorno virtual...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("   OK: Entorno virtual detectado")
        print(f"   Python: {sys.executable}")
        print(f"   Directorio: {os.getcwd()}")
    else:
        print("   ERROR: No se detecto entorno virtual")
        print("   Solucion: venv\\Scripts\\activate (Windows)")
        return False
    
    # Verificar Django
    print("\n3. Verificando Django...")
    try:
        import django
        print(f"   OK: Django {django.get_version()}")
    except ImportError:
        print("   ERROR: Django no instalado")
        print("   Solucion: pip install django")
        return False
    
    # Verificar pandas
    print("\n4. Verificando pandas...")
    try:
        import pandas
        print(f"   OK: Pandas {pandas.__version__}")
    except ImportError:
        print("   ERROR: Pandas no instalado")
        print("   Solucion: pip install pandas")
        return False
    
    # Verificar scikit-learn
    print("\n5. Verificando scikit-learn...")
    try:
        import sklearn
        print(f"   OK: Scikit-learn {sklearn.__version__}")
    except ImportError:
        print("   ERROR: Scikit-learn no instalado")
        print("   Solucion: pip install scikit-learn")
        return False
    
    print("\n" + "=" * 50)
    print("    RESUMEN")
    print("=" * 50)
    print("OK: Entorno listo para debugging")
    print("\nComandos disponibles:")
    print("  - run_debug.bat (Windows)")
    print("  - python debug_server.py")
    print("  - python debug_with_breakpoints.py")
    print("=" * 50)
    
    return True

if __name__ == '__main__':
    main()
