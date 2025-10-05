#!/usr/bin/env python3
"""
Script para probar el balanceo de clases usando sklearn
"""

import pandas as pd
import numpy as np
import sys
import os

# Agregar el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_balanceo_sklearn():
    """Prueba el balanceo de clases usando sklearn"""
    
    print("Probando balanceo de clases con sklearn...")
    
    # Crear datos desbalanceados de prueba
    np.random.seed(42)
    n_samples_0 = 1000  # Clase mayoritaria
    n_samples_1 = 100   # Clase minoritaria
    
    # Generar datos para clase 0
    X_0 = np.random.normal(0, 1, (n_samples_0, 3))
    y_0 = np.zeros(n_samples_0)
    
    # Generar datos para clase 1
    X_1 = np.random.normal(2, 1, (n_samples_1, 3))
    y_1 = np.ones(n_samples_1)
    
    # Combinar datos
    X = np.vstack([X_0, X_1])
    y = np.hstack([y_0, y_1])
    
    # Convertir a DataFrame
    X_df = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3'])
    y_series = pd.Series(y, name='target')
    
    print(f"Datos originales:")
    print(f"  Clase 0: {(y == 0).sum()} ({(y == 0).mean():.2%})")
    print(f"  Clase 1: {(y == 1).sum()} ({(y == 1).mean():.2%})")
    
    # Importar la función de balanceo
    try:
        from ml_models.preprocesamiento import balancear_clases
        
        # Probar oversampling
        print("\n--- Probando Oversampling ---")
        X_bal, y_bal = balancear_clases(X_df, y_series, metodo='oversample')
        
        print(f"\nDatos después del oversampling:")
        print(f"  Clase 0: {(y_bal == 0).sum()} ({(y_bal == 0).mean():.2%})")
        print(f"  Clase 1: {(y_bal == 1).sum()} ({(y_bal == 1).mean():.2%})")
        
        # Verificar que las clases están balanceadas
        assert abs((y_bal == 0).mean() - (y_bal == 1).mean()) < 0.1, "Las clases no están balanceadas"
        print("✅ Oversampling funcionando correctamente!")
        
        # Probar undersampling
        print("\n--- Probando Undersampling ---")
        X_bal2, y_bal2 = balancear_clases(X_df, y_series, metodo='undersample')
        
        print(f"\nDatos después del undersampling:")
        print(f"  Clase 0: {(y_bal2 == 0).sum()} ({(y_bal2 == 0).mean():.2%})")
        print(f"  Clase 1: {(y_bal2 == 1).sum()} ({(y_bal2 == 1).mean():.2%})")
        
        # Verificar que las clases están balanceadas
        assert abs((y_bal2 == 0).mean() - (y_bal2 == 1).mean()) < 0.1, "Las clases no están balanceadas"
        print("✅ Undersampling funcionando correctamente!")
        
        print("\n🎉 Todas las pruebas de balanceo exitosas!")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

if __name__ == "__main__":
    success = test_balanceo_sklearn()
    if not success:
        print("\n💡 Verificar que el archivo preprocesamiento.py esté en el directorio correcto")
        sys.exit(1)
    else:
        print("\n✅ Balanceo con sklearn listo para usar!")

