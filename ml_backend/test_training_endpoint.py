#!/usr/bin/env python3
"""
Script para probar el endpoint de entrenamiento con guardado de modelo .pkl
"""

import requests
import json
import pandas as pd
import numpy as np

def test_training_endpoint():
    """Prueba el endpoint de entrenamiento completo"""
    
    print("🧪 Probando endpoint de entrenamiento con guardado de modelo...")
    
    # URL del endpoint
    url = "http://localhost:8000/api/train-model/"
    
    # Crear datos de prueba
    print("📊 Generando datos de prueba...")
    np.random.seed(42)
    n_samples = 1000
    
    # Simular datos de exoplanetas
    data = {
        'koi_period': np.random.uniform(1, 1000, n_samples),
        'koi_impact': np.random.uniform(0, 1, n_samples),
        'koi_duration': np.random.uniform(0.1, 24, n_samples),
        'koi_depth': np.random.uniform(0.001, 0.1, n_samples),
        'koi_prad': np.random.uniform(0.1, 10, n_samples),
        'koi_teq': np.random.uniform(200, 2000, n_samples),
        'koi_insol': np.random.uniform(0.1, 10, n_samples),
        'koi_model_snr': np.random.uniform(1, 50, n_samples),
        'koi_slogg': np.random.uniform(3, 5, n_samples),
        'koi_srad': np.random.uniform(0.1, 5, n_samples),
        'koi_disposition': np.random.choice(['CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE'], n_samples, p=[0.3, 0.5, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Preparar datos para el endpoint
    training_data = {
        'analysis_type': 'own-data',
        'model': 'random-forest',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        },
        'target_variable': 'koi_disposition',
        'input_variables': [
            'koi_period', 'koi_impact', 'koi_duration', 'koi_depth',
            'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
            'koi_slogg', 'koi_srad'
        ],
        'csv_data': df.values.tolist(),
        'csv_columns': df.columns.tolist(),
        'column_types': {
            'koi_period': 'float',
            'koi_impact': 'float',
            'koi_duration': 'float',
            'koi_depth': 'float',
            'koi_prad': 'float',
            'koi_teq': 'float',
            'koi_insol': 'float',
            'koi_model_snr': 'float',
            'koi_slogg': 'float',
            'koi_srad': 'float',
            'koi_disposition': 'string'
        },
        'file_name': 'test_exoplanets.csv'
    }
    
    print(f"📊 Datos preparados: {len(df)} filas, {len(df.columns)} columnas")
    print(f"🎯 Variable objetivo: {training_data['target_variable']}")
    print(f"📋 Variables de entrada: {len(training_data['input_variables'])}")
    
    try:
        # Realizar petición
        print(f"\n🌐 Enviando petición a: {url}")
        response = requests.post(
            url,
            json=training_data,
            headers={'Content-Type': 'application/json'},
            timeout=120  # 2 minutos de timeout
        )
        
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Entrenamiento exitoso!")
            print(f"📈 Resultado: {result}")
            
            # Verificar que se guardó el modelo
            if 'model_id' in result:
                print(f"🆔 ID del modelo: {result['model_id']}")
            
            if 'model_file' in result:
                print(f"💾 Archivo del modelo: {result['model_file']}")
                
                # Verificar que el archivo existe
                import os
                model_path = f"media/{result['model_file']}"
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path)
                    print(f"✅ Archivo del modelo existe: {file_size} bytes")
                else:
                    print(f"❌ Archivo del modelo no encontrado: {model_path}")
            
            if 'accuracy' in result:
                print(f"🎯 Accuracy: {result['accuracy']:.4f}")
            
            return True
            
        else:
            print(f"❌ Error en el entrenamiento:")
            print(f"   Status: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Error de conexión: ¿Está el servidor Django ejecutándose?")
        print("💡 Ejecuta: python manage.py runserver")
        return False
        
    except requests.exceptions.Timeout:
        print("❌ Timeout: El entrenamiento tardó demasiado")
        return False
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_prediction_with_saved_model():
    """Prueba la predicción con el modelo guardado"""
    
    print("\n" + "="*60)
    print("🔮 PROBANDO PREDICCIÓN CON MODELO GUARDADO")
    print("="*60)
    
    # Datos de prueba para predicción
    test_data = {
        "model_id": 1,  # Asumir que el modelo tiene ID 1
        "input_data": {
            "koi_period": 365.25,
            "koi_impact": 0.1,
            "koi_duration": 8.0,
            "koi_depth": 0.01,
            "koi_prad": 1.0,
            "koi_teq": 288.0,
            "koi_insol": 1.0,
            "koi_model_snr": 10.0,
            "koi_slogg": 4.4,
            "koi_srad": 1.0
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/predict/",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Predicción exitosa con modelo guardado!")
            print(f"🔮 Resultado: {result}")
            return True
        else:
            print(f"❌ Error en predicción: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Iniciando pruebas del endpoint de entrenamiento...")
    
    # Probar entrenamiento
    success1 = test_training_endpoint()
    
    if success1:
        # Probar predicción
        success2 = test_prediction_with_saved_model()
        
        if success2:
            print("\n🎉 Todas las pruebas completadas exitosamente!")
            print("💡 El guardado de modelos .pkl está funcionando correctamente")
        else:
            print("\n⚠️ Entrenamiento exitoso, pero predicción falló")
    else:
        print("\n💡 Revisar la implementación del entrenamiento y guardado")
