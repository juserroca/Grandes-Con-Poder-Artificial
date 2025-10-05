#!/usr/bin/env python
"""
Script de prueba para verificar que la API funciona correctamente
"""
import requests
import json
import time

# URL base de la API
BASE_URL = "http://localhost:8000/api"

def test_train_model():
    """Prueba el endpoint de entrenamiento"""
    print("🧪 Probando endpoint de entrenamiento...")
    
    # Datos de prueba
    test_data = {
        "analysis_type": "app-data",
        "model": "random-forest",
        "hyperparameters": {
            "n_estimators": 50,
            "max_depth": 5,
            "min_samples_split": 2
        },
        "dataset_name": "kepler"
    }
    
    try:
        print('prueba 2')
        response = requests.post(f"{BASE_URL}/train-model/", json=test_data)
        print(response.json())
        if response.status_code == 200:
            result = response.json()
            print("✅ Entrenamiento exitoso!")
            print(f"   Modelo ID: {result.get('model_id')}")
            print(f"   Accuracy: {result.get('accuracy'):.3f}")
            print(f"   Tiempo: {result.get('training_time')}")
            return result.get('model_id')
        else:
            print(f"❌ Error en entrenamiento: {response.status_code}")
            print(f"   Respuesta: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return None

def test_predict(model_id):
    """Prueba el endpoint de predicción"""
    if not model_id:
        print("⚠️ No hay modelo para probar predicción")
        return
    
    print(f"🧪 Probando endpoint de predicción con modelo {model_id}...")
    
    # Datos de prueba para predicción
    test_input = {
        "model_id": model_id,
        "input_data": {
            "koi_period": 365.25,
            "koi_impact": 0.1,
            "koi_duration": 8.0,
            "koi_depth": 100,
            "koi_prad": 1.0,
            "koi_teq": 288,
            "koi_insol": 1.0,
            "koi_steff": 5778,
            "koi_slogg": 4.4,
            "koi_srad": 1.0
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict/", json=test_input)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Predicción exitosa!")
            print(f"   Predicción: {result.get('prediction')}")
            print(f"   Modelo: {result.get('model_name')}")
            print(f"   Accuracy: {result.get('model_accuracy'):.3f}")
        else:
            print(f"❌ Error en predicción: {response.status_code}")
            print(f"   Respuesta: {response.text}")
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")

def test_list_models():
    """Prueba el endpoint de listar modelos"""
    print("🧪 Probando endpoint de listar modelos...")
    
    try:
        response = requests.get(f"{BASE_URL}/models/")
        
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Lista de modelos exitosa! ({len(models)} modelos)")
            for model in models:
                print(f"   - {model['name']} ({model['model_type']}) - {model['accuracy']:.3f}")
        else:
            print(f"❌ Error listando modelos: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")

def main():
    """Función principal de prueba"""
    print("🚀 Iniciando pruebas de la API ML Backend...")
    print("=" * 50)
    
    # Verificar que el servidor esté ejecutándose
    try:
        response = requests.get(f"{BASE_URL}/models/", timeout=5)
        print("✅ Servidor Django está ejecutándose")
    except:
        print("❌ Error: El servidor Django no está ejecutándose")
        print("   Ejecuta: python manage.py runserver")
        return
    
    print()
    
    # Probar entrenamiento
    print('prueba')
    model_id = test_train_model()
    print()
    
    # Probar predicción
    test_predict(model_id)
    print()
    
    # Probar listar modelos
    test_list_models()
    print()
    
    print("=" * 50)
    print("✅ Pruebas completadas!")

if __name__ == "__main__":
    main()
