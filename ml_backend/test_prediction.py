#!/usr/bin/env python3
"""
Script para probar el endpoint de predicción
"""

import requests
import json

def test_prediction_endpoint():
    """Prueba el endpoint de predicción con datos de ejemplo"""
    
    print("🧪 Probando endpoint de predicción...")
    
    # URL del endpoint
    url = "http://localhost:8000/api/predict/"
    
    # Datos de prueba (simulando datos de exoplaneta)
    test_data = {
        "model_id": 1,
        "input_data": {
            "koi_period": 365.25,      # Período orbital en días
            "koi_impact": 0.1,         # Impacto
            "koi_duration": 8.0,        # Duración en horas
            "koi_depth": 0.01,          # Profundidad
            "koi_prad": 1.0,            # Radio planetario
            "koi_teq": 288.0,           # Temperatura de equilibrio
            "koi_insol": 1.0,           # Insolación
            "koi_model_snr": 10.0,      # Signal-to-noise ratio
            "koi_slogg": 4.4,          # Log de gravedad estelar
            "koi_srad": 1.0             # Radio estelar
        }
    }
    
    print(f"📊 Datos de prueba:")
    for key, value in test_data["input_data"].items():
        print(f"  {key}: {value}")
    
    try:
        # Realizar petición
        print(f"\n🌐 Enviando petición a: {url}")
        response = requests.post(
            url,
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Predicción exitosa!")
            print(f"📈 Resultado: {result}")
            
            if 'prediction' in result:
                prediction_value = result['prediction']
                if prediction_value > 0.5:
                    print(f"🔮 Predicción: Es exoplaneta ({prediction_value:.3f})")
                else:
                    print(f"🔮 Predicción: No es exoplaneta ({prediction_value:.3f})")
            
            return True
            
        else:
            print(f"❌ Error en la predicción:")
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
        print("❌ Timeout: La petición tardó demasiado")
        return False
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_multiple_predictions():
    """Prueba múltiples predicciones con diferentes datos"""
    
    print("\n" + "="*60)
    print("🧪 PROBANDO MÚLTIPLES PREDICCIONES")
    print("="*60)
    
    test_cases = [
        {
            "name": "Exoplaneta típico",
            "data": {
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
        },
        {
            "name": "Candidato débil",
            "data": {
                "koi_period": 100.0,
                "koi_impact": 0.5,
                "koi_duration": 2.0,
                "koi_depth": 0.001,
                "koi_prad": 0.5,
                "koi_teq": 500.0,
                "koi_insol": 2.0,
                "koi_model_snr": 3.0,
                "koi_slogg": 4.0,
                "koi_srad": 0.8
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Caso {i}: {test_case['name']} ---")
        
        test_data = {
            "model_id": 1,
            "input_data": test_case["data"]
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
                prediction = result.get('prediction', 0)
                print(f"✅ Predicción: {prediction:.3f}")
                if prediction > 0.5:
                    print(f"🔮 Resultado: Es exoplaneta ({prediction*100:.1f}%)")
                else:
                    print(f"🔮 Resultado: No es exoplaneta ({(1-prediction)*100:.1f}%)")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Iniciando pruebas de predicción...")
    
    # Probar predicción básica
    success = test_prediction_endpoint()
    
    if success:
        # Probar múltiples casos
        test_multiple_predictions()
        print("\n🎉 Todas las pruebas completadas!")
    else:
        print("\n💡 Asegúrate de que:")
        print("   1. El servidor Django esté ejecutándose (python manage.py runserver)")
        print("   2. Exista un modelo entrenado con ID 1")
        print("   3. El endpoint /api/predict/ esté disponible")
