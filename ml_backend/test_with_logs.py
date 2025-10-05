#!/usr/bin/env python
"""
Script de prueba mejorado para verificar que la API funciona correctamente
con logs detallados
"""
import requests
import json
import time
import sys

# URL base de la API
BASE_URL = "http://localhost:8000/api"

def print_separator(title):
    """Imprime un separador visual"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_health_check():
    """Verifica que el servidor esté ejecutándose"""
    print_separator("VERIFICACIÓN DE SALUD DEL SERVIDOR")
    
    try:
        print("🔍 Verificando conexión con el servidor...")
        response = requests.get(f"{BASE_URL}/models/", timeout=10)
        
        if response.status_code == 200:
            print("✅ Servidor Django está ejecutándose correctamente")
            print(f"   URL: {BASE_URL}")
            return True
        else:
            print(f"❌ Servidor respondió con código: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se puede conectar al servidor Django")
        print("   Asegúrate de que el servidor esté ejecutándose:")
        print("   cd ml_backend && python manage.py runserver")
        return False
    except requests.exceptions.Timeout:
        print("❌ Error: Timeout al conectar con el servidor")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_train_model():
    """Prueba el endpoint de entrenamiento con logs detallados"""
    print_separator("PRUEBA DE ENTRENAMIENTO DE MODELO")
    
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
    
    print("📊 Datos de prueba:")
    print(json.dumps(test_data, indent=2))
    print("\n🚀 Enviando petición de entrenamiento...")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/train-model/", json=test_data, timeout=60)
        end_time = time.time()
        
        print(f"⏱️ Tiempo de respuesta: {end_time - start_time:.2f} segundos")
        print(f"📡 Código de respuesta: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Entrenamiento exitoso!")
            print(f"   🆔 Modelo ID: {result.get('model_id')}")
            print(f"   📈 Accuracy: {result.get('accuracy'):.3f}")
            print(f"   🎯 Precision: {result.get('precision'):.3f}")
            print(f"   🔄 Recall: {result.get('recall'):.3f}")
            print(f"   ⚖️ F1-Score: {result.get('f1_score'):.3f}")
            print(f"   📊 MAE: {result.get('mae'):.3f}")
            print(f"   📈 R²: {result.get('r2_score'):.3f}")
            print(f"   ⏱️ Tiempo de entrenamiento: {result.get('training_time')}")
            
            # Verificar gráficos
            plots = result.get('plots', {})
            if plots:
                print(f"   🎨 Gráficos generados: {len(plots)}")
                for plot_name, plot_data in plots.items():
                    if plot_data:
                        print(f"      - {plot_name}: ✅ Generado")
                    else:
                        print(f"      - {plot_name}: ❌ No generado")
            else:
                print("   🎨 Gráficos: ❌ No generados")
            
            return result.get('model_id')
        else:
            print(f"\n❌ Error en entrenamiento: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Error desconocido')}")
                if 'details' in error_data:
                    print(f"   Detalles: {error_data['details']}")
            except:
                print(f"   Respuesta: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Error: Timeout en la petición de entrenamiento")
        return None
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return None

def test_predict(model_id):
    """Prueba el endpoint de predicción con logs detallados"""
    if not model_id:
        print_separator("PRUEBA DE PREDICCIÓN - OMITIDA")
        print("⚠️ No hay modelo para probar predicción")
        return
    
    print_separator("PRUEBA DE PREDICCIÓN")
    
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
    
    print(f"🔮 Probando predicción con modelo ID: {model_id}")
    print("📊 Datos de entrada:")
    print(json.dumps(test_input, indent=2))
    print("\n🚀 Enviando petición de predicción...")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/predict/", json=test_input, timeout=30)
        end_time = time.time()
        
        print(f"⏱️ Tiempo de respuesta: {end_time - start_time:.2f} segundos")
        print(f"📡 Código de respuesta: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Predicción exitosa!")
            print(f"   🆔 Predicción ID: {result.get('prediction_id')}")
            print(f"   🔮 Predicción: {result.get('prediction')}")
            print(f"   🤖 Modelo: {result.get('model_name')}")
            print(f"   📈 Accuracy del modelo: {result.get('model_accuracy'):.3f}")
        else:
            print(f"\n❌ Error en predicción: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Error desconocido')}")
                if 'details' in error_data:
                    print(f"   Detalles: {error_data['details']}")
            except:
                print(f"   Respuesta: {response.text}")
                
    except requests.exceptions.Timeout:
        print("❌ Error: Timeout en la petición de predicción")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

def test_list_models():
    """Prueba el endpoint de listar modelos con logs detallados"""
    print_separator("PRUEBA DE LISTADO DE MODELOS")
    
    print("📋 Obteniendo lista de modelos...")
    
    try:
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/models/", timeout=10)
        end_time = time.time()
        
        print(f"⏱️ Tiempo de respuesta: {end_time - start_time:.2f} segundos")
        print(f"📡 Código de respuesta: {response.status_code}")
        
        if response.status_code == 200:
            models = response.json()
            print(f"\n✅ Lista de modelos obtenida exitosamente!")
            print(f"   📊 Total de modelos: {len(models)}")
            
            if models:
                print("\n📋 Modelos disponibles:")
                for i, model in enumerate(models, 1):
                    print(f"   {i}. {model['name']}")
                    print(f"      - Tipo: {model['model_type']}")
                    print(f"      - Análisis: {model['analysis_type']}")
                    print(f"      - Accuracy: {model['accuracy']:.3f}")
                    print(f"      - Creado: {model['created_at']}")
                    print()
            else:
                print("   📭 No hay modelos disponibles")
        else:
            print(f"\n❌ Error listando modelos: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Error desconocido')}")
            except:
                print(f"   Respuesta: {response.text}")
                
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

def main():
    """Función principal de prueba con logs detallados"""
    print("🚀 INICIANDO PRUEBAS DETALLADAS DE LA API ML BACKEND")
    print("=" * 60)
    print("Este script probará todos los endpoints de la API")
    print("y mostrará logs detallados de cada operación.")
    print("=" * 60)
    
    # Verificar salud del servidor
    if not test_health_check():
        print("\n❌ No se puede continuar sin conexión al servidor")
        sys.exit(1)
    
    # Probar entrenamiento
    model_id = test_train_model()
    
    # Probar predicción
    test_predict(model_id)
    
    # Probar listar modelos
    test_list_models()
    
    print_separator("PRUEBAS COMPLETADAS")
    print("✅ Todas las pruebas han sido ejecutadas")
    print("📝 Revisa los logs del servidor Django para más detalles")
    print("💡 Si hay errores, revisa el archivo ml_backend.log")

if __name__ == "__main__":
    main()

