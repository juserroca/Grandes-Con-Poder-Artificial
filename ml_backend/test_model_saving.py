#!/usr/bin/env python3
"""
Script para probar el guardado y carga de modelos .pkl
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Agregar el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_saving():
    """Prueba el guardado y carga de modelos"""
    
    print("🧪 Probando guardado y carga de modelos .pkl...")
    
    try:
        from ml_models.ml_engine import MLEngine
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # Crear datos de prueba
        print("📊 Generando datos de prueba...")
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Crear scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar modelo
        print("🤖 Entrenando modelo de prueba...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Crear MLEngine y configurar
        ml_engine = MLEngine()
        ml_engine.model = model
        ml_engine.scaler = scaler
        ml_engine.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        ml_engine.target_name = 'target'
        
        # Crear directorio de modelos
        model_dir = "media/models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Guardar modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file_path = f"models/test_model_{timestamp}.pkl"
        full_path = f"media/{model_file_path}"
        
        print(f"💾 Guardando modelo en: {full_path}")
        ml_engine.save_model(full_path)
        
        # Verificar que el archivo existe
        if os.path.exists(full_path):
            print("✅ Archivo del modelo guardado exitosamente")
            file_size = os.path.getsize(full_path)
            print(f"📁 Tamaño del archivo: {file_size} bytes")
        else:
            print("❌ Error: Archivo no encontrado")
            return False
        
        # Cargar modelo
        print("📂 Cargando modelo...")
        ml_engine_loaded = MLEngine()
        ml_engine_loaded.load_model(full_path)
        
        # Verificar que el modelo se cargó correctamente
        if ml_engine_loaded.model is not None:
            print("✅ Modelo cargado exitosamente")
        else:
            print("❌ Error: Modelo no se cargó")
            return False
        
        # Verificar scaler
        if ml_engine_loaded.scaler is not None:
            print("✅ Scaler cargado exitosamente")
        else:
            print("❌ Error: Scaler no se cargó")
            return False
        
        # Verificar feature names
        if ml_engine_loaded.feature_names is not None:
            print(f"✅ Feature names cargados: {len(ml_engine_loaded.feature_names)}")
        else:
            print("❌ Error: Feature names no se cargaron")
            return False
        
        # Probar predicción con el modelo cargado
        print("🔮 Probando predicción con modelo cargado...")
        y_pred_original = model.predict(X_test_scaled)
        y_pred_loaded = ml_engine_loaded.model.predict(X_test_scaled)
        
        # Verificar que las predicciones son iguales
        if np.array_equal(y_pred_original, y_pred_loaded):
            print("✅ Predicciones idénticas entre modelo original y cargado")
        else:
            print("❌ Error: Predicciones diferentes")
            return False
        
        # Probar probabilidades
        y_proba_original = model.predict_proba(X_test_scaled)
        y_proba_loaded = ml_engine_loaded.model.predict_proba(X_test_scaled)
        
        if np.allclose(y_proba_original, y_proba_loaded):
            print("✅ Probabilidades idénticas entre modelo original y cargado")
        else:
            print("❌ Error: Probabilidades diferentes")
            return False
        
        print("\n🎉 Todas las pruebas de guardado y carga exitosas!")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_model_structure():
    """Prueba la estructura del archivo .pkl guardado"""
    
    print("\n" + "="*60)
    print("🔍 PROBANDO ESTRUCTURA DEL ARCHIVO .PKL")
    print("="*60)
    
    try:
        import pickle
        
        # Buscar archivos .pkl en el directorio de modelos
        model_dir = "media/models"
        if not os.path.exists(model_dir):
            print("❌ Directorio de modelos no existe")
            return False
        
        pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        
        if not pkl_files:
            print("❌ No se encontraron archivos .pkl")
            return False
        
        # Probar con el primer archivo encontrado
        test_file = os.path.join(model_dir, pkl_files[0])
        print(f"📁 Probando archivo: {test_file}")
        
        # Cargar y examinar el contenido
        with open(test_file, 'rb') as f:
            model_data = pickle.load(f)
        
        print("📋 Estructura del archivo .pkl:")
        for key, value in model_data.items():
            if hasattr(value, '__class__'):
                print(f"  {key}: {type(value).__name__}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
        
        # Verificar que tiene todos los componentes necesarios
        required_keys = ['model', 'scaler', 'label_encoders', 'feature_names', 'target_name']
        missing_keys = [key for key in required_keys if key not in model_data]
        
        if missing_keys:
            print(f"⚠️ Claves faltantes: {missing_keys}")
        else:
            print("✅ Todas las claves requeridas presentes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error examinando archivo .pkl: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Iniciando pruebas de guardado de modelos...")
    
    # Probar guardado y carga
    success1 = test_model_saving()
    
    # Probar estructura del archivo
    success2 = test_model_structure()
    
    if success1 and success2:
        print("\n🎉 Todas las pruebas completadas exitosamente!")
        print("💡 El guardado de modelos .pkl está funcionando correctamente")
    else:
        print("\n💡 Revisar la implementación del guardado de modelos")
