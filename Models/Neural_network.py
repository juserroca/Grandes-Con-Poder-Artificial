# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 17:01:47 2025

@author: juser
"""

"""
Clasificador de Exoplanetas usando Redes Neuronales - Versión Modular
Dataset: Kepler Exoplanet Search Results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import pre
warnings.filterwarnings('ignore')

# ============================================================================
# FUNCIONES DE CARGA Y PREPROCESAMIENTO
# ============================================================================

def preprocesar_datos(df):
    """Preprocesa y limpia los datos"""
    print("\n[2] Preprocesando datos...")
    
    # Crear variable objetivo
    df['es_exoplaneta'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)
    
    # Características importantes
    features = [
        'koi_period', 'koi_depth', 'koi_duration', 'koi_prad',
        'koi_teq', 'koi_insol', 'koi_steff', 'koi_srad'
    ]
    
    # Filtrar características disponibles
    available_features = [f for f in features if f in df.columns]
    print(f"✓ Características seleccionadas: {len(available_features)}")
    
    # Limpiar datos
    df_clean = df[available_features + ['es_exoplaneta']].dropna()
    print(f"✓ Datos limpios: {df_clean.shape[0]} objetos")
    
    # Separar X e y
    X = df_clean[available_features].values
    y = df_clean['es_exoplaneta'].values
    
    mostrar_distribucion_clases(y)
    
    return X, y, available_features


def mostrar_distribucion_clases(y):
    """Muestra la distribución de clases en el dataset"""
    print("✓ Distribución de clases:")
    print(f"  - Exoplanetas: {np.sum(y == 1)} ({100*np.mean(y):.1f}%)")
    print(f"  - No exoplanetas: {np.sum(y == 0)} ({100*np.mean(y == 0):.1f}%)")


# ============================================================================
# FUNCIONES DE DIVISIÓN Y NORMALIZACIÓN
# ============================================================================

def dividir_y_normalizar(X, y, test_size=0.2, random_state=42):
    """Divide los datos y normaliza las características"""
    print("\n[3] Dividiendo y normalizando datos...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================================================
# FUNCIONES DE CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO
# ============================================================================

def construir_modelo(input_dim):
    """Construye la arquitectura de la red neuronal"""
    print("\n[4] Construyendo arquitectura de red neuronal...")
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # Capa 1: Extracción de características
        layers.Dense(128, activation='relu', 
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Capa 2: Representación intermedia
        layers.Dense(64, activation='relu', 
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Capa 3: Abstracción
        layers.Dense(32, activation='relu', 
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.2),
        
        # Capa de salida
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


def configurar_callbacks():
    """Configura los callbacks para el entrenamiento"""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    )
    
    return [early_stopping, reduce_lr]


def entrenar_modelo(model, X_train, y_train, epochs=100, batch_size=32):
    """Entrena el modelo con los datos de entrenamiento"""
    print("\n[5] Entrenando modelo...")
    
    callbacks = configurar_callbacks()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )
    
    print(f"✓ Entrenamiento completado en {len(history.history['loss'])} épocas")
    
    return history


# ============================================================================
# FUNCIONES DE EVALUACIÓN Y PREDICCIÓN
# ============================================================================

def evaluar_modelo(model, X_test, y_test):
    """Evalúa el modelo y genera predicciones"""
    print("\n[6] Evaluando modelo...")
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, 
                              target_names=['No Exoplaneta', 'Exoplaneta']))
    
    return y_pred, y_pred_proba


def calcular_metricas(y_test, y_pred_proba):
    """Calcula métricas adicionales"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def graficar_entrenamiento(history, ax1, ax2):
    """Grafica las curvas de entrenamiento"""
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('Pérdida durante Entrenamiento', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Binary Crossentropy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
    ax2.set_title('Precisión durante Entrenamiento', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)


def graficar_matriz_confusion(y_test, y_pred, ax):
    """Grafica la matriz de confusión"""
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Exoplaneta', 'Exoplaneta'],
                yticklabels=['No Exoplaneta', 'Exoplaneta'], ax=ax)
    ax.set_title('Matriz de Confusión', fontsize=12, fontweight='bold')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicción')


def graficar_curva_roc(fpr, tpr, roc_auc, ax):
    """Grafica la curva ROC"""
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos')
    ax.set_ylabel('Tasa de Verdaderos Positivos')
    ax.set_title('Curva ROC', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


def graficar_distribucion_predicciones(y_test, y_pred_proba, ax):
    """Grafica la distribución de probabilidades predichas"""
    ax.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.6, 
            label='No Exoplanetas', color='red')
    ax.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.6, 
            label='Exoplanetas', color='blue')
    ax.set_xlabel('Probabilidad Predicha')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de Predicciones', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


def graficar_importancia_caracteristicas(model, feature_names, ax):
    """Grafica la importancia aproximada de las características"""
    weights = np.abs(model.layers[0].get_weights()[0]).mean(axis=1)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': weights
    }).sort_values('importance', ascending=True)
    
    ax.barh(feature_importance['feature'], feature_importance['importance'], 
            color='skyblue')
    ax.set_xlabel('Importancia Relativa')
    ax.set_title('Importancia de Características', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')


def generar_visualizaciones(history, y_test, y_pred, y_pred_proba, model, 
                           feature_names, roc_auc):
    """Genera todas las visualizaciones del modelo"""
    print("\n[7] Generando visualizaciones...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Configurar subplots
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5)
    ax6 = plt.subplot(2, 3, 6)
    
    # Generar gráficos
    graficar_entrenamiento(history, ax1, ax2)
    graficar_matriz_confusion(y_test, y_pred, ax3)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    graficar_curva_roc(fpr, tpr, roc_auc, ax4)
    graficar_distribucion_predicciones(y_test, y_pred_proba, ax5)
    graficar_importancia_caracteristicas(model, feature_names, ax6)
    
    plt.tight_layout()
    plt.savefig('exoplanet_classification_results.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# FUNCIONES DE GUARDADO Y CARGA
# ============================================================================

def guardar_modelo(model, scaler, feature_names, roc_auc, directorio='modelo_exoplanetas'):
    """
    Guarda el modelo, scaler y metadatos
    
    Args:
        model: modelo de Keras entrenado
        scaler: StandardScaler ajustado
        feature_names: lista de nombres de características
        roc_auc: métrica AUC-ROC del modelo
        directorio: carpeta donde guardar los archivos
    """
    import os
    import pickle
    import json
    from datetime import datetime
    
    print(f"\n[8] Guardando modelo en '{directorio}/'...")
    
    # Crear directorio si no existe
    os.makedirs(directorio, exist_ok=True)
    
    # Guardar modelo de Keras
    model_path = os.path.join(directorio, 'modelo_exoplaneta.keras')
    model.save(model_path)
    print(f"✓ Modelo guardado: {model_path}")
    
    # Guardar scaler
    scaler_path = os.path.join(directorio, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler guardado: {scaler_path}")
    
    # Guardar metadatos
    metadata = {
        'feature_names': feature_names,
        'roc_auc': float(roc_auc),
        'fecha_entrenamiento': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_features': len(feature_names)
    }
    metadata_path = os.path.join(directorio, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadatos guardados: {metadata_path}")
    
    print(f"\n✓ Modelo completo guardado exitosamente en '{directorio}/'")


def cargar_modelo(directorio='modelo_exoplanetas'):
    """
    Carga el modelo, scaler y metadatos previamente guardados
    
    Args:
        directorio: carpeta donde están los archivos
    
    Returns:
        model, scaler, feature_names, metadata
    """
    import os
    import pickle
    import json
    
    print(f"\nCargando modelo desde '{directorio}/'...")
    
    # Cargar modelo
    model_path = os.path.join(directorio, 'modelo_exoplaneta.keras')
    model = keras.models.load_model(model_path)
    print(f"✓ Modelo cargado: {model_path}")
    
    # Cargar scaler
    scaler_path = os.path.join(directorio, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler cargado: {scaler_path}")
    
    # Cargar metadatos
    metadata_path = os.path.join(directorio, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Metadatos cargados: {metadata_path}")
    
    feature_names = metadata['feature_names']
    print("\n✓ Modelo cargado exitosamente")
    print(f"  - AUC-ROC: {metadata['roc_auc']:.4f}")
    print(f"  - Fecha: {metadata['fecha_entrenamiento']}")
    print(f"  - Características: {metadata['n_features']}")
    
    return model, scaler, feature_names, metadata


# ============================================================================
# FUNCIÓN DE PREDICCIÓN
# ============================================================================

def crear_predictor(model, scaler, feature_names):
    """Crea una función de predicción reutilizable"""
    def predecir_exoplaneta(caracteristicas_dict):
        """
        Predice si un objeto es un exoplaneta
        
        Args:
            caracteristicas_dict: diccionario con las características del objeto
        
        Returns:
            probabilidad de ser exoplaneta
        """
        input_data = np.array([[caracteristicas_dict.get(f, 0) for f in feature_names]])
        input_scaled = scaler.transform(input_data)
        probabilidad = model.predict(input_scaled, verbose=0)[0][0]
        return probabilidad
    
    return predecir_exoplaneta


def ejemplo_prediccion(predictor_fn, feature_names):
    """Muestra un ejemplo de predicción"""
    print("\n[EJEMPLO DE USO]")
    ejemplo = {
        'koi_period': 10.5,
        'koi_depth': 500,
        'koi_duration': 3.5,
        'koi_prad': 2.0,
        'koi_teq': 400,
        'koi_insol': 50,
        'koi_steff': 5500,
        'koi_srad': 1.0
    }
    
    prob = predictor_fn(ejemplo)
    print("\nObjeto de ejemplo:")
    for k, v in ejemplo.items():
        if k in feature_names:
            print(f"  {k}: {v}")
    print(f"\n→ Probabilidad de ser exoplaneta: {prob*100:.2f}%")
    print(f"→ Clasificación: {'EXOPLANETA ✓' if prob > 0.5 else 'NO EXOPLANETA ✗'}")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que ejecuta el pipeline completo"""
    # Configuración inicial
    sns.set_style('darkgrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    
    print("=" * 60)
    print("CLASIFICADOR DE EXOPLANETAS - RED NEURONAL")
    print("=" * 60)
    
    # Pipeline de procesamiento
    resultados = pre.pipeline_completo('cumulative.csv')
    X_train = resultados['X_train']
    X_test = resultados['X_test']
    y_train = resultados['y_train']
    y_test = resultados['y_test']
    scaler = resultados['scaler']
    feature_names = resultados['feature_names']
    
    # Construcción y entrenamiento
    model = construir_modelo(X_train.shape[1])
    print(model.summary())
    history = entrenar_modelo(model, X_train, y_train)
    
    # Evaluación
    y_pred, y_pred_proba = evaluar_modelo(model, X_test, y_test)
    fpr, tpr, roc_auc = calcular_metricas(y_test, y_pred_proba)
    
    # Visualizaciones
    generar_visualizaciones(history, y_test, y_pred, y_pred_proba, 
                           model, feature_names, roc_auc)
    
    # Guardar modelo
    guardar_modelo(model, scaler, feature_names, roc_auc)
    
    # Resultados finales
    print("\n" + "=" * 60)
    print("✓ Modelo entrenado y evaluado exitosamente")
    print(f"✓ AUC-ROC: {roc_auc:.4f}")
    print("✓ Gráficos guardados en: exoplanet_classification_results.png")
    print("=" * 60)
    
    # Crear predictor y ejemplo
    predictor = crear_predictor(model, scaler, feature_names)
    ejemplo_prediccion(predictor, feature_names)
    
    return model, scaler, feature_names, predictor


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    model, scaler, feature_names, predictor = main()
    
    # ========================================================================
    # EJEMPLO DE CARGA DEL MODELO GUARDADO (descomenta para usar)
    # ========================================================================
    """
    # Para cargar un modelo previamente guardado:
    model_cargado, scaler_cargado, features_cargadas, metadata = cargar_modelo('modelo_exoplanetas')
    predictor_cargado = crear_predictor(model_cargado, scaler_cargado, features_cargadas)
    
    # Usar el predictor cargado
    ejemplo = {
        'koi_period': 15.3,
        'koi_depth': 800,
        'koi_duration': 4.2,
        'koi_prad': 3.5,
        'koi_teq': 500,
        'koi_insol': 75,
        'koi_steff': 5800,
        'koi_srad': 1.1
    }
    prob = predictor_cargado(ejemplo)
    print(f"\nProbabilidad de exoplaneta: {prob*100:.2f}%")
    """