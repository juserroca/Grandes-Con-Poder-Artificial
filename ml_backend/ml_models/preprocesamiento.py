# -*- coding: utf-8 -*-
# ============================================================================
# ETAPA 1: IMPORTACIÓN DE LIBRERÍAS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Librerías importadas correctamente")

# ============================================================================
# ETAPA 2: CARGA Y EXPLORACIÓN INICIAL DE DATOS
# ============================================================================

def cargar_datos(ruta_archivo):
    """
    Carga los datos del archivo CSV de KOI
    
    Parámetros:
    - ruta_archivo: string con la ruta al archivo CSV descargado
    
    Retorna:
    - DataFrame de pandas con los datos
    """
    print("Cargando datos...")
    df = pd.read_csv(ruta_archivo, comment='#')  # comment='#' ignora líneas de metadatos
    print(f"✓ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

def exploracion_inicial(df, nombre_variable_objetivo = "koi_disposition"):
    """
    Realiza exploración inicial del dataset
    """
    print("\n" + "="*60)
    print("EXPLORACIÓN INICIAL DE DATOS")
    print("="*60)
    
    # Información general
    print("\n1. Información del DataFrame:")
    print(df.info())
    
    # Primeras filas
    print("\n2. Primeras 5 filas:")
    print(df.head())
    
    # Estadísticas descriptivas
    print("\n3. Estadísticas descriptivas:")
    print(df.describe())
    
    # Valores faltantes
    print("\n4. Valores faltantes por columna:")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'Columna': missing.index,
        'Faltantes': missing.values,
        'Porcentaje': missing_pct.values
    })
    print(missing_df[missing_df['Faltantes'] > 0].sort_values('Faltantes', ascending=False))
    
    # Distribución de la variable objetivo
    if nombre_variable_objetivo in df.columns:
        print(f"\n5. Distribución de la variable objetivo ({nombre_variable_objetivo}):")
        print(df[nombre_variable_objetivo].value_counts())
        print("\nPorcentajes:")
        print(df[nombre_variable_objetivo].value_counts(normalize=True) * 100)

# Ejemplo de uso:
# df = cargar_datos('cumulative.csv')
# exploracion_inicial(df)
#df = cargar_datos('cumulative.csv')
#exploracion_inicial(df)

# ============================================================================
# ETAPA 3: PREPARACIÓN DE LA VARIABLE OBJETIVO
# ============================================================================

def preparar_variable_objetivo(df, nombre_variable_objetivo = "koi_disposition"):
    """
    Convierte la variable objetivo en binaria:
    - CONFIRMED = 1 (es exoplaneta)
    - FALSE POSITIVE, CANDIDATE = 0 (no es exoplaneta confirmado)
    
    En KOI, hay tres categorías principales:
    - CONFIRMED: Exoplaneta confirmado
    - CANDIDATE: Candidato (aún no confirmado)
    - FALSE POSITIVE: Falso positivo
    """
    print("\n" + "="*60)
    print("PREPARACIÓN DE VARIABLE OBJETIVO")
    print("="*60)
    
    # Crear copia para no modificar el original
    df_prep = df.copy()
    
    # Verificar columna de disposición
    if nombre_variable_objetivo not in df_prep.columns:
        print(f"⚠ Columna '{nombre_variable_objetivo}' no encontrada. Columnas disponibles:")
        print(df_prep.columns.tolist())
        return df_prep
    
    # Crear variable binaria
    df_prep['es_exoplaneta'] = (df_prep[nombre_variable_objetivo] == 'CONFIRMED').astype(int)
    print("df con variable objetivo")
    print(df_prep['es_exoplaneta'])
    
    print("\n✓ Variable objetivo creada:")
    print(f"  - Exoplanetas confirmados (1): {df_prep['es_exoplaneta'].sum()}")
    print(f"  - No confirmados (0): {(df_prep['es_exoplaneta'] == 0).sum()}")
    print(f"  - Proporción de exoplanetas: {df_prep['es_exoplaneta'].mean():.2%}")
    
    return df_prep
# df_prep = preparar_variable_objetivo(df)
# ============================================================================
# ETAPA 4: SELECCIÓN DE FEATURES
# ============================================================================

def seleccionar_features(df, nombre_features = [
    'koi_period',
    'koi_impact',
    'koi_duration',
    'koi_depth',
    'koi_prad',
    'koi_teq',
    'koi_insol',
    'koi_model_snr',
    'koi_slogg',
    'koi_srad'
    ]):
    
    """
    Toma las features que definió el usuario y devuelve el dataset solamente 
    con las columnas escogidas
    """
    print("\n" + "="*60)
    print("SELECCIÓN DE FEATURES")
    print("="*60)
    
    # Filtrar solo las columnas que existen en el dataset
    features_disponibles = [f for f in nombre_features if f in df.columns]
    
    print(f"✓ Features seleccionadas: {len(features_disponibles)}")
    print(f"  Disponibles de la lista: {features_disponibles[:10]}...")
    
    # Crear DataFrame solo con features y variable objetivo
    if 'es_exoplaneta' in df.columns:
        df_features = df[features_disponibles + ['es_exoplaneta']].copy()
    else:
        df_features = df[features_disponibles].copy()
        print("\n⚠ Variable objetivo no encontrada. Asegúrate de ejecutar preparar_variable_objetivo() primero.")
    
    return df_features, features_disponibles
#df_features, features_disponibles = seleccionar_features(df_prep)

# ============================================================================
# ETAPA 5: MANEJO DE VALORES FALTANTES
# ============================================================================

def manejar_valores_faltantes(nombre_variable_objetivo, df, estrategia='knn', umbral_columna=0.2):
    """
    Maneja valores faltantes en el dataset.
    
    Parámetros:
    - df: DataFrame con los datos
    - estrategia: 'mean', 'median', 'knn' o 'drop'
    - umbral_columna: si una columna tiene más de este % de faltantes, se elimina
    """
    print("\n" + "="*60)
    print("MANEJO DE VALORES FALTANTES")
    print("="*60)
    
    df_clean = df.copy()
    
    # Separar variable objetivo
    if 'es_exoplaneta' in df_clean.columns:
        y = df_clean['es_exoplaneta']
        X = df_clean.drop('es_exoplaneta', axis=1)
        X = X.drop(nombre_variable_objetivo, axis=1)
    else:
        y = None
        X = df_clean
    
    print("\n1. Estado inicial:")
    print(f"   - Filas: {X.shape[0]}, Columnas: {X.shape[1]}")
    print(f"   - Total de valores faltantes: {X.isnull().sum().sum()}")
    
    # Eliminar columnas con demasiados valores faltantes
    missing_pct = X.isnull().sum() / len(X)
    columnas_eliminar = missing_pct[missing_pct > umbral_columna].index.tolist()
    
    if columnas_eliminar:
        print(f"\n2. Eliminando {len(columnas_eliminar)} columnas con >{umbral_columna*100}% faltantes:")
        print(f"   {columnas_eliminar}")
        X = X.drop(columns=columnas_eliminar)
    
    # Imputación de valores faltantes
    if estrategia in ['mean', 'median']:
        print(f"\n3. Imputando valores faltantes con estrategia: {estrategia}")
        imputer = SimpleImputer(strategy=estrategia)
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    elif estrategia == 'knn':
        print("\n3. Imputando valores faltantes con KNN (k=5)")
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    elif estrategia == 'drop':
        print("\n3. Eliminando filas con valores faltantes")
        X_imputed = X.dropna()
        if y is not None:
            y = y.loc[X_imputed.index]
    else:
        print(f"⚠ Estrategia '{estrategia}' no reconocida. Usando 'median'")
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    
    print("\n✓ Estado final:")
    print(f"   - Filas: {X_imputed.shape[0]}, Columnas: {X_imputed.shape[1]}")
    print(f"   - Valores faltantes restantes: {X_imputed.isnull().sum().sum()}")
    
    # Recombinar con variable objetivo
    if y is not None:
        df_final = X_imputed.copy()
        df_final['es_exoplaneta'] = y
        return df_final
    else:
        return X_imputed
#df_final = manejar_valores_faltantes (df_features)

# ============================================================================
# ETAPA 6: DETECCIÓN DE OUTLIERS
# ============================================================================

def detectar_outliers(df, columnas=None, max_mostrar=10):
    print("\n" + "="*60)
    print("DETECCIÓN DE OUTLIERS")
    print("="*60)
    
    if columnas is None:
        columnas = df.select_dtypes(include=np.number).columns
    
    # filtrar solo columnas que estén en el DataFrame
    columnas = [c for c in columnas if c in df.columns]
    
    resumen_outliers = {}
    
    for col in columnas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)][col]
        
        resumen_outliers[col] = len(outliers)
        
        print(f"\nColumna: {col}")
        print(f"  - Outliers detectados: {len(outliers)}")
        if not outliers.empty:
            print(f"  - Ejemplos (máx {max_mostrar}):")
            print(outliers.head(max_mostrar).to_string(index=True))
    
    print("\n✓ Resumen de outliers por columna:")
    for col, n in resumen_outliers.items():
        print(f"  {col}: {n}")
    
    return resumen_outliers
# Ejemplo de uso después del manejo de faltantes
#resumen_outliers = detectar_outliers(df_final, columnas=['koi_prad','koi_period','koi_teq'])

# ============================================================================
# ETAPA 7: ESCALADO Y NORMALIZACIÓN
# ============================================================================

def escalar_datos(df, metodo='robust'):
    """
    Escala las features usando StandardScaler o RobustScaler.
    
    RobustScaler es preferible cuando hay outliers (común en datos astronómicos).
    """
    print("\n" + "="*60)
    print("ESCALADO Y NORMALIZACIÓN")
    print("="*60)
    
    df_scaled = df.copy()
    
    # Separar variable objetivo
    if 'es_exoplaneta' in df_scaled.columns:
        y = df_scaled['es_exoplaneta']
        X = df_scaled.drop('es_exoplaneta', axis=1)
    else:
        y = None
        X = df_scaled
    
    # Seleccionar scaler
    if metodo == 'standard':
        scaler = StandardScaler()
        print("Usando StandardScaler (sensible a outliers)")
    elif metodo == 'robust':
        scaler = RobustScaler()
        print("Usando RobustScaler (robusto a outliers)")
    else:
        print(f"⚠ Método '{metodo}' no reconocido. Usando RobustScaler")
        scaler = RobustScaler()
    
    # Escalar datos
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    print("\n✓ Datos escalados:")
    print(f"   - Media aproximada: {X_scaled.mean().mean():.4f}")
    print(f"   - Desviación estándar aproximada: {X_scaled.std().mean():.4f}")
    
    # Recombinar con variable objetivo
    if y is not None:
        X_scaled['es_exoplaneta'] = y
    
    return X_scaled, scaler
#df_escalado, scaler = escalar_datos(df_final, metodo='robust')
#X = df_escalado.drop('es_exoplaneta', axis=1)
#y = df_escalado['es_exoplaneta']

# ============================================================================
# ETAPA 8: BALANCEO DE CLASES
# ============================================================================

def balancear_clases(X, y, metodo='oversample'):
    """
    Balancea las clases usando métodos de sklearn.
    
    Parámetros:
    - X: Features
    - y: Variable objetivo
    - metodo: 'oversample', 'undersample', 'combine', o 'none'
    """
    print("\n" + "="*60)
    print("BALANCEO DE CLASES")
    print("="*60)
    
    print("\nDistribución original:")
    print(f"  Clase 0: {(y == 0).sum()} ({(y == 0).mean():.2%})")
    print(f"  Clase 1: {(y == 1).sum()} ({(y == 1).mean():.2%})")
    
    if metodo == 'none':
        print("\nNo se aplicó balanceo.")
        return X, y
    
    elif metodo == 'oversample':
        print("\nAplicando Random Over-sampling (sklearn)...")
        # Separar las clases
        X_0 = X[y == 0]
        X_1 = X[y == 1]
        y_0 = y[y == 0]
        y_1 = y[y == 1]
        
        # Determinar cuál clase es minoritaria
        if len(y_1) < len(y_0):
            # Clase 1 es minoritaria, hacer oversampling
            X_1_oversampled, y_1_oversampled = resample(
                X_1, y_1, 
                replace=True, 
                n_samples=len(y_0), 
                random_state=42
            )
            X_balanced = pd.concat([X_0, X_1_oversampled], ignore_index=True)
            y_balanced = pd.concat([y_0, y_1_oversampled], ignore_index=True)
        else:
            # Clase 0 es minoritaria, hacer oversampling
            X_0_oversampled, y_0_oversampled = resample(
                X_0, y_0, 
                replace=True, 
                n_samples=len(y_1), 
                random_state=42
            )
            X_balanced = pd.concat([X_0_oversampled, X_1], ignore_index=True)
            y_balanced = pd.concat([y_0_oversampled, y_1], ignore_index=True)
    
    elif metodo == 'undersample':
        print("\nAplicando Random Under-sampling (sklearn)...")
        # Separar las clases
        X_0 = X[y == 0]
        X_1 = X[y == 1]
        y_0 = y[y == 0]
        y_1 = y[y == 1]
        
        # Determinar cuál clase es mayoritaria
        if len(y_1) > len(y_0):
            # Clase 1 es mayoritaria, hacer undersampling
            X_1_undersampled, y_1_undersampled = resample(
                X_1, y_1, 
                replace=False, 
                n_samples=len(y_0), 
                random_state=42
            )
            X_balanced = pd.concat([X_0, X_1_undersampled], ignore_index=True)
            y_balanced = pd.concat([y_0, y_1_undersampled], ignore_index=True)
        else:
            # Clase 0 es mayoritaria, hacer undersampling
            X_0_undersampled, y_0_undersampled = resample(
                X_0, y_0, 
                replace=False, 
                n_samples=len(y_1), 
                random_state=42
            )
            X_balanced = pd.concat([X_0_undersampled, X_1], ignore_index=True)
            y_balanced = pd.concat([y_0_undersampled, y_1], ignore_index=True)
    
    elif metodo == 'combine':
        print("\nAplicando combinación de Over y Under-sampling (sklearn)...")
        # Primero hacer oversampling de la clase minoritaria
        X_0 = X[y == 0]
        X_1 = X[y == 1]
        y_0 = y[y == 0]
        y_1 = y[y == 1]
        
        if len(y_1) < len(y_0):
            # Oversample clase 1 hasta 70% de la clase 0
            target_size = int(len(y_0) * 0.7)
            X_1_oversampled, y_1_oversampled = resample(
                X_1, y_1, 
                replace=True, 
                n_samples=target_size, 
                random_state=42
            )
            # Undersample clase 0 hasta 80% del tamaño original
            target_size_0 = int(len(y_0) * 0.8)
            X_0_undersampled, y_0_undersampled = resample(
                X_0, y_0, 
                replace=False, 
                n_samples=target_size_0, 
                random_state=42
            )
            X_balanced = pd.concat([X_0_undersampled, X_1_oversampled], ignore_index=True)
            y_balanced = pd.concat([y_0_undersampled, y_1_oversampled], ignore_index=True)
        else:
            # Hacer lo mismo pero al revés
            target_size = int(len(y_1) * 0.7)
            X_0_oversampled, y_0_oversampled = resample(
                X_0, y_0, 
                replace=True, 
                n_samples=target_size, 
                random_state=42
            )
            target_size_1 = int(len(y_1) * 0.8)
            X_1_undersampled, y_1_undersampled = resample(
                X_1, y_1, 
                replace=False, 
                n_samples=target_size_1, 
                random_state=42
            )
            X_balanced = pd.concat([X_0_oversampled, X_1_undersampled], ignore_index=True)
            y_balanced = pd.concat([y_0_oversampled, y_1_undersampled], ignore_index=True)
    
    else:
        print(f"⚠ Método '{metodo}' no reconocido. No se aplicó balanceo.")
        return X, y
    
    print("\nDistribución balanceada:")
    print(f"  Clase 0: {(y_balanced == 0).sum()} ({(y_balanced == 0).mean():.2%})")
    print(f"  Clase 1: {(y_balanced == 1).sum()} ({(y_balanced == 1).mean():.2%})")
    
    return X_balanced, y_balanced
#X_bal, y_bal = balancear_clases(X, y, metodo='smote')


# ============================================================================
# ETAPA 9: DIVISIÓN DE DATOS (TRAIN/TEST)
# ============================================================================

def dividir_datos(df, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    print("\n" + "="*60)
    print("DIVISIÓN DE DATOS (TRAIN/TEST)")
    print("="*60)
    
    # Separar features y objetivo
    X = df.drop('es_exoplaneta', axis=1)
    y = df['es_exoplaneta']
    
    # División estratificada para mantener proporciones
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Mantiene la proporción de clases
    )
    
    print("\n✓ División completada:")
    print(f"   - Entrenamiento: {X_train.shape[0]} muestras ({(1-test_size)*100:.0f}%)")
    print(f"   - Prueba: {X_test.shape[0]} muestras ({test_size*100:.0f}%)")
    print("\n   Distribución en entrenamiento:")
    print(f"     Clase 0: {(y_train == 0).sum()} ({(y_train == 0).mean():.2%})")
    print(f"     Clase 1: {(y_train == 1).sum()} ({(y_train == 1).mean():.2%})")
    print("\n   Distribución en prueba:")
    print(f"     Clase 0: {(y_test == 0).sum()} ({(y_test == 0).mean():.2%})")
    print(f"     Clase 1: {(y_test == 1).sum()} ({(y_test == 1).mean():.2%})")
    
    return X_train, X_test, y_train, y_test
#X_train, X_test, y_train, y_test = dividir_datos(
#    pd.concat([X_bal, y_bal], axis=1), 
#    test_size=0.2, 
#    random_state=42
#    )

def pipeline_completo(ruta_archivo, nombre_variable_objetivo = "koi_disposition",
                nombres_features = [
                'koi_period',
                'koi_impact',
                'koi_duration',
                'koi_depth',
                'koi_prad',
                'koi_teq',
                'koi_insol',
                'koi_model_snr',
                'koi_slogg',
                'koi_srad'
                ], estrategia_valores_faltantes = 'knn', metodo_escalado = 'robust',
                                  metodo_balance = 'oversample'):
    """
    Ejecuta el pipeline completo de principio a fin.
    
    USO:
    pipeline_completo('ruta/al/archivo/cumulative.csv')
    """
    print("\n" + "="*70)
    print(" "*15 + "PIPELINE DE DETECCIÓN DE EXOPLANETAS")
    print("="*70)
    
    # 1. Cargar datos
    df = cargar_datos(ruta_archivo)
    exploracion_inicial(df)
    
    # 2. Preparar variable objetivo
    df = preparar_variable_objetivo(df, nombre_variable_objetivo)
    
    # 3. Seleccionar features
    df_features, features = seleccionar_features(df, nombres_features)
    
    # 4. Manejar valores faltantes
    df_clean = manejar_valores_faltantes(df_features, estrategia_valores_faltantes)
    
    # 5. Detectar outliers (solo informativo)
    detectar_outliers(df_clean)
    
    # 6. Escalar datos
    df_scaled, scaler = escalar_datos(df_clean, metodo_escalado)
    
    # 7. Dividir datos
    X_train, X_test, y_train, y_test = dividir_datos(df_scaled)
    
    # 8. Balancear clases (opcional, comentar si no se desea)
    X_train_bal, y_train_bal = balancear_clases(X_train, y_train, metodo_balance)
    
    print("\n" + "="*70)
    print(" "*20 + "PIPELINE COMPLETADO")
    print("="*70)
    
    return {
        'datos_limpios': df_clean,
        'datos_escalados': df_scaled,
        'X_train': X_train_bal,
        'X_test': X_test,
        'y_train': y_train_bal,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': features
    }