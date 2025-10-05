"""
Vistas de la API para entrenamiento y predicción de modelos ML
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core.files.base import ContentFile
import pandas as pd
import numpy as np
import os
import logging
import pdb
import traceback
from .models import TrainedModel, Prediction
from .serializers import TrainingRequestSerializer, PredictionRequestSerializer, TrainedModelSerializer
from .ml_engine import MLEngine
from . import preprocesamiento as pre
from . import ml_models as ml

logger = logging.getLogger(__name__)

def debug_breakpoint(message="Debug breakpoint", data=None):
    """
    Función helper para debugging con breakpoints
    """
    print(f"🔍 BREAKPOINT: {message}")
    if data is not None:
        print(f"📊 DATA: {data}")
    # Descomenta la siguiente línea para activar el debugger interactivo
    # pdb.set_trace()

def debug_log_request(request, function_name):
    """
    Log detallado de la petición para debugging
    """
    logger.debug(f"🔍 {function_name} - Request Method: {request.method}")
    logger.debug(f"🔍 {function_name} - Request Path: {request.path}")
    logger.debug(f"🔍 {function_name} - Request Headers: {dict(request.headers)}")
    
    if hasattr(request, 'data') and request.data:
        logger.debug(f"🔍 {function_name} - Request Data Keys: {list(request.data.keys())}")
        # Log de datos específicos sin exponer información sensible
        for key, value in request.data.items():
            if key == 'csv_data' and isinstance(value, list):
                logger.debug(f"🔍 {function_name} - {key}: {len(value)} registros")
            else:
                logger.debug(f"🔍 {function_name} - {key}: {type(value)} - {str(value)[:100]}...")

def convert_data_types(df, column_types):
    """
    Convierte los tipos de datos de las columnas seleccionadas según la configuración del usuario
    
    Args:
        df: DataFrame de pandas
        column_types: Diccionario con {columna: tipo_dato}
    
    Returns:
        DataFrame con tipos de datos convertidos
    """
    logger.info(f"🔄 Iniciando conversión de tipos de datos para {len(column_types)} columnas")
    
    for column, data_type in column_types.items():
        if column not in df.columns:
            logger.warning(f"⚠️ Columna '{column}' no encontrada en el DataFrame")
            continue
            
        try:
            logger.info(f"🔄 Convirtiendo columna '{column}' a tipo '{data_type}'")
            
            if data_type == 'int':
                # Convertir a entero, manejando valores nulos
                df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                logger.info(f"✅ Columna '{column}' convertida a entero")
                
            elif data_type == 'float':
                # Convertir a flotante
                df[column] = pd.to_numeric(df[column], errors='coerce')
                logger.info(f"✅ Columna '{column}' convertida a flotante")
                
            elif data_type == 'boolean':
                # Convertir valores booleanos comunes
                bool_map = {
                    'true': True, 'false': False, '1': True, '0': False,
                    'yes': True, 'no': False, 'si': True, 'no': False,
                    'verdadero': True, 'falso': False
                }
                # Convertir a string primero, luego mapear
                df[column] = df[column].astype(str).str.lower().map(bool_map).fillna(df[column])
                logger.info(f"✅ Columna '{column}' convertida a booleano")
                
            elif data_type == 'datetime':
                # Convertir a datetime
                df[column] = pd.to_datetime(df[column], errors='coerce')
                logger.info(f"✅ Columna '{column}' convertida a datetime")
                
            elif data_type == 'string':
                # Convertir a string
                df[column] = df[column].astype('string')
                logger.info(f"✅ Columna '{column}' convertida a string")
                
            else:
                logger.warning(f"⚠️ Tipo de dato '{data_type}' no reconocido para columna '{column}'")
                
        except Exception as e:
            logger.error(f"❌ Error convirtiendo columna '{column}' a tipo '{data_type}': {e}")
            # Mantener el tipo original si hay error
            continue
    
    # Mostrar información del DataFrame después de la conversión
    logger.info(f"📊 DataFrame después de conversión: {df.shape[0]} filas, {df.shape[1]} columnas")
    logger.info(f"📋 Tipos de datos finales: {dict(df.dtypes)}")
    
    return df


@api_view(['POST'])
def train_model(request):
    """
    Endpoint para entrenar un modelo de machine learning
    """
    try:
        # Debugging detallado
        debug_log_request(request, "train_model")
        debug_breakpoint("Iniciando entrenamiento de modelo", {
            'data_keys': list(request.data.keys()) if hasattr(request, 'data') else None,
            'analysis_type': request.data.get('analysis_type') if hasattr(request, 'data') else None
        })
        
        print('🚀 Iniciando entrenamiento de modelo...')
        print(f'📊 Datos recibidos: {request.data}')
        logger.debug("🚀 Iniciando entrenamiento de modelo...")
        logger.info(f"📊 Datos recibidos: {request.data}")
        
        column_types = request.data['column_types']

        for column, dtype in column_types.items():
            if dtype == 'categorical':
                column_types[column] = 'category' # Correct the value here

        df_data = pd.DataFrame(request.data['csv_data'], columns=request.data['csv_columns'])
        
        df_data.replace('', np.nan, inplace=True)
        input_columns = list(column_types.keys())
        df_filtered_data = df_data[input_columns].astype(column_types)
        df_filtered_data.dropna(inplace=True)
        print("dataframe inicial")
        print(df_filtered_data.head())
        print(df_filtered_data.info())
        
        """ y_data = df_data[request.data['target_variable']]
        X_data = df_data.drop(request.data['target_variable'], axis=1)
        
        print(y_data.head())
        print(X_data.head())
         """
        nombre_variable_objetivo = request.data['target_variable']
        nombres_features = input_columns
        estrategia_valores_faltantes = 'knn'
        metodo_escalado = 'robust'
        metodo_balance = 'oversample'
         
        pre.exploracion_inicial(df_filtered_data)
    
        # 2. Preparar variable objetivo
        df = pre.preparar_variable_objetivo(df_filtered_data, nombre_variable_objetivo)
        print(df.head())
        # 3. Seleccionar features
        df_features, features = pre.seleccionar_features(df, nombres_features)
        
        # 4. Manejar valores faltantes
        df_clean = pre.manejar_valores_faltantes(nombre_variable_objetivo, df_features, estrategia_valores_faltantes)
        
        # 5. Detectar outliers (solo informativo)
        pre.detectar_outliers(df_clean)
        
        # 6. Escalar datos
        df_scaled, scaler = pre.escalar_datos(df_clean, metodo_escalado)
        
        # 7. Dividir datos
        X_train, X_test, y_train, y_test = pre.dividir_datos(df_scaled)
        
        # 8. Balancear clases (opcional, comentar si no se desea)
        X_train_bal, y_train_bal = pre.balancear_clases(X_train, y_train, metodo_balance)
        
        print("\n" + "="*70)
        print(" "*20 + "PIPELINE COMPLETADO")
        print("="*70)
        
        trained_model = ml.RandomForest_trainer(X_train_bal, y_train_bal, X_test, y_test)
        statistics = ml.Evaluation_model(X_test, y_test)
        
        statistics['model_id'] = 1
        statistics['recall'] = 1
        statistics['mae'] = 1
        statistics['r2_score'] = 1
        statistics['training_time'] = 1
        statistics['plots'] = 1 
        
        """ response_data = {
            'model_id': trained_model.id,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'mae': results['mae'],
            'r2_score': results['r2_score'],
            'training_time': results['training_time'],
            'plots': results['plots']
        }
         """
        # Guardar archivo del modelo
        print('💾 Guardando archivo del modelo...')
        logger.info("💾 Guardando archivo del modelo...")
        
        return Response(statistics, status=status.HTTP_200_OK)
        
        # Validar datos de entrada
        """ print('🔍 Validando datos de entrada...')
        logger.info("🔍 Validando datos de entrada...")
        serializer = TrainingRequestSerializer(data=request.data)
        if not serializer.is_valid():
            print(f'❌ Datos inválidos: {serializer.errors}')
            logger.error(f"❌ Datos inválidos: {serializer.errors}")
            return Response({
                'error': 'Datos de entrada inválidos',
                'details': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        print(f'✅ Datos validados correctamente: {data}')
        logger.info(f"✅ Datos validados correctamente: {data}")
        
        # Preparar datos según el tipo de análisis
        print(f'📋 Tipo de análisis: {data["analysis_type"]}')
        logger.info(f"📋 Tipo de análisis: {data['analysis_type']}")
        
        if data['analysis_type'] == 'own-data':
            print('📁 Procesando datos propios (CSV)...')
            logger.info("📁 Procesando datos propios (CSV)...")
            # Usar datos del CSV proporcionado
            if not data.get('csv_data') or not data.get('csv_columns'):
                print('❌ Datos CSV no encontrados')
                logger.error("❌ Datos CSV no encontrados")
                return Response({
                    'error': 'Datos CSV requeridos para análisis con datos propios'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Crear DataFrame
            print(f'📊 Creando DataFrame con {len(data["csv_data"])} filas y {len(data["csv_columns"])} columnas')
            logger.info(f"📊 Creando DataFrame con {len(data['csv_data'])} filas y {len(data['csv_columns'])} columnas")
            df = pd.DataFrame(data['csv_data'], columns=data['csv_columns'])
            
            target_variable = data['target_variable']
            input_variables = data['input_variables']
            model_name = f"Modelo_{data['file_name']}_{data['model']}"
            
            print(f'🎯 Variable objetivo: {target_variable}')
            print(f'📝 Variables de entrada: {input_variables}')
            logger.info(f"🎯 Variable objetivo: {target_variable}")
            logger.info(f"📝 Variables de entrada: {input_variables}")
            
            # Procesar tipos de datos si se proporcionan
            if data.get('column_types'):
                print(f'🔧 Procesando tipos de datos: {data["column_types"]}')
                logger.info(f"🔧 Procesando tipos de datos: {data['column_types']}")
                
                # Filtrar solo las columnas que se van a usar
                selected_columns = [target_variable] + input_variables
                relevant_types = {k: v for k, v in data['column_types'].items() if k in selected_columns}
                
                print(f'📋 Tipos relevantes para variables seleccionadas: {relevant_types}')
                logger.info(f"📋 Tipos relevantes para variables seleccionadas: {relevant_types}")
                
                # Aplicar conversiones de tipos de datos
                df = convert_data_types(df, relevant_types)
                
                print(f'✅ Conversión de tipos completada')
                logger.info(f"✅ Conversión de tipos completada")
            else:
                print(f'⚠️ No se proporcionaron tipos de datos, usando tipos por defecto')
                logger.info(f"⚠️ No se proporcionaron tipos de datos, usando tipos por defecto")
            
        else:  # app-data
            print('🌐 Usando dataset predefinido de la aplicación...')
            logger.info("🌐 Usando dataset predefinido de la aplicación...")
            # Usar dataset predefinido de la aplicación
            df, target_variable, input_variables = get_app_dataset(data.get('dataset_name', 'kepler'))
            model_name = f"Modelo_{data['dataset_name']}_{data['model']}"
            print(f'📊 Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas')
            print(f'🎯 Variable objetivo: {target_variable}')
            print(f'📝 Variables de entrada: {input_variables}')
            logger.info(f"📊 Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
            logger.info(f"🎯 Variable objetivo: {target_variable}")
            logger.info(f"📝 Variables de entrada: {input_variables}")
        
        # Validar que las variables existen en los datos
        if target_variable not in df.columns:
            return Response({
                'error': f'Variable objetivo "{target_variable}" no encontrada en los datos'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        missing_vars = [var for var in input_variables if var not in df.columns]
        if missing_vars:
            return Response({
                'error': f'Variables de entrada no encontradas: {missing_vars}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Entrenar modelo
        print(f'🤖 Iniciando entrenamiento con modelo: {data["model"]}')
        print(f'⚙️ Hiperparámetros: {data["hyperparameters"]}')
        if data.get('column_types'):
            print(f'📊 Tipos de datos: {data["column_types"]}')
        logger.info(f"🤖 Iniciando entrenamiento con modelo: {data['model']}")
        logger.info(f"⚙️ Hiperparámetros: {data['hyperparameters']}")
        if data.get('column_types'):
            logger.info(f"📊 Tipos de datos: {data['column_types']}")
        
        ml_engine = MLEngine()
        results = ml_engine.train_model(
            X_data=X_data,
            y_data=y_data,
            model_type=data['model'],
            hyperparameters=data['hyperparameters'],
            analysis_type=data['analysis_type'],
            column_types=data.get('column_types')  # Pasar tipos de datos
        )
        
        print(f'✅ Entrenamiento completado!')
        print(f'📈 Accuracy: {results["accuracy"]:.3f}')
        print(f'⏱️ Tiempo: {results["training_time"]}')
        logger.info(f"✅ Entrenamiento completado!")
        logger.info(f"📈 Accuracy: {results['accuracy']:.3f}")
        logger.info(f"⏱️ Tiempo: {results['training_time']}")
        
        # Guardar modelo en la base de datos
        print('💾 Guardando modelo en la base de datos...')
        logger.info("💾 Guardando modelo en la base de datos...")
        
        trained_model = TrainedModel.objects.create(
            name=model_name,
            model_type=data['model'],
            analysis_type=data['analysis_type'],
            hyperparameters=data['hyperparameters'],
            accuracy=results['accuracy'],
            precision=results['precision'],
            recall=results['recall'],
            f1_score=results['f1_score'],
            mae=results['mae'],
            r2_score=results['r2_score'],
            training_time=results['training_time'],
            target_variable=target_variable if data['analysis_type'] == 'own-data' else None,
            input_variables=input_variables if data['analysis_type'] == 'own-data' else None,
            file_name=data.get('file_name') if data['analysis_type'] == 'own-data' else None,
            dataset_name=data.get('dataset_name') if data['analysis_type'] == 'app-data' else None,
        )
        
        print(f'✅ Modelo guardado con ID: {trained_model.id}')
        logger.info(f"✅ Modelo guardado con ID: {trained_model.id}")
        
        # Guardar archivo del modelo
        print('💾 Guardando archivo del modelo...')
        logger.info("💾 Guardando archivo del modelo...")
        
        model_file_path = f"models/model_{trained_model.id}.pkl"
        os.makedirs(os.path.dirname(f"media/{model_file_path}"), exist_ok=True)
        ml_engine.save_model(f"media/{model_file_path}")
        trained_model.model_file = model_file_path
        trained_model.save()
        
        print(f'✅ Archivo del modelo guardado: {model_file_path}')
        logger.info(f"✅ Archivo del modelo guardado: {model_file_path}")
        
        # Preparar respuesta
        print('📤 Preparando respuesta...')
        logger.info("📤 Preparando respuesta...")
        
        response_data = {
            'model_id': trained_model.id,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'mae': results['mae'],
            'r2_score': results['r2_score'],
            'training_time': results['training_time'],
            'plots': results['plots']
        }
        
        print(f'🎉 Entrenamiento completado exitosamente: {model_name}')
        print(f'📊 Respuesta: {response_data}')
        logger.info(f"🎉 Modelo entrenado exitosamente: {model_name}")
        logger.info(f"📊 Respuesta: {response_data}") """
        
        """ return Response(response_data, status=status.HTTP_200_OK) """
        
    except Exception as e:
        print(f'❌ Error en entrenamiento: {str(e)}')
        logger.error(f"❌ Error en entrenamiento: {str(e)}")
        return Response({
            'error': 'Error interno del servidor',
            'details': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def predict(request):
    """
    Endpoint para realizar predicciones con un modelo entrenado
    """
    try:
        """ print('🔮 Iniciando predicción...')
        print(f'📊 Datos de predicción recibidos: {request.data}')
        logger.info("🔮 Iniciando predicción...")
        logger.info(f"📊 Datos de predicción recibidos: {request.data}")
        
        # Validar datos de entrada
        print('🔍 Validando datos de predicción...')
        logger.info("🔍 Validando datos de predicción...")
        serializer = PredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            print(f'❌ Datos de predicción inválidos: {serializer.errors}')
            logger.error(f"❌ Datos de predicción inválidos: {serializer.errors}")
            return Response({
                'error': 'Datos de entrada inválidos',
                'details': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        print(f'✅ Datos de predicción validados: {data}')
        logger.info(f"✅ Datos de predicción validados: {data}")
        
        # Obtener modelo
        try:
            trained_model = TrainedModel.objects.get(id=data['model_id'])
        except TrainedModel.DoesNotExist:
            return Response({
                'error': 'Modelo no encontrado'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Cargar modelo
        ml_engine = MLEngine()
        if trained_model.model_file:
            ml_engine.load_model(f"media/{trained_model.model_file}")
        else:
            return Response({
                'error': 'Archivo del modelo no disponible'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Realizar predicción
        prediction = ml_engine.predict(data['input_data'])
        
        # Convertir predicción a tipo serializable
        if isinstance(prediction, (np.integer, np.floating)):
            prediction = prediction.item()
        elif isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        
        print(f'🔮 Predicción realizada: {prediction} (tipo: {type(prediction)})')
        logger.info(f"🔮 Predicción realizada: {prediction} (tipo: {type(prediction)})")
        
        # Guardar predicción en la base de datos
        prediction_record = Prediction.objects.create(
            model=trained_model,
            input_data=data['input_data'],
            prediction_result={'prediction': prediction},
            confidence=None  # Se puede calcular si es necesario
        ) """
        """ 
        response_data = {
            'prediction_id': prediction_record.id,
            'prediction': prediction,
            'model_name': trained_model.name,
            'model_accuracy': trained_model.accuracy
        } """
        
        response_data = {
            'prediction_id': 1,
            'prediction': 0.85,
            'model_name': "#2",
            'model_accuracy': 0.79
        }
        
        """ logger.info(f"Predicción realizada exitosamente: {prediction}") """
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return Response({
            'error': 'Error interno del servidor',
            'details': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def list_models(request):
    """
    Endpoint para listar todos los modelos entrenados
    """
    try:
        models = TrainedModel.objects.all()
        serializer = TrainedModelSerializer(models, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error listando modelos: {str(e)}")
        return Response({
            'error': 'Error interno del servidor',
            'details': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_model(request, model_id):
    """
    Endpoint para obtener detalles de un modelo específico
    """
    try:
        model = TrainedModel.objects.get(id=model_id)
        serializer = TrainedModelSerializer(model)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except TrainedModel.DoesNotExist:
        return Response({
            'error': 'Modelo no encontrado'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error obteniendo modelo: {str(e)}")
        return Response({
            'error': 'Error interno del servidor',
            'details': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def get_app_dataset(dataset_name):
    """
    Obtiene un dataset predefinido de la aplicación
    """
    if dataset_name == 'kepler':
        # Dataset de ejemplo basado en los datos de Kepler
        np.random.seed(42)
        n_samples = 1000
        
        # Generar datos numéricos correctamente
        data = {
            'koi_period': np.random.uniform(0.5, 1000, n_samples).astype(float),
            'koi_impact': np.random.uniform(0, 1, n_samples).astype(float),
            'koi_duration': np.random.uniform(0.1, 50, n_samples).astype(float),
            'koi_depth': np.random.uniform(0, 1000, n_samples).astype(float),
            'koi_prad': np.random.uniform(0.1, 20, n_samples).astype(float),
            'koi_teq': np.random.uniform(200, 3000, n_samples).astype(float),
            'koi_insol': np.random.uniform(0, 10, n_samples).astype(float),
            'koi_steff': np.random.uniform(3000, 8000, n_samples).astype(float),
            'koi_slogg': np.random.uniform(3, 5, n_samples).astype(float),
            'koi_srad': np.random.uniform(0.5, 2, n_samples).astype(float),
        }
        
        # Crear variable objetivo basada en las características
        # Simular clasificación: planeta habitable (1) o no habitable (0)
        habitable_score = (
            (data['koi_teq'] > 200) & (data['koi_teq'] < 300) &  # Temperatura adecuada
            (data['koi_insol'] > 0.5) & (data['koi_insol'] < 2) &  # Insolación adecuada
            (data['koi_prad'] > 0.5) & (data['koi_prad'] < 2)  # Tamaño adecuado
        ).astype(int)
        
        data['habitable'] = habitable_score
        
        # Crear DataFrame y asegurar tipos correctos
        df = pd.DataFrame(data)
        
        # Convertir todas las columnas numéricas a float64
        for col in df.columns:
            if col != 'habitable':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = df[col].astype(int)
        
        # Eliminar filas con valores NaN
        df = df.dropna()
        
        target_variable = 'habitable'
        input_variables = [col for col in df.columns if col != target_variable]
        
        print(f"📊 Dataset Kepler generado: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"📋 Tipos de datos: {df.dtypes.to_dict()}")
        logger.info(f"📊 Dataset Kepler generado: {df.shape[0]} filas, {df.shape[1]} columnas")
        logger.info(f"📋 Tipos de datos: {df.dtypes.to_dict()}")
        
        return df, target_variable, input_variables
    
    else:
        # Dataset por defecto
        np.random.seed(42)
        n_samples = 500
        
        data = {
            'feature_1': np.random.normal(0, 1, n_samples).astype(float),
            'feature_2': np.random.normal(0, 1, n_samples).astype(float),
            'feature_3': np.random.normal(0, 1, n_samples).astype(float),
            'feature_4': np.random.normal(0, 1, n_samples).astype(float),
        }
        
        # Variable objetivo
        data['target'] = (
            data['feature_1'] * 0.5 +
            data['feature_2'] * 0.3 +
            data['feature_3'] * 0.2 +
            np.random.normal(0, 0.1, n_samples)
        ).astype(float)
        
        # Crear DataFrame y asegurar tipos correctos
        df = pd.DataFrame(data)
        
        # Convertir todas las columnas a float64
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Eliminar filas con valores NaN
        df = df.dropna()
        
        target_variable = 'target'
        input_variables = [col for col in df.columns if col != target_variable]
        
        print(f"📊 Dataset por defecto generado: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"📋 Tipos de datos: {df.dtypes.to_dict()}")
        logger.info(f"📊 Dataset por defecto generado: {df.shape[0]} filas, {df.shape[1]} columnas")
        logger.info(f"📋 Tipos de datos: {df.dtypes.to_dict()}")
        
        return df, target_variable, input_variables
