"""
Motor de Machine Learning para entrenamiento y predicción de modelos
"""
import pandas as pd
import numpy as np
import pickle
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import logging

logger = logging.getLogger(__name__)


class MLEngine:
    """Motor principal para entrenamiento y predicción de modelos ML"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.target_name = None
        self.column_types = {}
    
    def process_data_types(self, df, column_types, target_variable, input_variables):
        """
        Procesa los tipos de datos solo para las variables seleccionadas por el usuario
        """
        # Filtrar solo las variables que se van a usar
        selected_variables = [target_variable] + input_variables
        relevant_types = {k: v for k, v in column_types.items() if k in selected_variables}
        
        logger.info(f"Procesando tipos de datos para variables seleccionadas: {relevant_types}")
        
        for column, data_type in relevant_types.items():
            if column not in df.columns:
                continue
                
            try:
                if data_type == 'int':
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                elif data_type == 'float':
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                elif data_type == 'boolean':
                    # Convertir valores booleanos comunes
                    bool_map = {
                        'true': True, 'false': False, '1': True, '0': False,
                        'yes': True, 'no': False, 'si': True, 'no': False
                    }
                    df[column] = df[column].str.lower().map(bool_map).fillna(df[column])
                elif data_type == 'datetime':
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                elif data_type == 'string':
                    df[column] = df[column].astype('string')
                
                logger.info(f"Columna {column} convertida a {data_type}")
                
            except Exception as e:
                logger.warning(f"Error procesando columna {column} como {data_type}: {e}")
                # Mantener el tipo original si hay error
                continue
        
        return df
        
    def train_model(self, data, target_variable, input_variables, model_type, hyperparameters, analysis_type, column_types=None):
        """
        Entrena un modelo de machine learning
        
        Args:
            data: DataFrame con los datos
            target_variable: Variable objetivo
            input_variables: Lista de variables de entrada
            model_type: Tipo de modelo ('random-forest', 'linear-regression', etc.)
            hyperparameters: Diccionario con hiperparámetros
            analysis_type: Tipo de análisis ('own-data' o 'app-data')
        
        Returns:
            dict: Métricas y gráficos del modelo entrenado
        """
        start_time = time.time()
        
        try:
            # Los tipos de datos ya fueron procesados en views.py
            if column_types:
                self.column_types = column_types
                logger.info(f"📊 Tipos de datos recibidos: {column_types}")
            
            # Preparar datos
            X = data[input_variables]
            y = data[target_variable]
            
            # Detectar si es clasificación o regresión
            is_classification = self._detect_task_type(y)
            
            # Preprocesar datos
            X_processed, y_processed = self._preprocess_data(X, y, is_classification)
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.2, random_state=42
            )
            
            # Crear y entrenar modelo
            self.model = self._create_model(model_type, hyperparameters, is_classification)
            self.model.fit(X_train, y_train)
            
            # Hacer predicciones
            y_pred = self.model.predict(X_test)
            
            # Calcular métricas
            metrics = self._calculate_metrics(y_test, y_pred, is_classification)
            
            # Generar gráficos
            plots = self._generate_plots(X_test, y_test, y_pred, is_classification)
            
            # Calcular tiempo de entrenamiento
            training_time = time.time() - start_time
            training_time_str = f"{int(training_time // 60)}m {int(training_time % 60)}s"
            
            # Guardar información del modelo
            self.feature_names = input_variables
            self.target_name = target_variable
            
            return {
                'accuracy': metrics.get('accuracy', 0.0),
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1_score': metrics.get('f1_score', 0.0),
                'mae': metrics.get('mae', 0.0),
                'r2_score': metrics.get('r2_score', 0.0),
                'training_time': training_time_str,
                'plots': plots,
                'is_classification': is_classification
            }
            
        except Exception as e:
            logger.error(f"Error en entrenamiento: {str(e)}")
            raise Exception(f"Error en el entrenamiento del modelo: {str(e)}")
    
    def _detect_task_type(self, y):
        """Detecta si es un problema de clasificación o regresión"""
        unique_values = len(y.unique())
        total_values = len(y)
        
        # Si hay pocos valores únicos en relación al total, es clasificación
        if unique_values / total_values < 0.1 or unique_values < 20:
            return True
        return False
    
    def _preprocess_data(self, X, y, is_classification):
        """Preprocesa los datos para el entrenamiento"""
        # Crear copias para no modificar los datos originales
        X_processed = X.copy()
        y_processed = y.copy()
        
        # Manejar valores faltantes
        X_processed = X_processed.fillna(X_processed.mean())
        
        # Codificar variables categóricas
        for column in X_processed.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_processed[column] = le.fit_transform(X_processed[column].astype(str))
            self.label_encoders[column] = le
        
        # Estandarizar características
        self.scaler = StandardScaler()
        X_processed = pd.DataFrame(
            self.scaler.fit_transform(X_processed),
            columns=X_processed.columns
        )
        
        # Codificar variable objetivo si es clasificación
        if is_classification and y_processed.dtype == 'object':
            le_target = LabelEncoder()
            y_processed = le_target.fit_transform(y_processed)
            self.label_encoders['target'] = le_target
        
        return X_processed, y_processed
    
    def _create_model(self, model_type, hyperparameters, is_classification):
        """Crea el modelo según el tipo especificado"""
        if model_type == 'random-forest':
            if is_classification:
                return RandomForestClassifier(**hyperparameters)
            else:
                return RandomForestRegressor(**hyperparameters)
        
        elif model_type == 'linear-regression':
            if is_classification:
                return LogisticRegression(**hyperparameters)
            else:
                return LinearRegression(**hyperparameters)
        
        elif model_type == 'neural-network':
            if is_classification:
                return MLPClassifier(**hyperparameters)
            else:
                return MLPRegressor(**hyperparameters)
        
        elif model_type == 'svm':
            if is_classification:
                return SVC(**hyperparameters)
            else:
                return SVR(**hyperparameters)
        
        elif model_type == 'gradient-boosting':
            if is_classification:
                return GradientBoostingClassifier(**hyperparameters)
            else:
                return GradientBoostingRegressor(**hyperparameters)
        
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    def _calculate_metrics(self, y_true, y_pred, is_classification):
        """Calcula las métricas del modelo"""
        metrics = {}
        
        if is_classification:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        else:
            metrics['r2_score'] = r2_score(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            # Para regresión, accuracy no es aplicable
            metrics['accuracy'] = 0.0
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1_score'] = 0.0
        
        return metrics
    
    def _generate_plots(self, X_test, y_test, y_pred, is_classification):
        """Genera gráficos para visualizar el modelo"""
        plots = {}
        
        try:
            # Asegurar que los datos sean del tipo correcto
            y_test = pd.Series(y_test).astype(float)
            y_pred = pd.Series(y_pred).astype(float)
            
            # Matriz de confusión (solo para clasificación)
            if is_classification:
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Matriz de Confusión')
                plt.ylabel('Valores Reales')
                plt.xlabel('Valores Predichos')
                plots['confusion_matrix'] = self._plot_to_base64()
                plt.close()
            
            # Importancia de características (si el modelo la soporta)
            if hasattr(self.model, 'feature_importances_'):
                plt.figure(figsize=(10, 6))
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                plt.barh(feature_importance['feature'], feature_importance['importance'])
                plt.title('Importancia de Variables')
                plt.xlabel('Importancia')
                plots['feature_importance'] = self._plot_to_base64()
                plt.close()
            
            # Gráfico de dispersión para regresión
            if not is_classification:
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel('Valores Reales')
                plt.ylabel('Valores Predichos')
                plt.title('Valores Reales vs Predichos')
                plots['learning_curve'] = self._plot_to_base64()
                plt.close()
            
        except Exception as e:
            logger.error(f"Error generando gráficos: {str(e)}")
            # Retornar gráficos vacíos si hay error
            plots = {
                'confusion_matrix': None,
                'feature_importance': None,
                'learning_curve': None
            }
        
        return plots
    
    def _plot_to_base64(self):
        """Convierte un gráfico de matplotlib a base64"""
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return image_base64
    
    def predict(self, input_data):
        """Realiza predicciones con el modelo entrenado"""
        if self.model is None:
            raise Exception("El modelo no ha sido entrenado")
        
        try:
            # Convertir input_data a DataFrame
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = pd.DataFrame(input_data)
            
            # Preprocesar datos de la misma manera que en el entrenamiento
            df_processed = df.copy()
            
            # Aplicar codificación de variables categóricas
            for column in df_processed.select_dtypes(include=['object']).columns:
                if column in self.label_encoders:
                    df_processed[column] = self.label_encoders[column].transform(
                        df_processed[column].astype(str)
                    )
            
            # Aplicar escalado
            if self.scaler:
                df_processed = pd.DataFrame(
                    self.scaler.transform(df_processed),
                    columns=df_processed.columns
                )
            
            # Hacer predicción
            prediction = self.model.predict(df_processed)
            
            # Si es clasificación, decodificar la predicción
            if 'target' in self.label_encoders:
                prediction = self.label_encoders['target'].inverse_transform(prediction)
            
            return prediction[0] if len(prediction) == 1 else prediction.tolist()
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            raise Exception(f"Error en la predicción: {str(e)}")
    
    def save_model(self, filepath):
        """Guarda el modelo entrenado"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'target_name': self.target_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Carga un modelo previamente entrenado"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
