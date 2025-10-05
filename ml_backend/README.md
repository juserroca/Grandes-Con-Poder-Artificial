# ML Backend - Django API para Machine Learning

Este es el backend Django que se conecta con el frontend React para gestionar el entrenamiento y predicción de modelos de machine learning.

## 🚀 Características

- **Entrenamiento de modelos ML**: Random Forest, Linear Regression, Neural Networks, SVM, Gradient Boosting
- **Predicciones en tiempo real**: API para realizar predicciones con modelos entrenados
- **Visualizaciones**: Generación automática de gráficos (matriz de confusión, importancia de variables, etc.)
- **Datos propios y del App**: Soporte para datos CSV del usuario y datasets predefinidos
- **API REST**: Endpoints bien documentados para integración con frontend

## 📋 Requisitos

- Python 3.8+
- pip
- virtualenv (recomendado)

## 🛠️ Instalación

### 1. Clonar y navegar al directorio
```bash
cd ml_backend
```

### 2. Crear entorno virtual
```bash
python -m venv venv
```

### 3. Activar entorno virtual
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 4. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 5. Configurar el proyecto
```bash
python setup.py
```

### 6. Ejecutar el servidor
```bash
python manage.py runserver
```

El servidor estará disponible en: http://localhost:8000

## 📚 API Endpoints

### Entrenar Modelo
```
POST /api/train-model/
```

**Datos de entrada:**
```json
{
  "analysis_type": "own-data" | "app-data",
  "model": "random-forest",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10
  },
  "target_variable": "price",
  "input_variables": ["size", "rooms", "location"],
  "csv_data": [...],
  "csv_columns": [...],
  "file_name": "data.csv"
}
```

**Respuesta:**
```json
{
  "model_id": 1,
  "accuracy": 0.942,
  "precision": 0.928,
  "recall": 0.935,
  "f1_score": 0.931,
  "mae": 0.087,
  "r2_score": 0.941,
  "training_time": "2m 34s",
  "plots": {
    "confusion_matrix": "base64_encoded_image",
    "feature_importance": "base64_encoded_image",
    "learning_curve": "base64_encoded_image"
  }
}
```

### Realizar Predicción
```
POST /api/predict/
```

**Datos de entrada:**
```json
{
  "model_id": 1,
  "input_data": {
    "size": 120,
    "rooms": 3,
    "location": "downtown"
  }
}
```

**Respuesta:**
```json
{
  "prediction_id": 1,
  "prediction": 250000,
  "model_name": "Modelo_data.csv_random-forest",
  "model_accuracy": 0.942
}
```

### Listar Modelos
```
GET /api/models/
```

### Obtener Modelo Específico
```
GET /api/models/{id}/
```

## 🔧 Configuración

### Variables de Entorno
Puedes configurar las siguientes variables en `ml_backend/settings.py`:

- `DEBUG`: Modo debug (True/False)
- `ALLOWED_HOSTS`: Hosts permitidos
- `CORS_ALLOWED_ORIGINS`: Orígenes permitidos para CORS

### Base de Datos
Por defecto usa SQLite. Para cambiar a PostgreSQL o MySQL, modifica la configuración en `settings.py`.

## 📊 Modelos Soportados

### Clasificación
- **Random Forest**: Para problemas complejos con muchas variables
- **Linear Regression (Logistic)**: Para relaciones lineales simples
- **Neural Network**: Para patrones no lineales complejos
- **SVM**: Para clasificación con márgenes claros
- **Gradient Boosting**: Alto rendimiento en competencias

### Regresión
- **Random Forest**: Regresión robusta
- **Linear Regression**: Relaciones lineales
- **Neural Network**: Patrones complejos
- **SVM**: Regresión con kernels
- **Gradient Boosting**: Alto rendimiento

## 🎨 Visualizaciones

El sistema genera automáticamente:

1. **Matriz de Confusión**: Para modelos de clasificación
2. **Importancia de Variables**: Para modelos que la soporten
3. **Curva de Aprendizaje**: Para modelos de regresión
4. **Gráficos de Dispersión**: Valores reales vs predichos

## 🔍 Logs

Los logs se guardan en `ml_backend.log` y incluyen:
- Entrenamientos exitosos
- Errores de procesamiento
- Predicciones realizadas

## 🚨 Solución de Problemas

### Error de CORS
Si el frontend no puede conectarse, verifica que la URL esté en `CORS_ALLOWED_ORIGINS`.

### Error de Dependencias
Asegúrate de que todas las dependencias estén instaladas:
```bash
pip install -r requirements.txt
```

### Error de Base de Datos
Si hay problemas con la base de datos, recrea las migraciones:
```bash
python manage.py makemigrations
python manage.py migrate
```

## 📞 Soporte

Para problemas o preguntas, revisa los logs en `ml_backend.log` o contacta al equipo de desarrollo.

## 🔄 Integración con Frontend

Este backend está diseñado para trabajar con el frontend React. Asegúrate de que:

1. El frontend esté ejecutándose en un puerto permitido en CORS
2. Las URLs de la API coincidan con las configuradas
3. Los datos enviados sigan el formato esperado por los serializers

## 📈 Próximas Mejoras

- [ ] Soporte para más algoritmos de ML
- [ ] Optimización de hiperparámetros automática
- [ ] Validación cruzada
- [ ] Exportación de modelos en diferentes formatos
- [ ] Dashboard de monitoreo de modelos
- [ ] API de versionado
