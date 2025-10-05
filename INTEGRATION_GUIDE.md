# Guía de Integración Frontend-Backend

Esta guía explica cómo integrar el frontend React con el backend Django para el sistema de Machine Learning.

## 🏗️ Arquitectura del Sistema

```
Frontend (React)          Backend (Django)
┌─────────────────┐       ┌─────────────────┐
│   tree-visualizer-app   │   ml_backend    │
│                         │                 │
│  - Análisis de datos    │  - API REST     │
│  - Entrenamiento ML     │  - Modelos ML   │
│  - Predicciones         │  - Base de datos│
│  - Visualizaciones      │  - Gráficos     │
└─────────────────┘       └─────────────────┘
         │                           │
         └──────── HTTP/JSON ────────┘
```

## 🚀 Configuración Inicial

### 1. Backend Django

```bash
# Navegar al directorio del backend
cd ml_backend

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar proyecto
python setup.py

# Ejecutar servidor
python manage.py runserver
```

El backend estará disponible en: http://localhost:8000

### 2. Frontend React

```bash
# Navegar al directorio del frontend
cd tree-visualizer-app

# Instalar dependencias
npm install

# Ejecutar servidor de desarrollo
npm run dev
```

El frontend estará disponible en: http://localhost:5173

## 🔌 Integración de APIs

### Configuración de la API

El frontend ya incluye la configuración necesaria en `src/lib/api.ts`:

```typescript
const API_BASE_URL = 'http://localhost:8000/api';
```

### Uso del Hook useMLAPI

```typescript
import { useMLAPI } from '../hooks/useMLAPI';

function MyComponent() {
  const { trainModel, predict, loading, error } = useMLAPI();
  
  const handleTrain = async () => {
    const result = await trainModel({
      analysis_type: 'own-data',
      model: 'random-forest',
      hyperparameters: { n_estimators: 100 },
      // ... otros datos
    });
  };
}
```

## 📡 Endpoints Disponibles

### 1. Entrenar Modelo
```
POST /api/train-model/
```

**Datos de entrada:**
```json
{
  "analysis_type": "own-data",
  "model": "random-forest",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10
  },
  "target_variable": "price",
  "input_variables": ["size", "rooms"],
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
    "confusion_matrix": "base64_image",
    "feature_importance": "base64_image"
  }
}
```

### 2. Realizar Predicción
```
POST /api/predict/
```

**Datos de entrada:**
```json
{
  "model_id": 1,
  "input_data": {
    "size": 120,
    "rooms": 3
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

### 3. Listar Modelos
```
GET /api/models/
```

### 4. Obtener Modelo Específico
```
GET /api/models/{id}/
```

## 🔄 Flujo de Trabajo

### 1. Análisis con Datos Propios

1. Usuario selecciona archivo CSV
2. Frontend parsea CSV y extrae columnas
3. Usuario configura variables objetivo y de entrada
4. Usuario selecciona modelo y hiperparámetros
5. Frontend envía datos al backend via `POST /api/train-model/`
6. Backend entrena modelo y retorna métricas + gráficos
7. Frontend muestra resultados y gráficos

### 2. Análisis con Datos del App

1. Usuario selecciona dataset predefinido
2. Usuario selecciona modelo y hiperparámetros
3. Frontend envía configuración al backend
4. Backend usa dataset predefinido para entrenar
5. Backend retorna métricas + gráficos
6. Frontend muestra resultados

### 3. Predicciones

1. Usuario selecciona modelo entrenado
2. Usuario ingresa valores de entrada
3. Frontend envía datos via `POST /api/predict/`
4. Backend realiza predicción
5. Frontend muestra resultado

## 🎨 Visualizaciones

El backend genera automáticamente gráficos en base64:

- **Matriz de Confusión**: Para modelos de clasificación
- **Importancia de Variables**: Para modelos que la soporten
- **Curva de Aprendizaje**: Para modelos de regresión

Para mostrar gráficos en el frontend:

```typescript
import { displayPlot } from '../lib/api';

// Mostrar gráfico
displayPlot(plots.confusion_matrix, 'confusion-matrix-container');
```

## 🛠️ Modelos Soportados

### Clasificación
- Random Forest
- Linear Regression (Logistic)
- Neural Network
- SVM
- Gradient Boosting

### Regresión
- Random Forest
- Linear Regression
- Neural Network
- SVM
- Gradient Boosting

## 🔧 Configuración Avanzada

### CORS
El backend está configurado para aceptar peticiones desde:
- http://localhost:3000
- http://localhost:5173
- http://127.0.0.1:3000
- http://127.0.0.1:5173

### Variables de Entorno
Puedes configurar la URL del backend en `src/lib/api.ts`:

```typescript
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
```

## 🚨 Solución de Problemas

### Error de CORS
Si el frontend no puede conectarse:
1. Verifica que el backend esté ejecutándose
2. Revisa la configuración de CORS en `ml_backend/settings.py`
3. Asegúrate de que la URL del frontend esté en `CORS_ALLOWED_ORIGINS`

### Error de Conexión
1. Verifica que el backend esté en http://localhost:8000
2. Revisa los logs del backend en `ml_backend.log`
3. Usa el componente `BackendStatus` para monitorear la conexión

### Error de Entrenamiento
1. Verifica que los datos CSV sean válidos
2. Revisa que las variables existan en los datos
3. Asegúrate de que los hiperparámetros sean correctos

## 📊 Monitoreo

### Logs del Backend
Los logs se guardan en `ml_backend.log` e incluyen:
- Entrenamientos exitosos
- Errores de procesamiento
- Predicciones realizadas

### Estado de Conexión
El componente `BackendStatus` muestra:
- Estado de conexión en tiempo real
- Botón para reintentar conexión
- Alertas si el backend no está disponible

## 🔄 Próximos Pasos

1. **Integrar APIs en las páginas existentes**:
   - Modificar `Training.tsx` para usar `useMLAPI`
   - Actualizar `Prediction.tsx` para conectar con el backend
   - Agregar `BackendStatus` a la navegación

2. **Mejorar la experiencia de usuario**:
   - Mostrar gráficos generados por el backend
   - Agregar indicadores de progreso
   - Implementar manejo de errores más robusto

3. **Optimizaciones**:
   - Cache de modelos entrenados
   - Validación de datos en el frontend
   - Mejores mensajes de error

## 📞 Soporte

Para problemas o preguntas:
1. Revisa los logs del backend
2. Verifica la configuración de CORS
3. Usa el script de prueba `test_api.py`
4. Consulta la documentación de Django REST Framework
