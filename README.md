# 🌟 Tree Visualizer App - Machine Learning Platform

Una aplicación web completa para análisis de datos, entrenamiento de modelos de machine learning y predicciones, con un enfoque especial en la detección de exoplanetas.

## 🚀 Características Principales

### 📊 Análisis de Datos
- **Carga de archivos CSV**: Subida y procesamiento de datasets personalizados
- **Análisis exploratorio**: Visualización automática de datos y estadísticas
- **Selección de variables**: Configuración dinámica de variables objetivo y de entrada
- **Tipos de datos**: Configuración automática y manual de tipos de datos
- **Preprocesamiento**: Manejo de valores faltantes, detección de outliers, escalado

### 🤖 Machine Learning
- **Múltiples algoritmos**: Random Forest, Linear Regression, Neural Networks, SVM, Gradient Boosting
- **Hiperparámetros**: Configuración personalizada de parámetros del modelo
- **Balanceo de clases**: Técnicas de oversampling y undersampling con sklearn
- **Validación**: Métricas de evaluación completas (accuracy, precision, recall, F1-score)
- **Guardado de modelos**: Persistencia en archivos .pkl con metadatos

### 🔮 Predicciones
- **Formulario individual**: Predicciones punto a punto con interfaz intuitiva
- **Predicción masiva**: Procesamiento de archivos CSV completos
- **Resultados detallados**: Probabilidades y clasificaciones con confianza

## 🏗️ Arquitectura del Sistema

### Frontend (React + TypeScript)
```
src/
├── pages/
│   ├── Home.tsx          # Página principal
│   ├── Analysis.tsx      # Análisis de datos
│   ├── Training.tsx      # Entrenamiento de modelos
│   └── Prediction.tsx    # Predicciones
├── components/
│   └── ui/               # Componentes shadcn/ui
├── hooks/
│   └── useAnalysisData.ts # Gestión de estado global
└── lib/
    └── utils.ts          # Utilidades
```

### Backend (Django + Python)
```
ml_backend/
├── ml_models/
│   ├── views.py          # API endpoints
│   ├── ml_engine.py      # Motor de ML
│   ├── preprocesamiento.py # Pipeline de datos
│   └── models.py         # Modelos Django
├── media/
│   └── models/           # Archivos .pkl guardados
└── requirements.txt      # Dependencias Python
```

## 🛠️ Tecnologías Utilizadas

### Frontend
- **React 18** - Framework de UI
- **TypeScript** - Tipado estático
- **Vite** - Build tool y dev server
- **shadcn/ui** - Componentes de UI
- **Tailwind CSS** - Estilos
- **React Router** - Navegación
- **Lucide React** - Iconos

### Backend
- **Django 4.2** - Framework web
- **Django REST Framework** - API REST
- **scikit-learn** - Machine Learning
- **pandas** - Manipulación de datos
- **numpy** - Computación numérica
- **matplotlib/seaborn** - Visualización
- **pickle** - Serialización de modelos

## 🚀 Instalación y Configuración

### Prerrequisitos
- Node.js 18+ y npm
- Python 3.8+
- pip (gestor de paquetes Python)

### 1. Clonar el Repositorio
```bash
git clone <YOUR_GIT_URL>
cd tree-visualizer-app
```

### 2. Configurar Frontend
```bash
# Instalar dependencias
npm install

# Iniciar servidor de desarrollo
npm run dev
```

### 3. Configurar Backend
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

# Aplicar migraciones
python manage.py makemigrations
python manage.py migrate

# Iniciar servidor Django
python manage.py runserver
```

### 4. Acceder a la Aplicación
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **Admin Django**: http://localhost:8000/admin

## 📋 Uso de la Aplicación

### 1. Análisis de Datos
1. **Cargar datos**: Sube tu archivo CSV o usa el dataset de exoplanetas
2. **Configurar variables**: Selecciona variable objetivo y variables de entrada
3. **Tipos de datos**: Configura los tipos de datos para cada variable
4. **Previsualización**: Revisa los datos procesados

### 2. Entrenamiento de Modelos
1. **Seleccionar algoritmo**: Elige entre Random Forest, SVM, Neural Networks, etc.
2. **Configurar hiperparámetros**: Ajusta parámetros del modelo
3. **Entrenar**: Ejecuta el entrenamiento y observa el progreso
4. **Resultados**: Revisa métricas y gráficos de evaluación

### 3. Predicciones
1. **Formulario individual**: Ingresa valores para predicción punto a punto
2. **Predicción masiva**: Sube archivo CSV para predicciones en lote
3. **Resultados**: Obtén probabilidades y clasificaciones

## 🔧 API Endpoints

### Entrenamiento
```
POST /api/train-model/
Content-Type: application/json

{
  "analysis_type": "own-data",
  "model": "random-forest",
  "hyperparameters": {...},
  "target_variable": "koi_disposition",
  "input_variables": [...],
  "csv_data": [...],
  "csv_columns": [...],
  "column_types": {...}
}
```

### Predicción
```
POST /api/predict/
Content-Type: application/json

{
  "model_id": 1,
  "input_data": {
    "koi_period": 365.25,
    "koi_impact": 0.1,
    ...
  }
}
```

## 📁 Estructura de Archivos

```
tree-visualizer-app/
├── src/                    # Frontend React
├── ml_backend/            # Backend Django
│   ├── ml_models/         # App principal de ML
│   ├── media/models/      # Modelos guardados (.pkl)
│   └── requirements.txt   # Dependencias Python
├── public/                # Archivos estáticos
├── package.json          # Dependencias Node.js
└── README.md             # Este archivo
```

## 🧪 Testing

### Frontend
```bash
npm run test
```

### Backend
```bash
cd ml_backend
python test_model_saving.py      # Prueba guardado de modelos
python test_training_endpoint.py # Prueba endpoint completo
python test_prediction.py        # Prueba predicciones
```

## 🚀 Despliegue

### Frontend (Lovable)
1. Abre [Lovable](https://lovable.dev/projects/e3c8acb5-4078-4cc9-b692-477027bb7f20)
2. Click en Share → Publish

### Backend (Producción)
```bash
# Configurar variables de entorno
export DEBUG=False
export ALLOWED_HOSTS=yourdomain.com

# Instalar dependencias de producción
pip install gunicorn

# Ejecutar servidor
gunicorn ml_backend.wsgi:application
```

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🆘 Soporte

Si tienes problemas o preguntas:

1. Revisa la documentación
2. Busca en los issues existentes
3. Crea un nuevo issue con detalles del problema

## 🎯 Roadmap

- [ ] Soporte para más algoritmos de ML
- [ ] Visualizaciones interactivas avanzadas
- [ ] Exportación de resultados en múltiples formatos
- [ ] Integración con bases de datos externas
- [ ] API de autenticación y usuarios
- [ ] Dashboard de monitoreo de modelos

---

**Desarrollado con ❤️ para la detección de exoplanetas y análisis de datos**