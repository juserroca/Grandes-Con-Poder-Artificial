# Guía de Logs y Debugging - ML Backend

Esta guía te explica cómo ver y usar los logs del backend Django para debugging y monitoreo.

## 🚀 **Inicio Rápido con Logs**

### **Opción 1: Script de Windows (Recomendado)**
```bash
# Ejecutar desde el directorio ml_backend
start_with_logs.bat
```

### **Opción 2: Script de Linux/Mac**
```bash
# Ejecutar desde el directorio ml_backend
./start_with_logs.sh
```

### **Opción 3: Manual**
```bash
# Activar entorno virtual
venv\Scripts\activate  # Windows
# o
source venv/bin/activate  # Linux/Mac

# Ejecutar Django
python manage.py runserver
```

## 📊 **Dónde Ver los Logs**

### **1. Consola/Terminal (Tiempo Real)**
Los logs aparecen directamente en la terminal donde ejecutas Django:
```
🚀 Iniciando entrenamiento de modelo...
📊 Datos recibidos: {'analysis_type': 'app-data', 'model': 'random-forest'...}
🔍 Validando datos de entrada...
✅ Datos validados correctamente: {'analysis_type': 'app-data'...}
📋 Tipo de análisis: app-data
🌐 Usando dataset predefinido de la aplicación...
📊 Dataset cargado: 1000 filas, 11 columnas
🎯 Variable objetivo: habitable
📝 Variables de entrada: ['koi_period', 'koi_impact'...]
🤖 Iniciando entrenamiento con modelo: random-forest
⚙️ Hiperparámetros: {'n_estimators': 50, 'max_depth': 5...}
✅ Entrenamiento completado!
📈 Accuracy: 0.942
⏱️ Tiempo: 2m 34s
💾 Guardando modelo en la base de datos...
✅ Modelo guardado con ID: 1
💾 Guardando archivo del modelo...
✅ Archivo del modelo guardado: models/model_1.pkl
📤 Preparando respuesta...
🎉 Entrenamiento completado exitosamente: Modelo_kepler_random-forest
📊 Respuesta: {'model_id': 1, 'accuracy': 0.942...}
```

### **2. Archivo de Logs (ml_backend.log)**
Los logs también se guardan en el archivo `ml_backend.log`:

```bash
# Ver logs en tiempo real (Windows PowerShell)
Get-Content ml_backend.log -Wait -Tail 10

# Ver logs en tiempo real (Linux/Mac)
tail -f ml_backend.log

# Ver todo el archivo
type ml_backend.log  # Windows
cat ml_backend.log   # Linux/Mac
```

## 🧪 **Probar la API con Logs Detallados**

### **Script de Prueba Mejorado**
```bash
# Ejecutar el script de prueba con logs detallados
python test_with_logs.py
```

Este script te mostrará:
- ✅ Estado de conexión del servidor
- 📊 Datos enviados y recibidos
- ⏱️ Tiempos de respuesta
- 🎨 Estado de los gráficos generados
- ❌ Errores detallados si los hay

## 🔍 **Tipos de Logs Disponibles**

### **1. Logs de Entrenamiento**
- 🚀 Inicio del proceso
- 📊 Datos recibidos
- 🔍 Validación de datos
- 📋 Tipo de análisis
- 📁 Procesamiento de datos
- 🤖 Configuración del modelo
- ⚙️ Hiperparámetros
- ✅ Resultados del entrenamiento
- 💾 Guardado en base de datos
- 📤 Preparación de respuesta

### **2. Logs de Predicción**
- 🔮 Inicio de predicción
- 📊 Datos de entrada
- 🔍 Validación
- 🤖 Carga del modelo
- 🔮 Resultado de predicción

### **3. Logs de Error**
- ❌ Errores de validación
- ❌ Errores de conexión
- ❌ Errores de entrenamiento
- ❌ Errores de predicción

## 🛠️ **Configuración de Logs**

### **Niveles de Log**
- **INFO**: Información general (por defecto)
- **DEBUG**: Información detallada para debugging
- **WARNING**: Advertencias
- **ERROR**: Errores
- **CRITICAL**: Errores críticos

### **Cambiar Nivel de Log**
Para ver más detalles, modifica `ml_backend/settings.py`:

```python
'loggers': {
    'ml_models': {
        'handlers': ['console', 'file'],
        'level': 'DEBUG',  # Cambiar de INFO a DEBUG
        'propagate': False,
    },
}
```

## 📱 **Monitoreo en Tiempo Real**

### **Opción 1: Dos Terminales**
```bash
# Terminal 1: Ejecutar Django
cd ml_backend
python manage.py runserver

# Terminal 2: Monitorear logs
tail -f ml_backend.log
```

### **Opción 2: Un Solo Terminal (Windows)**
```bash
# Ejecutar Django en background y monitorear logs
start /B python manage.py runserver
Get-Content ml_backend.log -Wait -Tail 10
```

## 🐛 **Debugging Común**

### **Problema: No se ven logs en consola**
**Solución**: Verifica que el logging esté configurado correctamente en `settings.py`

### **Problema: Logs muy verbosos**
**Solución**: Cambia el nivel de `DEBUG` a `INFO` en `settings.py`

### **Problema: Archivo de logs no se crea**
**Solución**: Verifica permisos de escritura en el directorio

### **Problema: Logs de error no aparecen**
**Solución**: Verifica que el logger esté configurado para capturar errores

## 📊 **Ejemplo de Uso Completo**

### **1. Iniciar Servidor con Logs**
```bash
cd ml_backend
start_with_logs.bat
```

### **2. En otra terminal, probar la API**
```bash
cd ml_backend
python test_with_logs.py
```

### **3. Ver logs en tiempo real**
```bash
# En una tercera terminal
Get-Content ml_backend.log -Wait -Tail 20
```

## 🎯 **Logs Específicos por Endpoint**

### **POST /api/train-model/**
```
🚀 Iniciando entrenamiento de modelo...
📊 Datos recibidos: {...}
🔍 Validando datos de entrada...
✅ Datos validados correctamente: {...}
📋 Tipo de análisis: app-data
🌐 Usando dataset predefinido...
📊 Dataset cargado: 1000 filas, 11 columnas
🎯 Variable objetivo: habitable
📝 Variables de entrada: [...]
🤖 Iniciando entrenamiento con modelo: random-forest
⚙️ Hiperparámetros: {...}
✅ Entrenamiento completado!
📈 Accuracy: 0.942
⏱️ Tiempo: 2m 34s
💾 Guardando modelo en la base de datos...
✅ Modelo guardado con ID: 1
💾 Guardando archivo del modelo...
✅ Archivo del modelo guardado: models/model_1.pkl
📤 Preparando respuesta...
🎉 Entrenamiento completado exitosamente
```

### **POST /api/predict/**
```
🔮 Iniciando predicción...
📊 Datos de predicción recibidos: {...}
🔍 Validando datos de predicción...
✅ Datos de predicción validados: {...}
🤖 Cargando modelo ID: 1
🔮 Realizando predicción...
✅ Predicción completada: 1
```

### **GET /api/models/**
```
📋 Obteniendo lista de modelos...
✅ Lista de modelos obtenida: 3 modelos
```

## 💡 **Consejos de Debugging**

1. **Siempre revisa la consola primero** - Los logs más importantes aparecen ahí
2. **Usa el archivo de logs para historial** - Para revisar logs anteriores
3. **Ejecuta el script de prueba** - Para ver el flujo completo
4. **Monitorea en tiempo real** - Para debugging activo
5. **Revisa los códigos de error HTTP** - Para identificar problemas de API

## 🚨 **Solución de Problemas**

### **Error: "No se puede conectar al servidor"**
- Verifica que Django esté ejecutándose
- Revisa que el puerto 8000 esté libre
- Verifica la URL en el script de prueba

### **Error: "Datos inválidos"**
- Revisa el formato de los datos enviados
- Verifica que todos los campos requeridos estén presentes
- Revisa los tipos de datos

### **Error: "Modelo no encontrado"**
- Verifica que el modelo ID exista
- Revisa que el modelo esté guardado correctamente
- Verifica la base de datos

¡Con esta configuración tendrás visibilidad completa de todo lo que sucede en tu backend Django! 🎉

