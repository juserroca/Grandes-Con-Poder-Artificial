# Guía de Debugging para Django ML Backend

## 🐛 Configuración de Debugging

### 1. Debugging con VS Code

1. **Configurar launch.json** (ya creado en `.vscode/launch.json`)
2. **Colocar breakpoints** en el código donde necesites parar
3. **Presionar F5** o usar "Run and Debug" en VS Code
4. **Seleccionar "Django Debug"** en la configuración

### 2. Debugging con PDB (Python Debugger)

Para activar el debugger interactivo, descomenta esta línea en `views.py`:
```python
# pdb.set_trace()
```

Luego ejecuta:
```bash
python debug_with_breakpoints.py
```

### 3. Debugging con Logging Detallado

Ejecuta el servidor con logging detallado:
```bash
python debug_server.py
```

Los logs se guardan en:
- `debug.log` - Logs generales
- `debug_detailed.log` - Logs detallados
- `debug_breakpoints.log` - Logs de breakpoints

### 4. Debugging de Vistas Específicas

Para probar las vistas individualmente:
```bash
python debug_views.py
```

## 🔍 Funciones de Debugging Disponibles

### debug_breakpoint(message, data=None)
- Imprime mensaje de debugging
- Muestra datos estructurados
- Puede activar pdb.set_trace()

### debug_log_request(request, function_name)
- Log detallado de peticiones HTTP
- Muestra headers, método, path
- Analiza datos de la petición

## 📊 Middleware de Debugging

El `DebugMiddleware` automáticamente:
- Logea todas las peticiones entrantes
- Mide tiempo de respuesta
- Captura excepciones
- Logea respuestas JSON

## 🚀 Scripts de Inicio

### ⚠️ IMPORTANTE: Activar Entorno Virtual Primero

**Windows:**
```cmd
cd ml_backend
venv\Scripts\activate
```

**Linux/Mac:**
```bash
cd ml_backend
source venv/bin/activate
```

### Scripts de Debugging (Recomendados)

**Windows:**
```cmd
run_debug.bat
```

**Linux/Mac:**
```bash
./run_debug.sh
```

### Scripts Manuales

**Modo Normal con Debugging:**
```bash
python debug_server.py
```

**Modo con Breakpoints:**
```bash
python debug_with_breakpoints.py
```

### Verificar Entorno
```bash
python check_environment.py
```

## 🔧 Configuración de Logging

### Niveles de Log
- `DEBUG` - Información detallada
- `INFO` - Información general
- `WARNING` - Advertencias
- `ERROR` - Errores
- `CRITICAL` - Errores críticos

### Loggers Específicos
- `django` - Logs de Django
- `ml_models` - Logs de modelos ML
- `debug` - Logs de debugging
- `debug_breakpoints` - Logs de breakpoints

## 🎯 Puntos de Debugging Comunes

### 1. En train_model()
```python
debug_breakpoint("Validando datos de entrada", request.data)
```

### 2. En predict()
```python
debug_breakpoint("Procesando predicción", prediction_data)
```

### 3. En ml_engine.py
```python
debug_breakpoint("Entrenando modelo", model_params)
```

## 📝 Ejemplo de Uso

```python
# En cualquier función de views.py
def mi_funcion(request):
    debug_log_request(request, "mi_funcion")
    
    # Tu código aquí
    data = request.data
    
    debug_breakpoint("Procesando datos", {
        'data_type': type(data),
        'data_keys': list(data.keys()) if isinstance(data, dict) else None
    })
    
    # Descomenta para activar debugger interactivo
    # pdb.set_trace()
    
    # Resto del código...
```

## 🚨 Solución de Problemas

### Error: "No module named 'pdb'"
- PDB está incluido en Python estándar
- Verifica que estés usando Python 3.x

### Error: "Middleware not found"
- Verifica que `DEBUG = True` en settings.py
- Reinicia el servidor después de cambios

### Logs no aparecen
- Verifica permisos de escritura en el directorio
- Asegúrate de que el logger esté configurado correctamente

## 📚 Recursos Adicionales

- [Django Debugging](https://docs.djangoproject.com/en/stable/topics/debugging/)
- [Python PDB](https://docs.python.org/3/library/pdb.html)
- [VS Code Debugging](https://code.visualstudio.com/docs/python/debugging)
