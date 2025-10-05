@echo off
echo ========================================
echo    INICIANDO DJANGO DEBUG MODE
echo ========================================
echo.

REM Verificar si existe el entorno virtual
if not exist "venv\Scripts\activate.bat" (
    echo ❌ ERROR: No se encontró el entorno virtual
    echo 💡 Ejecuta: python -m venv venv
    echo 💡 Luego: venv\Scripts\activate
    echo 💡 Finalmente: pip install -r requirements.txt
    pause
    exit /b 1
)

echo 🔧 Activando entorno virtual...
call venv\Scripts\activate.bat

echo.
echo ✅ Entorno virtual activado
echo 📁 Directorio: %CD%
echo 🐍 Python: %VIRTUAL_ENV%\Scripts\python.exe
echo.

REM Verificar que las dependencias estén instaladas
echo 🔍 Verificando dependencias...
python -c "import django; print('✅ Django:', django.get_version())" 2>nul
if errorlevel 1 (
    echo ❌ ERROR: Django no está instalado
    echo 💡 Ejecuta: pip install -r requirements.txt
    pause
    exit /b 1
)

python -c "import pandas; print('✅ Pandas:', pandas.__version__)" 2>nul
if errorlevel 1 (
    echo ❌ ERROR: Pandas no está instalado
    echo 💡 Ejecuta: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo 🚀 Iniciando servidor Django en modo DEBUG...
echo 📊 Logs se guardarán en: debug.log
echo 🌐 Servidor disponible en: http://localhost:8000
echo.
echo 💡 Para detener el servidor presiona Ctrl+C
echo ========================================
echo.

REM Ejecutar el servidor de debugging
python debug_server.py

pause
