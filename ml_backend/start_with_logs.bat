@echo off
REM Script de inicio con logs para Windows
echo 🚀 ML BACKEND - SERVIDOR DJANGO CON LOGS
echo ================================================
echo Iniciando servidor Django con logging detallado...
echo Los logs aparecerán en tiempo real en esta consola
echo ================================================
echo.

REM Verificar que Python esté instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python no está instalado
    echo    Por favor instala Python 3.8+
    pause
    exit /b 1
)

REM Verificar que pip esté instalado
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: pip no está instalado
    echo    Por favor instala pip
    pause
    exit /b 1
)

REM Crear entorno virtual si no existe
if not exist "venv" (
    echo 📦 Creando entorno virtual...
    python -m venv venv
)

REM Activar entorno virtual
echo 🔧 Activando entorno virtual...
call venv\Scripts\activate

REM Instalar dependencias
echo 📚 Instalando dependencias...
pip install -r requirements.txt

REM Configurar Django si es necesario
if not exist "db.sqlite3" (
    echo ⚙️ Configurando Django...
    python setup.py
)

echo.
echo 🌐 El servidor estará disponible en: http://localhost:8000
echo 📊 API disponible en: http://localhost:8000/api/
echo 🔧 Panel de administración: http://localhost:8000/admin/
echo 📝 Logs guardados en: ml_backend.log
echo.
echo 💡 Presiona Ctrl+C para detener el servidor
echo ================================================
echo.

REM Ejecutar Django con logs
echo 🔧 Iniciando servidor Django...
python manage.py runserver

pause

