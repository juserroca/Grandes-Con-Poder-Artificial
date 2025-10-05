@echo off
REM Script de inicio rápido para ML Backend (Windows)
echo 🚀 Iniciando ML Backend Django...

REM Verificar si Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python no está instalado. Por favor instala Python 3.8+
    pause
    exit /b 1
)

REM Verificar si pip está instalado
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip no está instalado. Por favor instala pip
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

REM Configurar Django
echo ⚙️ Configurando Django...
python setup.py

REM Iniciar servidor
echo 🌐 Iniciando servidor Django...
echo    URL: http://localhost:8000
echo    Admin: http://localhost:8000/admin
echo    API: http://localhost:8000/api/
echo.
echo Presiona Ctrl+C para detener el servidor
python manage.py runserver

pause
