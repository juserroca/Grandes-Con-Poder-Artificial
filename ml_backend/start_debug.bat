@echo off
echo Iniciando Django en modo DEBUG con breakpoints...
echo.

REM Activar entorno virtual
call venv\Scripts\activate

REM Ejecutar con configuración de debugging
python debug_with_breakpoints.py

pause
