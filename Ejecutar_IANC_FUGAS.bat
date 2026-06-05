@echo off
setlocal

set "PROJECT_DIR=D:\IANC_FUGAS"

cd /d "%PROJECT_DIR%"
if errorlevel 1 (
    echo No se pudo abrir la carpeta del proyecto: %PROJECT_DIR%
    pause
    exit /b 1
)

echo.
echo Actualizando desde GitHub...
where git >nul 2>&1
if errorlevel 1 (
    echo Git no esta disponible. Se continuara con la copia local.
) else (
    git pull --ff-only
    if errorlevel 1 (
        echo No se pudo actualizar automaticamente. Se continuara con la copia local.
    )
)

echo.
echo Preparando Python...
where py >nul 2>&1
if errorlevel 1 (
    set "PYTHON_CMD=python"
) else (
    set "PYTHON_CMD=py"
)

echo.
echo Instalando dependencias...
%PYTHON_CMD% -m pip install -r requirements.txt
if errorlevel 1 (
    echo Error instalando dependencias.
    pause
    exit /b 1
)

echo.
echo Iniciando IANC_FUGAS...
%PYTHON_CMD% -m streamlit run app.py

pause
endlocal
