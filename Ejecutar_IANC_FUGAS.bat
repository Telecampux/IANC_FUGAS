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
set "DEPS_STAMP=%PROJECT_DIR%\.deps_installed"
set "INSTALL_DEPS=0"

if not exist "%DEPS_STAMP%" (
    set "INSTALL_DEPS=1"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "if ((Get-Item 'requirements.txt').LastWriteTime -gt (Get-Item '%DEPS_STAMP%').LastWriteTime) { exit 1 }"
    if errorlevel 1 set "INSTALL_DEPS=1"
)

if "%INSTALL_DEPS%"=="1" (
    echo Instalando dependencias...
    %PYTHON_CMD% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Error instalando dependencias.
        pause
        exit /b 1
    )
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Content -Path '%DEPS_STAMP%' -Value (Get-Date -Format s)"
) else (
    echo Dependencias ya instaladas. Se omite instalacion.
)

echo.
echo Iniciando IANC_FUGAS...
%PYTHON_CMD% -m streamlit run app.py

pause
endlocal
