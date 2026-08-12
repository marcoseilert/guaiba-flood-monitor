@echo off
setlocal
set "PROJECT=C:\Users\User\anaconda_projects\Previsao_Nivel_Guaiba_v2"
set "LOG=%PROJECT%\auto_update_task.log"
set "HERMES_PYTHON=C:\Users\User\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"

cd /d "%PROJECT%"
if errorlevel 1 exit /b 1

>>"%LOG%" echo.
>>"%LOG%" echo ========================================
>>"%LOG%" echo  Inicio: %date% %time%
>>"%LOG%" echo ========================================

if exist "%HERMES_PYTHON%" (
    >>"%LOG%" echo Usando Python Hermes: %HERMES_PYTHON%
    "%HERMES_PYTHON%" auto_update.py >>"%LOG%" 2>&1
) else (
    >>"%LOG%" echo ERRO: Python Hermes nao encontrado: %HERMES_PYTHON%
    exit /b 1
)
set "EXIT_CODE=%ERRORLEVEL%"

>>"%LOG%" echo Fim: %date% %time% - codigo %EXIT_CODE%
exit /b %EXIT_CODE%
