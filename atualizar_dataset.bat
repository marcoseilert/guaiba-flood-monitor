@echo off
setlocal
set "PROJECT=C:\Users\User\anaconda_projects\Previsao_Nivel_Guaiba_v2"
set "LOG=%PROJECT%\auto_update_task.log"
set "UV_PYTHON=C:\Users\User\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe"

cd /d "%PROJECT%"
if errorlevel 1 exit /b 1

>>"%LOG%" echo.
>>"%LOG%" echo ========================================
>>"%LOG%" echo  Inicio: %date% %time%
>>"%LOG%" echo ========================================

if exist "%UV_PYTHON%" (
    >>"%LOG%" echo Usando Python: %UV_PYTHON%
    "%UV_PYTHON%" auto_update.py >>"%LOG%" 2>&1
) else (
    >>"%LOG%" echo Usando Python do PATH via py -3.11
    py -3.11 auto_update.py >>"%LOG%" 2>&1
)
set "EXIT_CODE=%ERRORLEVEL%"

>>"%LOG%" echo Fim: %date% %time% - codigo %EXIT_CODE%
exit /b %EXIT_CODE%
