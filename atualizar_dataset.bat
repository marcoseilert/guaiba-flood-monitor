@echo off
echo ========================================
echo  Atualizando dataset Guaiba
echo ========================================
cd /d C:\Users\User\anaconda_projects\Previsao_Nivel_Guaiba_v2

echo.
echo [1/3] Rodando update_dataset.py...
python update_dataset.py
if errorlevel 1 (
    echo ERRO no update_dataset.py
    pause
    exit /b 1
)

echo.
echo [2/3] Commitando no git...
git add data/processed/dataset_historico.parquet
git commit -m "data: auto-update %date% %time%"
if errorlevel 1 (
    echo Nada para commitar (ja atualizado)
)

echo.
echo [3/3] Enviando para GitHub...
git push
if errorlevel 1 (
    echo ERRO no push
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Atualizacao concluida!
echo ========================================
pause
