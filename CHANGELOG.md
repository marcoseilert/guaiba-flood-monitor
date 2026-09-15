# Changelog

## 2026-08-15

- Adicionado alerta ntfy no `auto_update.py`: após a atualização/publicação do dataset, a projeção T+5 mais recente acima de 2,50 m é enviada ao tópico `alertas_mosoeilert`.
- Restrito o alerta a horário posterior às 17h BRT — equivalente à execução das 18h no Task Scheduler — com no máximo um envio por dia. O próximo dia libera novo alerta mesmo sem retorno abaixo de 2,50 m. Falhas de envio ficam registradas no log; a atualização do dataset não é interrompida.

## 2026-08-12

- Corrigido o executável usado pelo Windows Task Scheduler: `atualizar_dataset.bat` passa a chamar explicitamente o Python do ambiente Hermes (`C:\\Users\\User\\AppData\\Local\\hermes\\hermes-agent\\venv\\Scripts\\python.exe`), que contém `numpy`, `pandas` e `pyarrow`. O Python do uv não tinha `numpy`, causando falha nas execuções de 11/08/2026.

## 2026-08-10

- Invalidado o cache de dados do Streamlit após a atualização do parquet, para que o dashboard passe a recarregar o dataset publicado quando a versão dos dados mudar.
- Corrigido o cartão de probabilidade do dashboard para tratar linhas sem dados suficientes do modelo binário como **Indisponível**, evitando a exibição de `nan%` e o falso status de risco crítico.
- Corrigida a rotina do Windows Task Scheduler: a tarefa passa a executar `auto_update.py`, usar o Python 3.11 com dependências instaladas, registrar saída em `auto_update_task.log` e terminar sem `pause` interativo.
- Incluído no repositório o artefato `models/binary_model.pkl`, necessário para recalcular a probabilidade de evento extremo durante as atualizações.
- Documentado que a atualização é local, via Task Scheduler às 14h e 18h BRT; o workflow do GitHub Actions permanece desativado.
