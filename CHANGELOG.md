# Changelog

## 2026-08-10

- Corrigido o cartão de probabilidade do dashboard para tratar linhas sem dados suficientes do modelo binário como **Indisponível**, evitando a exibição de `nan%` e o falso status de risco crítico.
- Corrigida a rotina do Windows Task Scheduler: a tarefa passa a executar `auto_update.py`, usar o Python 3.11 com dependências instaladas, registrar saída em `auto_update_task.log` e terminar sem `pause` interativo.
- Documentado que a atualização é local, via Task Scheduler às 14h e 18h BRT; o workflow do GitHub Actions permanece desativado.
