# 🌊 Monitor de Enchentes — Rio Guaíba

Dashboard de previsão de enchentes do Rio Guaíba em Porto Alegre, RS.

**URL**: [porto-alegre-flood-monitor.streamlit.app](https://porto-alegre-flood-monitor.streamlit.app)

## 📊 Funcionalidades

- Gráfico interativo do nível do Guaíba com projeção T+5
- Projeção T+5 plotada na data projetada (data-base + 5 dias), com divisor entre dado real e futuro
- **Classificação binária** de probabilidade de evento extremo (Δ > 1m)
- Barras de probabilidade no gráfico com cores por nível de risco
- Painel de variáveis com ordenação dinâmica por severidade
- Contribuição de cada variável ao risco (modelo LogReg+OptBin)
- Badges indicando modelo de cada variável (CB / LR)
- Análise de impacto da direção do vento (rosa dos ventos)
- Ícones dinâmicos de precipitação (☀️⛅🌤️🌧️⛈️)
- Atualização automática dos dados (Windows Task Scheduler, 2x/dia)
- Alerta ntfy quando a projeção T+5 ultrapassa 2,50 m (sem duplicidade enquanto permanecer acima)
- Glossário visual para não-técnicos

## 🏗️ Arquitetura

```
app.py                    # Dashboard Streamlit
update_dataset.py         # Atualização incremental de dados
models/
  model_delta_5d.pkl      # CatBoost (regressão, Δ5d)
  model_delta_3d.pkl      # LightGBM (regressão, Δ3d)
  model_metadata.pkl      # Features + métricas
  binary_model.pkl        # LogReg+OptBin (classificação)
data/processed/
  dataset_historico.parquet  # Dataset unificado (2019-hoje)
  sfs_results_logreg_optbin.json  # Features SFS LogReg
  wind_direction_impact.json      # Impacto direção do vento
docs/                     # Documentação técnica
```

## 📦 Modelos

### Regressão (projeção de nível)

| Modelo | Algoritmo | Target | Features | RMSE |
|--------|-----------|--------|----------|------|
| delta_5d | CatBoost | Δ acumulado 5 dias | 13 (SFS) | 0.1413m |
| delta_3d | LightGBM | Δ acumulado 3 dias | 13 (SFS) | 0.1075m |

### Classificação (probabilidade de evento extremo)

| Modelo | Algoritmo | Target | Features | AUC | KS |
|--------|-----------|--------|----------|-----|-----|
| binário | LogReg+OptBin | P(Δ5d > 1m) | 7 (SFS) | 0.9949 | 0.9859 |

## 🚦 Sistema de Alerta

### Nível do Guaíba

| Nível | Cota | Ação |
|-------|------|------|
| 🟢 Normal | < 1.0m | — |
| 🟡 Atenção | 1.0 - 2.0m | Monitorar |
| 🟠 Alerta | 2.0 - 3.0m | Preparação |
| 🔴 INUNDAÇÃO | > 3.0m | Evacuação |

### Probabilidade de evento extremo

| Probabilidade | Status | Ação |
|---------------|--------|------|
| < 1% | Normal | — |
| 1 - 5% | Atenção | Monitorar |
| 5 - 20% | Alerta Precoce | Preparação |
| > 20% | Risco Crítico | Ação imediata |

## 🔄 Atualização

O dataset é atualizado automaticamente no Windows via **Task Scheduler, 2x ao dia** (14h e 18h, horário de Brasília). A tarefa chama `auto_update.py`, que busca os dados, recalcula as previsões e publica o dataset no GitHub. O workflow do GitHub Actions permanece desativado porque a API da ANA não funciona de forma confiável no runner do GitHub.

Após cada atualização, `auto_update.py` verifica o horário de Brasília e só permite o alerta quando o horário for **posterior às 17h**. Como o Task Scheduler executa às 14h e 18h, o alerta ocorre somente na atualização das 18h. Se a projeção T+5 mais recente for **maior que 2,50 m**, uma mensagem é enviada ao tópico ntfy `alertas_mosoeilert` em `https://ntfy.sh`. É permitido um alerta por dia: o estado local registra a data do último envio em `.ntfy_alert_state.json`; no dia seguinte, um novo alerta será enviado se a condição continuar verdadeira, sem exigir retorno abaixo do limite. Para substituir servidor, tópico ou limite, podem ser usadas as variáveis `NTFY_SERVER`, `NTFY_TOPIC` e `NTFY_THRESHOLD_M`.

Para atualizar manualmente:
```bash
python auto_update.py
```

## 🛠️ Desenvolvimento

```bash
# Instalar dependências
pip install -r requirements.txt

# Rodar localmente
streamlit run app.py

# Atualizar dados e publicar no GitHub
python auto_update.py
```

## 📖 Documentação

- [Documentação Técnica](docs/DOCUMENTACAO_TECNICA.md)
- [Estudo Direção do Vento](docs/ESTUDO_DIRECAO_VENTO.md)

## 📝 Nota

Este é um projeto pessoal sem grandes pretensões. O autor é estatístico e cientista de dados com experiência em modelagem preditiva, mas não é especialista em hidrologia ou meteorologia. Eventuais imprecisões técnicas são de sua responsabilidade.
