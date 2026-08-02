# 🌊 Monitor de Enchentes — Rio Guaíba

Dashboard de previsão de enchentes do Rio Guaíba em Porto Alegre, RS.

**URL**: [porto-alegre-flood-monitor.streamlit.app](https://porto-alegre-flood-monitor.streamlit.app)

## 📊 Funcionalidades

- Gráfico interativo do nível do Guaíba com projeção T+5
- Comparação entre realizado e previsto pelo modelo
- **Classificação binária** de probabilidade de evento extremo (Δ > 1m)
- Barras de probabilidade no gráfico com cores por nível de risco
- Painel de variáveis com ordenação dinâmica por severidade
- Contribuição de cada variável ao risco (modelo LogReg+OptBin)
- Badges indicando modelo de cada variável (CB / LR)
- Análise de impacto da direção do vento (rosa dos ventos)
- Ícones dinâmicos de precipitação (☀️⛅🌤️🌧️⛈️)
- Atualização automática dos dados (GitHub Actions, 4x/dia)
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

O dataset é atualizado automaticamente via GitHub Actions **4x ao dia** (02h, 08h, 14h, 20h horário de Brasília).

Para atualizar manualmente:
```bash
python update_dataset.py
```

## 🛠️ Desenvolvimento

```bash
# Instalar dependências
pip install -r requirements.txt

# Rodar localmente
streamlit run app.py

# Atualizar dados
python update_dataset.py
```

## 📖 Documentação

- [Documentação Técnica](docs/DOCUMENTACAO_TECNICA.md)
- [Estudo Direção do Vento](docs/ESTUDO_DIRECAO_VENTO.md)

## 📝 Nota

Este é um projeto pessoal sem grandes pretensões. O autor é estatístico e cientista de dados com experiência em modelagem preditiva, mas não é especialista em hidrologia ou meteorologia. Eventuais imprecisões técnicas são de sua responsabilidade.
