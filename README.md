# 🌊 Monitor de Enchentes — Rio Guaíba

Dashboard de previsão de enchentes do Rio Guaíba em Porto Alegre, RS.

## 🚀 Acesso

**URL**: [guaiba-flood-monitor.streamlit.app](https://guaiba-flood-monitor.streamlit.app)

## 📊 Funcionalidades

- Gráfico interativo do nível do Guaíba com projeções T+3 e T+5
- Comparação entre realizado e previsto pelo modelo
- Monitor de variáveis (chuva, vento, nível) com percentis
- Análise de impacto da direção do vento
- Atualização automática dos dados (GitHub Actions)
- Glossário visual para não-técnicos

## 🏗️ Arquitetura

```
app.py                    # Dashboard Streamlit
update_dataset.py         # Atualização incremental de dados
models/                   # Modelos treinados (LightGBM + CatBoost)
data/processed/           # Dataset histórico
docs/                     # Documentação técnica
```

## 📦 Modelos

| Modelo | Algoritmo | Target | Features | RMSE |
|--------|-----------|--------|----------|------|
| delta_3d | LightGBM | Δ acumulado 3 dias | 13 (SFS) | 0.1075m |
| delta_5d | CatBoost | Δ acumulado 5 dias | 13 (SFS) | 0.1413m |

## 🚦 Sistema de Alerta

| Nível | Cota | Ação |
|-------|------|------|
| 🟢 Normal | < 1.0m | — |
| 🟡 Atenção | 1.0 - 2.0m | Monitorar |
| 🟠 Alerta | 2.0 - 3.0m | Preparação |
| 🔴 INUNDAÇÃO | > 3.0m | Evacuação |

## 📖 Documentação

- [Documentação Técnica](docs/DOCUMENTACAO_TECNICA.md)
- [Estudo Direção do Vento](docs/ESTUDO_DIRECAO_VENTO.md)

## 🔄 Atualização

O dataset é atualizado automaticamente via GitHub Actions todo dia às 6h (horário de Brasília).

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

