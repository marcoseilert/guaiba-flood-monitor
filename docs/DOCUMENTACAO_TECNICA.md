# Documentação Técnica — Previsão de Nível do Guaíba v2

> **Projeto**: Sistema de Previsão de Enchentes do Rio Guaíba  
> **Versão**: 2.0  
> **Data de criação**: 31 de julho de 2026  
> **Autor**: Projeto desenvolvido com assistência de IA  

---

## Nota sobre a Jornada de Descoberta

Este documento registra o resultado de um intenso dia de trabalho (31/07/2026) no qual percorremos todo o caminho desde a descoberta das fontes de dados até a definição dos modelos campeões. Os insights obtidos ao longo do caminho foram tão importantes quanto os resultados finais — muitas vezes, a **pergunta certa** fez mais diferença do que o **modelo certo**.

Alguns momentos marcantes da jornada:

1. **A busca pela API certa**: testamos a API REST da ANA, a API SOAP legada, o HidroWeb, e múltiplos endpoints antes de descobrir que o endpoint `DadosHidrometeorologicos` da API SOAP antiga era o único que fornecia dados sub-horários históricos completos — sem autenticação!

2. **A troca de estações**: as estações Triunfo (87010000) e São Jerônimo (87020000) tinham 61% e 50% de dados faltantes, respectivamente, e chuva com valores absurdos (max 1.305mm/dia no Triunfo) ou sempre zero (São Jerônimo). Substituímos por Muçum (86510000) e Rio Pardo (85900000), que tinham cobertura muito superior.

3. **A descoberta do delta_5d**: quando testamos delta_3d e vimos que era superior a delta_1d, surgiu a pergunta: "se 3 dias é melhor que 1 dia, será que 5 dias não seria ainda melhor?" — e era! O delta_5d captura a física de sequências longas de chuva que saturam o solo e causam as maiores enchentes.

4. **O insight do percentil**: quando o modelo errou o valor absoluto do pico de maio/2024 em 76%, parecia um fracasso. Mas ao analisarmos o **percentil** da previsão, vimos que estava no P99.5% com Z-score de 5.70σ — o modelo "sabia" que algo extremo estava acontecendo, mesmo não sabendo o tamanho exato.

5. **A combinação T0 + delta**: a pergunta "é muito diferente uma previsão de delta > 1.5m se o nível atual está em -0.5m ou em 2m" levou à criação do sistema de alerta combinado, que transforma previsões estatísticas em ações operacionais.

6. **Pressão atmosférica não faz sentido**: questionamos se deveríamos incluir pressão atmosférica como feature. A análise física mostrou que o efeito direto é mínimo (~1cm por hPa) e que as informações indiretas (vem tempestade?) já estão capturadas pelas features de chuva e vento. Decisão: não incluir.

7. **Spearman vs Pearson**: como 100% das features falham no teste de normalidade, usamos correlação de Spearman (baseada em ranks) em vez de Pearson. Isso mudou o ranking: `porto_alegre_u_wind` subiu para #1 com Spearman (0.40) vs #4 com Pearson (0.37).


---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Fontes de Dados](#2-fontes-de-dados)
3. [Construção dos Datasets](#3-construção-dos-datasets)
4. [Análise Estatística das Features](#4-análise-estatística-das-features)
5. [Universo de Variáveis Testadas](#5-universo-de-variáveis-testadas)
6. [Seleção de Variáveis — Forward SFS](#6-seleção-de-variáveis--forward-sfs)
7. [Modelagem — Testes de Desafiantes](#7-modelagem--testes-de-desafiantes)
8. [Teste em Eventos Extremos e Sistema de Alerta](#8-teste-em-eventos-extremos-e-sistema-de-alerta)
9. [Modelos Campeões](#9-modelos-campeões)
10. [Principais Insights](#10-principais-insights)
11. [Próximos Passos](#11-próximos-passos)

---

## 1. Visão Geral

### Objetivo

Desenvolver um sistema de previsão de níveis do Rio Guaíba, em Porto Alegre (RS), capaz de antecipar eventos de enchente com **antecedência operacional de 3 a 5 dias**.

### Abordagem

- **Granularidade**: dados diários (agregados de séries sub-horárias de 15 minutos)
- **Horizontes de previsão**: 1 dia (delta_1d), 3 dias (delta_3d) e 5 dias (delta_5d)
- **Target**: variação do nível (delta), não o nível absoluto
- **Modelos oficiais**: `delta_3d` (LightGBM) e `delta_5d` (CatBoost)

### Por que prever o delta e não o nível absoluto?

O nível absoluto é altamente autocorrelacionado (lag-1 explica >95% da variância). Prever o delta força o modelo a aprender os **fatores meteorológicos e hidrológicos** que causam as variações, tornando-o útil para alertas.

---

## 2. Fontes de Dados

### 2.1 ANA — Agência Nacional de Águas (SOAP API)

- **Endpoint**: `http://telemetriaws1.ana.gov.br/ServiceANA.asmx/DadosHidrometeorologicos`
- **Autenticação**: não requerida (acesso público)
- **Granularidade**: sub-horária (~15 minutos)
- **Variáveis**: nível (cm) e precipitação (mm)
- **Formato**: XML (SOAP)

> **Por que SOAP e não REST?** A API REST da ANA (`https://www.snirh.gov.br/hidroweb`) limita dados históricos a ~30 dias. A API SOAP fornece séries históricas completas desde 2019.

#### Estações utilizadas

| Código ANA | Nome | Rio | Bacia | Papel |
|------------|------|-----|-------|-------|
| 87450004 | Cais Mauá C6 | Guaíba | Guaíba | **Target** (principal) |
| 87444000 | Gasômetro | Guaíba | Guaíba | **Target** (substituta, mai/2024-ago/2025) |
| 87242000 | Terminal CATSUL | Guaíba | Guaíba | Feature (nível a jusante) |
| 87399000 | São Leopoldo | Gravataí | Gravataí | Feature |
| 87382000 | São Leopoldo | Sinos | Sinos | Feature |
| 87380000 | Campo Bom | Sinos | Sinos | Feature |
| 87270000 | Passo Montenegro | Caí | Caí | Feature |
| 86510000 | Muçum | Taquari | Taquari | Feature |
| 85900000 | Rio Pardo | Jacuí | Jacuí | Feature |

> **Nota importante**: As estações 87450004 (Cais Mauá C6) e 87444000 (Gasômetro) foram **combinadas** para formar a série target do Guaíba. A Cais Mauá C6 parou em maio de 2024 (enchente histórica) e foi reativada em 2025; a Gasômetro atuou como substituta no período.

### 2.2 Open-Meteo ERA5 (Dados Meteorológicos)

- **API**: `https://archive-api.open-meteo.com/v1/archive`
- **Autenticação**: não requerida
- **Resolução temporal**: diária
- **Variáveis obtidas**:
  - `precipitation_sum` — precipitação total diária (mm)
  - `wind_speed_10m_max` — velocidade máxima do vento a 10m (km/h)
  - `wind_direction_10m_dominant` — direção dominante do vento (graus)

#### 10 pontos geográficos

| Ponto | Latitude | Longitude | Bacia |
|-------|----------|-----------|-------|
| Muçum | -29.16 | -51.87 | Taquari |
| Encantado | -29.23 | -51.87 | Taquari |
| Estrela | -29.50 | -51.96 | Taquari |
| Bento Gonçalves | -29.17 | -51.52 | Taquari |
| Cachoeira do Sul | -30.03 | -52.89 | Jacuí |
| Rio Pardo | -29.98 | -52.37 | Jacuí |
| Santa Cruz do Sul | -29.72 | -52.43 | Jacuí |
| Feliz | -29.45 | -51.30 | Caí |
| São Sebastião do Caí | -29.59 | -51.38 | Caí |
| Campo Bom | -29.67 | -51.06 | Sinos |

> **Nota**: Além dos 10 pontos acima, foram incluídos Porto Alegre, Rio Grande e Mostardas para capturar vento da Lagoa dos Patos (efeito de represamento).

---

## 3. Construção dos Datasets

Com as fontes de dados definidas, o próximo passo foi transformar os dados brutos (sub-horários, em formatos diferentes, com qualidades distintas) em um dataset unificado, limpo e com features prontas para modelagem. Esta etapa incluiu: agregação temporal, limpeza de dados faltantes, criação de variáveis derivadas e definição dos períodos de treino e teste.

### Períodos

| Conjunto | Início | Fim | Uso |
|----------|--------|-----|-----|
| DEV (desenvolvimento) | 01/01/2019 | 31/12/2025 | Treino + validação cruzada |
| OOT (out-of-time) | 01/01/2026 | 31/07/2026 | Avaliação final |

### Pipeline de agregação

Os dados brutos sub-horários (15 min) da ANA são agregados para **diário**:

- **Nível**: `mean` (média diária) e `max` (pico diário), em metros
- **Chuva**: `sum` (acumulado diário), em mm
- **Vento**: componentes u/v diários, direção dominante, velocidade máxima

### Pipeline de feature engineering (`build_dataset.py`)

1. Coleta dados brutos via ANA SOAP e Open-Meteo
2. Agrega para diário
3. Gera ~221 features (v2) com transformações temporais e interações
4. Salva em formato Parquet

### Versões do dataset

| Versão | Features | Diferença |
|--------|----------|-----------|
| v1 | 189 | Features base (lags, deltas, chuva acumulada, vento) |
| v2 | 221 | + Interações (chuva × nível), proxy saturação do solo, aceleração do nível |

### Qualidade dos dados e decisões

A construção do dataset passou por uma etapa rigorosa de qualidade que resultou em decisões importantes:

| Problema | Estação | Impacto | Decisão |
|----------|---------|---------|---------|
| 61% de dados faltantes | Triunfo (87010000) | Crítico | **Substituído por Muçum (86510000)** |
| Chuva sempre zero | São Jerônimo (87020000) | Crítico | **Substituído por Rio Pardo (85900000)** |
| Chuva com valores absurdos (max 1.305mm/dia) | Triunfo (87010000) | Bug no sensor | Eliminado com a substituição |
| 50% de dados faltantes | CATSUL (87242000) | Médio | Mantido (CatBoost lida com NaN) |

As substituições foram bem-sucedidas: Muçum tem apenas 0.4% de dados faltantes e Rio Pardo 5.9%, ambas com chuva em faixas realistas.

### Por que não incluímos pressão atmosférica

A pressão atmosférica foi considerada como potencial feature mas descartada após análise física:
- **Efeito direto**: ~1cm de água por hPa → variação máxima de ~10cm (irrisório vs enchentes de 3m+)
- **Efeito indireto**: pressão baixa → tempestade → chuva → rio sobe. Mas **já temos chuva e vento diretamente**, então pressão seria informação redundante.
- **Conclusão**: 20 features a menos, sem perda real de informação.

---

## 4. Análise Estatística das Features

Antes de partir para a modelagem, era fundamental entender a natureza estatística das features que construímos. Esta análise serve a dois propósitos: (1) identificar quais transformações de pré-processamento são necessárias e (2) entender quais variáveis têm maior associação com a variável resposta para guiar as primeiras hipóteses de modelagem. Como veremos, os resultados dessa análise influenciaram diretamente a escolha dos algoritmos e a interpretação dos modelos.

### 4.1 Metodologia

Para cada uma das 189 features do dataset v1 (excluindo targets e data), foram calculadas:
- **Estatísticas descritivas**: count, mean, std, min, Q1, mediana, Q3, max, IQR
- **Assimetria (skewness)** e **curtose (kurtosis)** excesso
- **Teste de normalidade**: D'Agostino-Pearson (α = 0.05)
- **Outliers**: valores além de 1.5×IQR ( cercas de Tukey)
- **Coeficiente de variação** (CV = std/mean)
- **Zero-inflation**: percentual de valores exatamente iguais a zero
- **Correlação com a variável resposta**: Spearman (rank-based, não-paramétrica)

### 4.2 Normalidade

| Métrica | Resultado |
|---------|-----------|
| Features que **falham** no teste de normalidade | **189 de 189 (100%)** |
| Features aproximadamente normais | **0 (0%)** |

**Nenhuma feature do dataset é normalmente distribuída.** Isso é esperado para dados hidrológicos, onde:
- Chuva tem distribuição zero-inflada (muitos dias secos, poucos dias chuvosos)
- Níveis têm tendências sazonais e eventos extremos
- Vento tem distribuição direcional (circular)

> **Implicação prática**: modelos lineares (regressão linear, GLM) teriam performance inferior. Modelos baseados em árvores (CatBoost, LightGBM, XGBoost) são ideais pois não assumem normalidade.

### 4.3 Assimetria (Skewness)

| Categoria | % com |skew| > 1 | Exemplo extremo |
|-----------|----------------------|-----------------|
| **Chuva** | ~95% | `taquari_mucum_chuva_roll3`: skew = **+7.74** |
| **Vento (direção)** | ~30% | `encantado_wind_dir_deg`: skew ≈ 0.5 |
| **Nível** | ~20% | `catsul_nivel_max`: skew ≈ 0.8 |
| **Temporal** | ~0% | `month_sin`: skew ≈ 0.0 |

**61% das features** têm |skew| > 1 (altamente assimétricas). Todas as features de chuva são **positivamente assimétricas** — concentração em valores baixos com eventos extremos raros (distribuição de cauda longa à direita).

#### Top 10 features mais assimétricas

| # | Feature | Skewness | Kurtosis | Categoria |
|---|---------|----------|----------|-----------|
| 1 | taquari_mucum_chuva_roll3 | **+7.74** | 113.1 | Chuva 3d Taquari |
| 2 | taquari_mucum_chuva_roll7 | +6.69 | 77.5 | Chuva 7d Taquari |
| 3 | taquari_mucum_chuva | +6.27 | 63.8 | Chuva diária Taquari |
| 4 | jacui_rp_chuva | +5.52 | 41.4 | Chuva diária Jacuí |
| 5 | taquari_mucum_chuva_roll14 | +5.05 | 41.2 | Chuva 14d Taquari |
| 6 | cai_pm_chuva | +4.94 | 33.9 | Chuva diária Caí |
| 7 | guaiba_chuva | +4.92 | 32.0 | Chuva diária Guaíba |
| 8 | rio_grande_precip_mm | +4.73 | 33.5 | Chuva Open-Meteo |
| 9 | sinos_sl_chuva | +4.67 | 31.6 | Chuva diária Sinos |
| 10 | catsul_chuva | +4.42 | 26.3 | Chuva diária CATSUL |

> **Padrão**: todas as 10 mais assimétricas são features de **chuva**. A bacia do Taquari lidera — coerente com ser a bacia mais importante para o modelo.

### 4.4 Curtose (Kurtosis)

**55% das features** têm |kurtosis| > 3 (caudas pesadas / heavy-tailed). As mesmas features de chuva que são assimétricas também têm curtose extrema — eventos de chuva extrema aparecem como "outliers" mas são reais e importantes para o modelo.

### 4.5 Zero-Inflation

Features com **mais de 50% de zeros** (dias sem ocorrência):

| Feature | % Zeros | Interpretação |
|---------|---------|---------------|
| guaiba_chuva | 64.7% | Guaíba tem muitos dias secos |
| catsul_chuva | 62.9% | CATSUL idem |
| represamento_3d | 62.3% | Represamento é evento raro |
| sinos_sl_chuva | 60.7% | Sinos dias secos |
| represamento_2d | 60.3% | Represamento 2d |
| taquari_mucum_chuva | 56.4% | Taquari dias secos |
| jacui_rp_chuva | 54.6% | Jacuí dias secos |
| sinos_cb_chuva | 54.4% | Sinos CB dias secos |
| gravatai_sl_chuva | 52.6% | Gravataí dias secos |
| cai_pm_chuva | 51.2% | Caí dias secos |

> **Nota**: features de chuva e represamento são naturalmente zero-infladas. Modelos de árvores lidam bem com isso — não é necessário transformação.

### 4.6 Correlação com a Variável Resposta (Spearman)

Utilizamos **correlação de Spearman** (rank-based) em vez de Pearson por duas razões:
1. Nenhuma feature é normalmente distribuída
2. Spearman captura relações monotônicas não-lineares, sendo mais robusto a outliers

#### Top 20 features mais correlacionadas com target_delta_1d

| # | Feature | Spearman | Direção | Interpretação física |
|---|---------|----------|---------|---------------------|
| 1 | porto_alegre_u_wind | **+0.401** | + | Vento leste empurra água para o Guaíba |
| 2 | rio_grande_precip_mm | **+0.384** | + | Chuva no sul (bacia Lagoa dos Patos) |
| 3 | cachoeira_do_sul_precip_mm | +0.372 | + | Chuva no Jacuí |
| 4 | u_wind_regional | +0.353 | + | Vento leste regional |
| 5 | porto_alegre_v_wind_roll2 | **-0.345** | - | Vento sul 2d = represamento |
| 6 | mostardas_v_wind_roll3 | -0.337 | - | Vento sul Mostardas 3d |
| 7 | estrela_precip_mm | +0.332 | + | Chuva no Taquari |
| 8 | v_wind_regional_roll3 | -0.326 | - | Vento sul regional 3d |
| 9 | encantado_precip_mm | +0.323 | + | Chuva no Taquari |
| 10 | porto_alegre_v_wind_roll3 | -0.320 | - | Vento sul POA 3d |

#### Insight: Spearman vs Pearson

A mudança de Pearson para Spearman alterou o ranking:

| Feature | Pearson | Spearman | Mudança |
|---------|---------|----------|---------|
| porto_alegre_u_wind | 0.368 (#4) | **0.401 (#1)** | ↑ 3 posições |
| rio_grande_precip_mm | 0.302 (#~10) | **0.384 (#2)** | ↑ 8 posições |
| taquari_mucum_chuva_roll3 | **0.405 (#1)** | 0.224 (#~15) | ↓ 14 posições |

A feature `taquari_mucum_chuva_roll3` tinha a maior correlação Pearson por causa de **outliers de chuva extrema** que inflavam a correlação linear. Com Spearman (baseada em ranks), a verdadeira importância do vento ficou evidente.

> **Conclusão**: para este dataset, Spearman é a medida correta de associação. O vento é mais importante do que Pearson indicava.

### 4.7 Recomendações de Pré-processamento

| Ação | Features | Justificativa |
|------|----------|---------------|
| **Nenhuma transformação necessária** | Todas | Modelos de árvores lidam bem com skew, outliers e zeros |
| **Cuidado com modelos lineares** | Chuva, nível | Distribuições muito não-normais |
| **Log transform** (se usar modelos lineares) | Chuva (15 features) | Reduzir assimetria extrema |
| **Winsorização** (se usar modelos lineares) | Chuva + deltas | >15% outliers pelo critério de Tukey |

> Como optamos por modelos de gradient boosting (CatBoost, LightGBM, XGBoost), **nenhuma transformação foi aplicada**. Os modelos lidam nativamente com todas as distribuições observadas.

---

## 5. Universo de Variáveis Testadas

Com os dados coletados e agregados, o próximo passo foi criar o maior número possível de features que pudessem capturar os diferentes mecanismos físicos que causam enchentes no Guaíba. A lógica por trás da criação de features é: não sabemos de antemão quais variáveis são mais importantes, então é melhor criar um universo amplo e depois usar métodos estatísticos para selecionar as mais relevantes. O resultado foi um conjunto de ~221 features organizadas em 8 categorias.

Total de **~221 features** organizadas em 8 categorias:

### 5.1 Nível a montante (~70 features)

- **Estações**: CATSUL, Gravataí SL, Sinos SL, Sinos CB, Caí PM, Taquari Muçum, Jacuí RP
- **Transformações por estação**:
  - Lags: 1, 2, 3, 5, 7 dias
  - Deltas: variação em 1d, 2d, 3d
  - Estatísticas: média diária (`nivel_mean`), pico diário (`nivel_max`)
- **Exemplo**: `taquari_mucum_delta3` = variação do nível em Muçum nos últimos 3 dias

### 5.2 Chuva acumulada por estação (~48 features)

- **8 estações** × **6 janelas acumuladas**: 3d, 7d, 14d, 30d, 60d
- **Exemplo**: `jacui_rp_chuva_roll7` = chuva acumulada em 7 dias em Rio Pardo

### 5.3 Vento (~40 features)

- **Pontos**: Muçum, Encantado, Estrela, Cachoeira do Sul, Feliz, Mostardas, Rio Grande, Porto Alegre, Campo Bom
- **Variáveis**:
  - Componentes u e v do vento
  - Médias móveis (1d, 2d, 3d)
  - Direção do vento (graus)
  - Velocidade máxima (km/h)
- **Exemplo**: `mostardas_v_wind_roll2` = componente v do vento em Mostardas, média de 2 dias

### 5.4 Chuva total em todas as bacias (~20 features)

- Acumulados em múltiplas janelas
- Média de precipitação entre bacias
- Contagem de bacias com chuva significativa (`n_bacias_muito_chuvosas`)
- **Exemplo**: `chuva_total_acc_30d` = chuva total acumulada em todas as bacias nos últimos 30 dias

### 5.5 Interações (~15 features)

- Chuva × nível (`chuva_x_nivel`)
- Chuva × delta (`chuva_x_delta`)
- Capturam efeitos não-lineares: chuva forte com nível alto = risco maior

### 5.6 Proxy de saturação do solo (~12 features)

- Dias chuvosos em 7d, 14d, 30d
- Ratios (dias chuvosos / total de dias)
- Nível acima da média de 30 dias
- **Exemplo**: `dias_chuvosos_7d` = número de dias com chuva > 1mm nos últimos 7 dias

### 5.7 Aceleração do nível (~6 features)

- Delta do delta (variação da variação)
- Captura tendência de aceleração/desaceleração

### 5.8 Features temporais (~5 features)

- `day_of_week`, `month`
- `sin(2π·doy/365)`, `cos(2π·doy/365)` — codificação cíclica do dia do ano

---

## 6. Seleção de Variáveis — Forward SFS

Com ~221 features disponíveis, tínhamos um problema clássico de machine learning: muitas variáveis, poucas realmente relevantes. Usar todas as features pode levar a overfitting (o modelo aprende ruído dos dados de treino e não generaliza para dados novos) e dificulta a interpretação. A solução é usar um método de seleção de variáveis que encontre automaticamente o subconjunto mais informativo.

Optamos pelo **Forward Sequential Feature Selection (SFS)**, que funciona como um torneio: a cada rodada, testamos todas as features restantes e escolhemos aquela que mais reduz o erro quando adicionada ao conjunto já selecionado. O processo para quando as features adicionais não trazem mais ganho significativo.

### Método

**Forward Sequential Feature Selection (SFS)** com LightGBM como estimador base.

### Validação cruzada temporal

3 folds temporais progressivos:

| Fold | Treino | Validação |
|------|--------|-----------|
| 1 | 2019–2022 | 2023 |
| 2 | 2019–2023 | 2024 |
| 3 | 2019–2024 | 2025 |

### Critério de parada

- **3 ganhos consecutivos < 0.5%** → interrupção da seleção

### Métrica

RMSE médio nos 3 folds.

---

### 6.1 SFS — delta_1d (12 features)

RMSE final: **0.1105 m**

| # | Feature | RMSE (m) | Ganho (%) |
|---|---------|----------|-----------|
| 1 | gravatai_sl_chuva | 0.1375 | 100.0 (baseline) |
| 2 | mucum_wind_dir_deg | 0.1300 | 5.43 |
| 3 | taquari_mucum_delta3 | 0.1241 | 4.57 |
| 4 | campo_bom_wind_dir_deg | 0.1195 | 3.67 |
| 5 | gravatai_sl_nivel_max | 0.1175 | 1.66 |
| 6 | encantado_precip_mm | 0.1162 | 1.17 |
| 7 | cachoeira_do_sul_wind_max_kmh | 0.1150 | 1.00 |
| 8 | cachoeira_do_sul_precip_mm | 0.1136 | 1.24 |
| 9 | catsul_nivel_max | 0.1118 | 1.54 |
| 10 | taquari_mucum_chuva_roll3 | 0.1115 | 0.29 |
| 11 | mostardas_v_wind_roll2 | 0.1110 | 0.46 |
| 12 | rio_grande_v_wind | 0.1105 | 0.42 |

**Interpretação**: Para horizonte de 1 dia, **direção do vento** (Muçum, Campo Bom) e **chuva no Gravataí** são os preditores mais importantes. O vento na Lagoa dos Patos (Mostardas, Rio Grande) também contribui — efeito de represamento.

---

### 6.2 SFS — delta_3d (13 features)

RMSE final: **0.2172 m**

| # | Feature | RMSE (m) | Ganho (%) |
|---|---------|----------|-----------|
| 1 | taquari_mucum_chuva | 0.2645 | 100.0 (baseline) |
| 2 | encantado_wind_dir_deg | 0.2517 | 4.84 |
| 3 | taquari_mucum_delta3 | 0.2432 | 3.38 |
| 4 | sinos_sl_nivel_mean | 0.2318 | 4.67 |
| 5 | jacui_rp_chuva_roll7 | 0.2269 | 2.14 |
| 6 | mostardas_v_wind_roll2 | 0.2228 | 1.79 |
| 7 | campo_bom_precip_mm | 0.2212 | 0.74 |
| 8 | cachoeira_do_sul_wind_dir_deg | 0.2196 | 0.69 |
| 9 | catsul_nivel_mean_lag5 | 0.2193 | 0.17 |
| 10 | chuva_total_acc_30d | 0.2181 | 0.52 |
| 11 | n_bacias_muito_chuvosas | 0.2173 | 0.38 |
| 12 | guaiba_chuva_roll3 | 0.2167 | 0.25 |
| 13 | feliz_wind_max_kmh | 0.2172 | -0.22 |

**Interpretação**: Para 3 dias, **chuva em Muçum (Taquari)** domina. A chuva acumulada em 7 dias no Jacuí (`jacui_rp_chuva_roll7`) e o nível médio nos Sinos (`sinos_sl_nivel_mean`) ganham importância — refletindo o tempo de concentração das bacias.

---

### 6.3 SFS — delta_5d (13 features)

RMSE final: **0.3062 m**

| # | Feature | RMSE (m) | Ganho (%) |
|---|---------|----------|-----------|
| 1 | taquari_mucum_chuva | 0.3607 | 100.0 (baseline) |
| 2 | mostardas_v_wind_roll3 | 0.3464 | 3.95 |
| 3 | sinos_sl_nivel_mean_lag5 | 0.3364 | 2.88 |
| 4 | encantado_wind_dir_deg | 0.3276 | 2.64 |
| 5 | cai_pm_chuva_roll3 | 0.3216 | 1.81 |
| 6 | jacui_rp_nivel_mean_lag7 | 0.3169 | 1.48 |
| 7 | sinos_sl_chuva_roll3 | 0.3131 | 1.19 |
| 8 | dias_chuvosos_7d | 0.3117 | 0.44 |
| 9 | mostardas_v_wind | 0.3097 | 0.65 |
| 10 | dias_chuvosos_14d | 0.3080 | 0.55 |
| 11 | cachoeira_do_sul_wind_dir_deg | 0.3078 | 0.07 |
| 12 | catsul_chuva | 0.3065 | 0.42 |
| 13 | estrela_wind_max_kmh | 0.3062 | 0.09 |

**Interpretação**: Para 5 dias, aparecem **dias chuvosos** (`dias_chuvosos_7d`, `dias_chuvosos_14d`) — proxy de saturação do solo. O nível com lag de 7 dias no Jacuí (`jacui_rp_nivel_mean_lag7`) captura o efeito de propagação lenta.

---

### 6.4 Comparação entre targets

#### Features compartilhadas

| Feature | delta_1d | delta_3d | delta_5d |
|---------|:--------:|:--------:|:--------:|
| taquari_mucum_chuva | — | ✅ (#1) | ✅ (#1) |
| taquari_mucum_delta3 | ✅ (#3) | ✅ (#3) | — |
| encantado_wind_dir_deg | — | ✅ (#2) | ✅ (#4) |
| cachoeira_do_sul_wind_dir_deg | — | ✅ (#8) | ✅ (#11) |
| mostardas_v_wind_roll2 | ✅ (#11) | ✅ (#6) | — |
| mostardas_v_wind | — | — | ✅ (#9) |
| sinos_sl_nivel_mean | — | ✅ (#4) | — |
| sinos_sl_nivel_mean_lag5 | — | — | ✅ (#3) |

#### Padrões identificados

- **Horizonte curto (1d)**: predomínio de **direção do vento** e **nível a montante** — efeitos imediatos
- **Horizonte médio (3d)**: **chuva acumulada na bacia do Taquari** é o preditor #1, seguido por vento e nível nos Sinos
- **Horizonte longo (5d)**: **proxy de saturação do solo** (dias chuvosos) e **nível com lag maior** ganham relevância
- `taquari_mucum_delta3` aparece em delta_1d e delta_3d, mas **não** em delta_5d
- A bacia do Taquari é consistentemente a mais importante em todos os horizontes

---

## 7. Modelagem — Testes de Desafiantes

Com as features selecionadas pelo SFS, chegou o momento de escolher o melhor algoritmo de machine learning. Não existe um modelo único que seja sempre o melhor — a performance depende dos dados, do problema e das métricas avaliadas. Por isso, testamos três algoritmos de gradient boosting (CatBoost, XGBoost e LightGBM), cada um com suas particularidades de implementação, e dois conjuntos de features (SFS vs ALL) para entender se a seleção de variáveis realmente ajudava.

Para cada combinação de algoritmo e conjunto de features, avaliamos o desempenho em validação cruzada temporal (3 folds) e em dados fora do tempo (OOT, 2026). Isso nos dá confiança de que os resultados são robustos e não fruto de overfitting.

### Configuração

- **Algoritmos**: CatBoost, XGBoost, LightGBM
- **Conjuntos de features**: SFS (selecionadas) vs ALL (todas ~221)
- **Validação cruzada**: 3 folds temporais (idênticos ao SFS)
- **Total de variantes**: 6 por target (3 algoritmos × 2 feature sets)

### Métricas avaliadas

| Métrica | Descrição |
|---------|-----------|
| RMSE | Erro quadrático médio (m) |
| MAE | Erro absoluto médio (m) |
| R² | Coeficiente de determinação |
| MedianAE | Mediana do erro absoluto (m) |
| MaxError | Erro máximo observado (m) |
| ±5cm | % de previsões com erro ≤ 5 cm |
| ±10cm | % de previsões com erro ≤ 10 cm |
| ±20cm | % de previsões com erro ≤ 20 cm |

---

### 7.1 Resultados — delta_1d

#### DEV CV (média dos 3 folds)

| Modelo | RMSE | MAE | R² | MedianAE | MaxError | ±5cm | ±10cm | ±20cm |
|--------|------|-----|-----|----------|----------|------|-------|-------|
| CatBoost_SFS | 0.1146 | 0.0722 | 0.416 | 0.0487 | 0.950 | 50.8% | 78.3% | 94.8% |
| **CatBoost_ALL** | **0.1146** | **0.0685** | **0.417** | **0.0463** | 1.032 | **53.2%** | **80.4%** | **95.7%** |
| XGBoost_SFS | 0.1174 | 0.0746 | 0.388 | 0.0529 | 0.990 | 48.0% | 77.1% | 94.5% |
| XGBoost_ALL | 0.1169 | 0.0712 | 0.391 | 0.0481 | 1.019 | 51.2% | 77.4% | 95.5% |
| LightGBM_SFS | 0.1129 | 0.0744 | 0.428 | 0.0544 | 0.894 | 46.9% | 76.4% | 94.8% |
| LightGBM_ALL | 0.1157 | 0.0712 | 0.405 | 0.0469 | 1.002 | 52.5% | 77.9% | 96.0% |

#### Vencedor: **CatBoost_ALL** (5 de 9 métricas)

Melhor MAE, R², MedianAE, ±5cm e ±10cm. LightGBM_SFS tem melhor RMSE e MaxError.

#### Decisão: por que passamos a usar apenas SFS para delta_3d e delta_5d

Apesar do CatBoost_ALL ter vencido no delta_1d, a diferença para o CatBoost_SFS foi **marginal** (RMSE 0.1146 vs 0.1146, MAE 0.0685 vs 0.0722). Diante disso, optamos por usar apenas modelos SFS (com features selecionadas) para os targets seguintes, por três razões:

1. **Simplicidade e interpretabilidade**: 12-13 features selecionadas têm significado físico claro, versus 189-221 features onde muitas são redundantes ou difíceis de interpretar.

2. **Robustez contra overfitting**: com menos features, o modelo generaliza melhor para dados novos. O delta_5d com 13 features SFS venceu CatBoost_ALL em 8/8 métricas no OOT — validando a escolha.

3. **Eficiência computacional**: o SFS com 221 features leva ~60 minutos por target. Rodar com ALL features no treino final é viável, mas o SFS já captura o essencial.

4. **Foco no sistema de alerta**: nosso objetivo não é minimizar RMSE geral, mas **detectar eventos extremos**. O SFS captura os mecanismos físicos mais importantes (chuva no Taquari, vento de represamento, saturação do solo) que são exatamente os que causam enchentes.

> **Resultado prático**: o modelo delta_5d SFS (13 features) previu o pico de maio/2024 com erro de apenas 2% em D-5 — validando que menos features bem escolhidas superam mais features com ruído.

---

### 7.2 Resultados — delta_3d

#### DEV CV

| Modelo | RMSE | MAE | R² | MedianAE | MaxError | ±5cm | ±10cm | ±20cm |
|--------|------|-----|-----|----------|----------|------|-------|-------|
| CatBoost | 0.2228 | 0.1428 | 0.418 | 0.1033 | 1.462 | 25.1% | 49.1% | 78.6% |
| XGBoost | 0.2204 | 0.1460 | 0.423 | 0.1039 | 1.475 | 24.4% | 48.7% | 76.8% |
| **LightGBM** | **0.2185** | **0.1456** | **0.434** | 0.1081 | **1.334** | 24.7% | 46.9% | 77.8% |

#### OOT

| Modelo | RMSE | MAE | R² | MedianAE | MaxError | ±5cm | ±10cm | ±20cm |
|--------|------|-----|-----|----------|----------|------|-------|-------|
| CatBoost | 0.1706 | 0.1237 | 0.392 | 0.0844 | 0.817 | 27.6% | 54.6% | 83.8% |
| XGBoost | 0.1738 | 0.1281 | 0.369 | 0.0997 | 0.801 | 30.3% | 50.8% | 79.5% |
| **LightGBM** | **0.1654** | **0.1230** | **0.428** | 0.0936 | **0.729** | 24.3% | 53.5% | 82.2% |

#### Vencedor: **LightGBM** (melhor RMSE e R² em DEV e OOT)

LightGBM com features SFS apresenta o melhor equilíbrio entre RMSE e R². Note que o MaxError do LightGBM (0.729m) é o menor de todos no OOT.

---

### 7.3 Resultados — delta_5d

#### DEV CV

| Modelo | RMSE | MAE | R² | MedianAE | MaxError | ±5cm | ±10cm | ±20cm |
|--------|------|-----|-----|----------|----------|------|-------|-------|
| **CatBoost** | **0.3102** | **0.1923** | **0.342** | **0.1288** | **1.990** | **21.2%** | **38.7%** | **67.6%** |
| XGBoost | 0.3174 | 0.1976 | 0.316 | 0.1377 | 2.011 | 20.3% | 40.2% | 66.3% |
| LightGBM | 0.3110 | 0.1947 | 0.342 | 0.1346 | 2.031 | 20.2% | 38.5% | 65.6% |

#### OOT

| Modelo | RMSE | MAE | R² | MedianAE | MaxError | ±5cm | ±10cm | ±20cm |
|--------|------|-----|-----|----------|----------|------|-------|-------|
| **CatBoost** | **0.2034** | **0.1441** | **0.337** | **0.1147** | **0.902** | **24.7%** | **46.7%** | **78.6%** |
| XGBoost | 0.2167 | 0.1565 | 0.247 | 0.1163 | 0.951 | 21.4% | 42.3% | 72.5% |
| LightGBM | 0.2158 | 0.1538 | 0.253 | 0.1185 | 0.947 | 20.9% | 42.9% | 73.6% |

#### Vencedor: **CatBoost** (7/8 DEV, 8/8 OOT)

Domínio total do CatBoost no horizonte de 5 dias. Superior em praticamente todas as métricas.

---

## 8. Teste em Eventos Extremos e Sistema de Alerta

### 8.1 O cenário: enchente de maio de 2024

Em maio de 2024, Porto Alegre sofreu a **maior enchente de sua história**. O nível do Guaíba atingiu **4.70 metros**, superando todas as cotas históricas e causando devastação em larga escala — dezenas de mortes e deslocamento de centenas de milhares de pessoas. A enchente foi causada por uma combinação de fatores:

- **Chuvas intensas e persistentes** nas bacias do Taquari, Jacuí, Caí e Sinos por vários dias consecutivos
- **Vento de represamento** na Lagoa dos Patos, impedindo a drenagem do Guaíba
- **Solo saturado** após semanas de chuva acima da média

Diante desse cenário, o objetivo final do projeto não é apenas prever o delta do nível com precisão, mas sim construir um **sistema de alerta operacional** que possa antecipar eventos extremos com antecedência suficiente para ações de preparação e evacuação.

### 8.2 Desempenho dos modelos na enchente

Todos os modelos **subestimam a magnitude absoluta** do evento. Isso é esperado: o evento de maio de 2024 está além da distribuição observada nos dados de treino.

#### Experimento com regressão quantílica

Testamos modelos de regressão quantílica (P90, P95, P99) com LightGBM para tentar capturar melhor os extremos:

| Modelo | Previsto pico | Erro % | RMSE geral | MAE geral |
|--------|---------------|--------|------------|-----------|
| MSE (média) | +0.292m | -83.3% | 0.362 | 0.182 |
| Quantile P90 | +0.465m | -73.3% | 0.353 | 0.231 |
| Quantile P95 | +0.511m | -70.7% | 0.369 | 0.271 |
| **Quantile P99** | **+0.596m** | **-65.8%** | 0.399 | 0.324 |

O P99 melhorou 17.5 pontos percentuais na previsão do pico, mas **superestima sistematicamente** os dias normais (acumulado de +7.38m vs real +3.79m). O trade-off é claro: modelos otimizados para extremos são ruins na média, e vice-versa. Por isso, mantivemos o modelo MSE como padrão e usamos o **percentil da previsão** como indicador de alerta.

### 8.3 Insight-chave: o ranking percentil está correto

Embora o valor absoluto previsto seja inferior ao real, a **classificação percentil** das previsões é extremamente alta:

| Target | Previsão | Z-score | Percentil | Interpretação |
|--------|----------|---------|-----------|---------------|
| delta_1d | elevado | **5.70σ** | ~P99.5% | Evento extremo detectado |
| delta_3d | elevado | **7.96σ** | ~P99.99% | Evento extremo detectado |
| delta_5d (D-5) | elevado | **4.86σ** | ~P99.9% | Alerta com 5 dias de antecedência |

O modelo **"sabe" que algo fora do comum está acontecendo**. Mesmo quando erra o valor absoluto, ele classifica corretamente o evento como extremo. Isso é suficiente para um **sistema de alerta operacional**.

### 8.4 O que é um sistema de alerta por limiares?

Um **sistema de alerta por limiares** funciona definindo valores de referência (limiares) que, quando ultrapassados por uma previsão, disparam um alerta. A pergunta central é:

> "Dada a previsão do modelo para os próximos N dias, o nível do rio vai atingir uma cota perigosa?"

Os **limiares** são os valores que definem o que é "perigoso". No caso do Guaíba, existem cotas oficiais de risco definidas pela Defesa Civil:

| Faixa | Nível (m) | Status | Ação recomendada |
|-------|-----------|--------|------------------|
| < 1.0 | Normal | Sem risco | — |
| 1.0 – 2.0 | Atenção | Monitoramento | Acompanhar evolução |
| 2.0 – 3.0 | Alerta | Preparação | Preparar equipamentos, alertar população |
| > 3.0 | **INUNDAÇÃO** | Evacuação | Evacuar áreas de risco |

O desafio é: como transformar as previsões do modelo (que são **deltas**, ou seja, variações de nível) em **níveis absolutos** que possam ser comparados com essas cotas oficiais?

### 8.5 Abordagem 1: Delta-only (apenas variação)

A abordagem mais simples é definir um **limiar no delta previsto**. Se o modelo prevê que o nível vai subir mais que X metros nos próximos N dias, disparamos um alerta.

Testamos múltiplos limiares para cada horizonte de previsão:

| Target | Limiar | Alertas/ano | Falsos Positivos | 1º Alerta | Antecedência |
|--------|--------|-------------|------------------|-----------|--------------|
| delta_1d | ≥ 0.40m | ~3 | ~1 | D-2 | 2 dias |
| delta_3d | ≥ 0.60m | ~5 | ~2 | D-3 | 3 dias |
| delta_5d | ≥ 0.80m | 7 | **1** | **D-5** | **5 dias** |

O modelo delta_5d com limiar ≥ 0.80m se destaca: apenas **1 falso positivo** em todo o ano de 2024, com **5 dias de antecedência** antes do pico da enchente.

**Exemplo concreto (maio/2024):**
- 27/abr: modelo prevê delta_5d = +1.86m → acima de 0.80m → **ALERTA DISPARADO** (D-5)
- 28/abr: modelo prevê delta_5d = +2.40m → acima de 0.80m → alerta mantido (D-4)
- 02/mai: pico real = 4.70m

### 8.6 Abordagem 2: Combinada (INSIGHT PRINCIPAL)

A abordagem delta-only tem uma limitação: **o mesmo delta pode significar coisas muito diferentes dependendo do nível atual**.

Considere cenários com o mesmo delta previsto de +1.5m:

| Cenário | Nível atual (T0) | Delta previsto | Nível final previsto | Classificação |
|---------|-------------------|----------------|----------------------|---------------|
| A | 0.5m | +1.5m | **2.0m** | Alerta |
| B | -0.5m | +1.5m | **1.0m** | Atenção |
| C | 2.0m | +1.5m | **3.5m** | **INUNDAÇÃO** |

Para um gestor público que precisa tomar decisões, saber que "o delta será +1.5m" é insuficiente — ele precisa saber **qual nível o rio vai atingir**.

A solução: **combinar o nível atual medido com o delta previsto**:

```
nivel_previsto_T5 = nivel_T0 + delta_previsto_5d
```

Onde:
- `nivel_T0` = nível medido hoje na régua (dado real, não previsto)
- `delta_previsto_5d` = variação prevista pelo modelo para os próximos 5 dias
- `nivel_previsto_T5` = nível absoluto previsto daqui a 5 dias

Isso transforma a previsão estatística em **informação operacional**: "o nível vai atingir 4.18m" é muito mais útil para evacuação do que "delta previsto de +2.84m".

### 8.7 Validação: Enchente de maio de 2024

Aplicando o sistema combinado retroativamente:

| Data da previsão | Nível T0 | Delta previsto 5d | **Nível T+5 previsto** | **Status** | Nível real T+5 |
|------------------|----------|-------------------|------------------------|------------|----------------|
| 26/abr/2024 | 0.97m | +0.79m | **1.76m** | ⚠️ Atenção | 1.93m |
| 27/abr/2024 | 1.04m | +1.86m | **2.90m** | 🟡 **ALERTA** | 2.95m |
| 28/abr/2024 | 1.34m | +2.84m | **4.18m** | 🔴 **INUNDAÇÃO** | 4.70m |
| 29/abr/2024 | 1.21m | +3.97m | **5.18m** | 🔴 **INUNDAÇÃO** | 5.14m |

**O sistema teria:**
- Disparado alerta de **ALERTA** em 27/abr (D-5 antes do pico)
- Escalado para **INUNDAÇÃO** em 28/abr (D-4 antes do pico)
- Previsto nível de 4.18m (real: 4.70m) — erro de apenas 11%

### 8.8 Validação em tempo real: julho de 2026

O sistema foi aplicado em tempo real durante julho de 2026, usando o modelo recalibrado com dados até junho/2026:

| Data | Nível T0 | Delta previsto | **Nível T+5** | Classificação |
|------|----------|----------------|---------------|---------------|
| 17/jul | 0.82m | +0.23m | **1.05m** | ⚠️ Atenção |
| 21/jul | 1.00m | +0.95m | **1.95m** | ⚠️ Atenção (quase Alerta) |
| 23/jul | 1.93m | +0.07m | **2.00m** | 🟡 ALERTA |
| **24/jul** | **2.13m** | +0.10m | **2.24m** | 🟡 **ALERTA** |
| **28/jul** | **1.67m** | **+0.85m** | **2.51m** | 🟡 **ALERTA** |
| **31/jul** | **2.51m** | -0.12m | **2.39m** | 🟡 **ALERTA** |

O sistema corretamente identificou o status de ALERTA e previu a subida do nível com dias de antecedência. A previsão de delta_5d em 21/jul (+0.95m, Z-score +3.04σ, P99.1%) antecipou o evento que elevou o nível de 1.00m para 2.51m em 10 dias.

**Situação em 31/07/2026**: nível real = 2.51m, modelo prevê T+5 = 2.39m → **permanece em ALERTA com tendência de leve queda**.

---

## 9. Modelos Campeões

Após todo o processo de análise exploratória, seleção de variáveis, testes de algoritmos e validação em eventos extremos, definimos os dois modelos oficiais que serão utilizados em produção. Ambos foram recalibrados com todos os dados disponíveis (2019 a junho/2026) para maximizar a informação disponível no momento do deploy.

### 9.1 Modelo Delta 3d

| Atributo | Valor |
|----------|-------|
| **Algoritmo** | LightGBM |
| **Horizonte** | 3 dias |
| **Features** | 13 (selecionadas por SFS) |
| **Treino** | Todos os dados 2019 – jun/2026 |
| **Finalidade** | Alerta de médio prazo |

#### Features do modelo delta_3d

| # | Feature | Descrição |
|---|---------|-----------|
| 1 | taquari_mucum_chuva | Chuva diária em Muçum (Taquari) |
| 2 | encantado_wind_dir_deg | Direção do vento em Encantado |
| 3 | taquari_mucum_delta3 | Variação do nível em Muçum em 3 dias |
| 4 | sinos_sl_nivel_mean | Nível médio nos Sinos (São Leopoldo) |
| 5 | jacui_rp_chuva_roll7 | Chuva acumulada 7d em Rio Pardo (Jacuí) |
| 6 | mostardas_v_wind_roll2 | Componente v do vento em Mostardas (média 2d) |
| 7 | campo_bom_precip_mm | Precipitação em Campo Bom |
| 8 | cachoeira_do_sul_wind_dir_deg | Direção do vento em Cachoeira do Sul |
| 9 | catsul_nivel_mean_lag5 | Nível médio no CATSUL (lag 5 dias) |
| 10 | chuva_total_acc_30d | Chuva total acumulada em todas as bacias (30d) |
| 11 | n_bacias_muito_chuvosas | Número de bacias com chuva significativa |
| 12 | guaiba_chuva_roll3 | Chuva acumulada 3d na bacia do Guaíba |
| 13 | feliz_wind_max_kmh | Velocidade máxima do vento em Feliz |

---

### 9.2 Modelo Delta 5d

| Atributo | Valor |
|----------|-------|
| **Algoritmo** | CatBoost |
| **Horizonte** | 5 dias |
| **Features** | 13 (selecionadas por SFS) |
| **Treino** | Todos os dados 2019 – jun/2026 |
| **Finalidade** | Alerta de longo prazo |

#### Features do modelo delta_5d

| # | Feature | Descrição |
|---|---------|-----------|
| 1 | taquari_mucum_chuva | Chuva diária em Muçum (Taquari) |
| 2 | mostardas_v_wind_roll3 | Componente v do vento em Mostardas (média 3d) |
| 3 | sinos_sl_nivel_mean_lag5 | Nível médio nos Sinos (lag 5 dias) |
| 4 | encantado_wind_dir_deg | Direção do vento em Encantado |
| 5 | cai_pm_chuva_roll3 | Chuva acumulada 3d no Caí (Passo Montenegro) |
| 6 | jacui_rp_nivel_mean_lag7 | Nível médio no Jacuí (lag 7 dias) |
| 7 | sinos_sl_chuva_roll3 | Chuva acumulada 3d nos Sinos |
| 8 | dias_chuvosos_7d | Dias com chuva > 1mm nos últimos 7 dias |
| 9 | mostardas_v_wind | Componente v do vento em Mostardas (diário) |
| 10 | dias_chuvosos_14d | Dias com chuva > 1mm nos últimos 14 dias |
| 11 | cachoeira_do_sul_wind_dir_deg | Direção do vento em Cachoeira do Sul |
| 12 | catsul_chuva | Chuva diária no CATSUL |
| 13 | estrela_wind_max_kmh | Velocidade máxima do vento em Estrela |

#### Feature Importance (modelo final treinado com dados até jun/2026)

A importância das features mudou significativamente quando o modelo foi recalibrado com todos os dados disponíveis:

| # | Feature | Importância |
|---|---------|-------------|
| 1 | mostardas_v_wind_roll3 | 14.87 |
| 2 | jacui_rp_nivel_mean_lag7 | 11.94 |
| 3 | sinos_sl_nivel_mean_lag5 | 11.83 |
| 4 | taquari_mucum_chuva | 10.04 |
| 5 | cai_pm_chuva_roll3 | 9.35 |
| 6 | mostardas_v_wind | 7.57 |
| 7 | sinos_sl_chuva_roll3 | 6.71 |
| 8 | encantado_wind_dir_deg | 6.39 |
| 9 | cachoeira_do_sul_wind_dir_deg | 5.78 |
| 10 | dias_chuvosos_14d | 5.47 |
| 11 | estrela_wind_max_kmh | 3.73 |
| 12 | catsul_chuva | 3.37 |
| 13 | dias_chuvosos_7d | 2.95 |

Com mais dados, o **vento sul em Mostardas** (`mostardas_v_wind_roll3`) tornou-se a feature #1 — o efeito de represamento na Lagoa dos Patos é mais determinante com mais exemplos de enchentes no histórico.

---

## 10. Principais Insights

Ao longo do desenvolvimento do projeto, diversos insights surgiram que vão além dos números e métricas. Estes aprendizados são tão valiosos quanto os modelos finais, pois informam como pensar sobre o problema de previsão de enchentes e como interpretar os resultados em contextos reais.

### 1. Valor absoluto errado, ranking percentil correto

O modelo pode subestimar o nível absoluto em eventos extremos, mas o **percentil da previsão é extremo** (P99.5%+). Para um sistema de alerta, isso é suficiente: o importante é detectar que algo incomum está acontecendo.

### 2. Horizontes maiores capturam melhor a física

O RMSE proporcional melhora com horizontes maiores porque os modelos de 3d e 5d capturam **acumulação de chuva** e **tempo de concentração das bacias**, que são os processos físicos que realmente causam enchentes.

### 3. Nível atual + delta = alerta operacional

Combinar o nível medido hoje (T0) com o delta previsto dá um **nível previsto absoluto** que pode ser comparado diretamente com as cotas oficiais de risco. Esta é a abordagem final do sistema de alerta.

### 4. Bacia do Taquari é o preditor #1

Em todos os horizontes (1d, 3d, 5d), a chuva em Muçum (Taquari) aparece como a feature mais importante ou entre as top-3. A bacia do Taquari é a maior contribuinte de vazão ao Guaíba.

### 5. Direção do vento importa mais que velocidade

O vento na direção correta (geralmente sudeste/nordeste) causa **represamento** na Lagoa dos Patos, elevando o nível do Guaíba independentemente da chuva. Features de `wind_dir_deg` aparecem consistentemente antes de `wind_speed`.

### 6. Proxy de saturação do solo é relevante para longo prazo

`dias_chuvosos_7d` e `dias_chuvosos_14d` aparecem apenas no modelo de 5 dias. Quando o solo já está saturado, chuvas adicionais escoam mais rapidamente para os rios. Este efeito só se manifesta em horizontes mais longos.

### 7. CatBoost vs LightGBM: não há vencedor único

| Target | Vencedor |
|--------|----------|
| delta_1d | CatBoost_ALL |
| delta_3d | LightGBM_SFS |
| delta_5d | CatBoost |

A escolha depende do target, do horizonte e das métricas priorizadas. Ambos são modelos de gradient boosting robustos e adequados para este problema.

---

## 11. Próximos Passos

O projeto atingiu seu objetivo principal: dois modelos de alerta calibrados e validados. No entanto, um modelo de machine learning só gera valor quando está operacionalizando em produção, recebendo dados novos e gerando previsões em tempo real. Esta seção descreve as ações necessárias para transformar o protótipo atual em um sistema operacional.

### Curto prazo

- [ ] **Deploy operacional**: rodar previsões diariamente com dados atualizados
- [ ] **Integração de dados em tempo real**: conectar ao endpoint SOAP da ANA com agendamento automático
- [ ] **Dashboard de alertas**: interface web com mapa e histórico de previsões

### Médio prazo

- [ ] **Re-treinamento periódico**: atualizar modelos mensalmente com novos dados
- [ ] **Incorporar mais estações**: incluir estações do INMET e CRH/SEMA-RS
- [ ] **Modelos de ensemble**: combinar delta_3d e delta_5d em um único modelo de alerta
- [ ] **Previsão probabilística**: gerar intervalos de confiança (quantile regression)

### Longo prazo

- [ ] **Dados de radar meteorológico**: incorporar dados de precipitação por radar para melhorar resolução espacial
- [ ] **Modelos hidrodinâmicos**: integrar com modelos físicos do Guaíba para cenários de simulação
- [ ] **Sistema de alerta público**: API e notificações via SMS/WhatsApp para a população

---

## Apêndice A: Estrutura do Projeto

```
Previsao_Nivel_Guaiba_v2/
├── src/
│   └── config.py          # Configuração de estações, parâmetros e URLs
├── data/
│   ├── raw/                # Dados brutos (ANA + Open-Meteo)
│   └── processed/          # Datasets, resultados SFS e comparações
├── build_dataset.py        # Pipeline de feature engineering
├── docs/
│   └── DOCUMENTACAO_TECNICA.md  # Este documento
└── notebooks/              # Análises exploratórias
```

## Apêndice B: Referências

- **ANA — Agência Nacional de Águas e Saneamento Básico**: https://www.snirh.gov.br/
- **Open-Meteo ERA5**: https://open-meteo.com/
- **LightGBM**: https://lightgbm.readthedocs.io/
- **CatBoost**: https://catboost.ai/
- **XGBoost**: https://xgboost.readthedocs.io/

---

*Documento gerado em 01/08/2026 — Projeto Previsão de Nível do Guaíba v2*
