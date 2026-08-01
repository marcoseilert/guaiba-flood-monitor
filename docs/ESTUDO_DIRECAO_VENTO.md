# 💨 Estudo: Impacto da Direção do Vento na Previsão de Enchentes

## Metodologia

Para cada variável de direção do vento usada nos modelos (`encantado_wind_dir_deg` e `cachoeira_do_sul_wind_dir_deg`), simulamos as 8 direções da rosa dos ventos em **365 dias** (ago/2025 a ago/2026), mantendo todas as outras variáveis constantes. Medimos o impacto médio na previsão de delta_3d e delta_5d.

**Interpretação**: impacto positivo = o nível previsto AUMENTA (pior para enchente), impacto negativo = o nível previsto DIMINUI (melhor).

## Resultados

### Encantado (vale do Taquari)

| Direção | Impacto Δ3d | Impacto Δ5d | Classificação |
|---------|-------------|-------------|---------------|
| ⬆️ Norte | -0.003m | +0.024m | 🟡 Neutro |
| ↗️ Nordeste | -0.016m | +0.006m | 🟢 Melhor |
| ➡️ Leste | -0.034m | -0.016m | 🟢 Melhor |
| ↘️ Sudeste | -0.033m | -0.029m | 🟢 Melhor |
| ⬇️ Sul | -0.042m | -0.029m | 🟢 **Melhor** |
| ↙️ Sudoeste | -0.015m | -0.013m | 🟢 Melhor |
| ⬅️ Oeste | +0.022m | +0.027m | 🔴 Pior |
| ↖️ Noroeste | **+0.071m** | **+0.046m** | 🔴 **Pior** |

**Padrão**: Vento de **sul a sudeste** reduz o nível (favorável). Vento de **noroeste** aumenta significativamente (desfavorável — +7cm em 3 dias em média).

**Interpretação física**: Vento sul em Encantado empurra a água do rio Taquari "para baixo" (rio corre de norte para sul). Vento noroeste traz umidade e chuva das bacias do noroeste.

### Cachoeira do Sul (rio Jacuí)

| Direção | Impacto Δ3d | Impacto Δ5d | Classificação |
|---------|-------------|-------------|---------------|
| ⬆️ Norte | **+0.060m** | **+0.075m** | 🔴 **Pior** |
| ↗️ Nordeste | +0.051m | +0.060m | 🔴 Pior |
| ➡️ Leste | +0.009m | +0.036m | 🔴 Pior |
| ↘️ Sudeste | -0.009m | -0.024m | 🟢 Melhor |
| ⬇️ Sul | -0.013m | -0.029m | 🟢 Melhor |
| ↙️ Sudoeste | -0.014m | -0.027m | 🟢 Melhor |
| ⬅️ Oeste | -0.005m | -0.009m | 🟡 Neutro |
| ↖️ Noroeste | +0.017m | +0.008m | 🔴 Pior |

**Padrão**: Vento de **norte/nordeste** aumenta significativamente o nível (desfavorável — +6-8cm em 3 dias). Vento de **sul/sudoeste** reduz (favorável).

**Interpretação física**: O Jacuí corre de nordeste para sudoeste. Vento norte/nordeste empurra a água "contra" o fluxo do rio, causando represamento. Vento sul/sudoeste "ajuda" a água a escoar.

## Síntese

| Variável | 🟢 Melhor direção | 🔴 Pior direção | Diferença |
|----------|-------------------|-----------------|-----------|
| Encantado | Sul (-0.04m) | Noroeste (+0.06m) | **10cm** |
| Cachoeira do Sul | Sudoeste (-0.02m) | Norte (+0.07m) | **9cm** |

## Aplicação no Dashboard

A tabela de variáveis do dashboard substitui o percentil por um indicador de cor:
- 🟢 Verde: direção favorável (reduz risco de enchente)
- 🟡 Neutro: impacto insignificante
- 🔴 Vermelho: direção desfavorável (aumenta risco de enchente)

O indicador é baseado na média de 365 dias de simulação, não em um único dia.

## Limitações

- O impacto depende das condições hidrológicas do dia (chuva, nível atual)
- Desvios padrão altos (±3-5cm) indicam variabilidade
- O efeito é marginal (~5-7cm) comparado com variáveis de chuva e nível
- A classificação é uma simplificação: em dias secos o impacto é menor, em dias chuvosos pode ser maior
