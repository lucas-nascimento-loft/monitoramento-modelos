# Monitoramento Blend4

Para atualizar o monitoramento, rode **4 notebooks**, nesta ordem, **na raiz do repositório**.

```
1. 01.Base_Monitoramento_CR.ipynb
2. 01.Base_Monitoramento_Funil.ipynb
3. 02.Monitoramento_Blend4.ipynb
4. 02.Monitoramento_Funil_Blend4.ipynb
```

Os dois primeiros montam as bases. Os dois últimos geram os monitores.

---

## Passo a passo

### 1. Montar a base de CR

Rode `01.Base_Monitoramento_CR.ipynb`.

Ele busca as consultas realizadas no BigQuery, escora o Blend4 e salva a base em:

`data/analytics/df_predict_blend4.csv`

### 2. Montar a base do funil

Rode `01.Base_Monitoramento_Funil.ipynb`.

Ele busca o funil no BigQuery, cruza com a base de CR e salva em:

`data/analytics/df_funil_blend4.csv`

**Importante:** este notebook usa o arquivo do passo 1. Não pule a ordem.

### 3. Rodar o monitor de CR

Rode `02.Monitoramento_Blend4.ipynb`.

Aqui ficam scores, ratings, mix vs Blend3, PSI das variáveis e cortes de rating.

### 4. Rodar o monitor do funil

Rode `02.Monitoramento_Funil_Blend4.ipynb`.

Aqui ficam conversão, mix de modelos, renda, cidades e imobiliárias.

Os passos 3 e 4 podem ser rodados em qualquer ordem entre si. Os dois só precisam das bases prontas.

---

## Onde ver o resultado

Duas formas:

1. **No próprio notebook `02`** — gráficos e tabelas já aparecem nas células.
2. **Na pasta `Monitores/`** — cada notebook `02` gera um HTML no final:

   - `Monitores/02.Monitoramento_Blend4_report.html`
   - `Monitores/02.Monitoramento_Funil_Blend4_report.html`

Antes de gerar o HTML de novo, salve o notebook (Cmd+S / Ctrl+S). Assim o relatório sai com os gráficos atualizados.

---

## Antes de rodar pela primeira vez

1. Tenha Python, Jupyter e acesso ao BigQuery (`loft-dl-fintech`).
2. Instale os pacotes:

```bash
pip install pandas numpy matplotlib seaborn pandas-gbq google-cloud-bigquery google-auth unidecode requests nbformat nbconvert beautifulsoup4
```

3. Faça login no Google:

```bash
gcloud auth application-default login
```

Os notebooks criam sozinhos as pastas em `data/`. Esses arquivos ficam só na sua máquina (não sobem no Git).

---

## Se o notebook `02` reclamar de arquivo faltando

Alguns arquivos de referência já precisam existir em `data/analytics/` (eles não são gerados toda semana):

- `blend4_psi_reference.pkl`
- `blend4_psi_baseline_ref.csv`
- `blend4_bvs_score_psi_reference.pkl`
- `psi_income_rental_reference.pkl`
- `dev_rating_pol_blend4.csv`

Se algum estiver faltando, rode uma vez `01.Monitoramento_Variaveis_PSI_Blend4.ipynb`.
