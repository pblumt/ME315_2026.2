import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from datetime import datetime
from pathlib import Path

#instalar piparrow

# 1. Utilizando o arquivo renda_adulta.csv e sabendo que ele não possui cabeçalho, 
# faça a importação do banco de dados utilizando os nomes das colunas conforme apresentado acima e 
# na sequência ali indicada. No momento da importação do arquivo, você deve, também, 
# indicar os tipos de cada uma das colunas. Utilize o fato de que o símbolo ? representa valores faltantes.

colunas = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

esquema = {
    "age": pl.Int64,
    "workclass": pl.Utf8,
    "fnlwgt": pl.Int64,
    "education": pl.Utf8,
    "education-num": pl.Int64,
    "marital-status": pl.Utf8,
    "occupation": pl.Utf8,
    "relationship": pl.Utf8,
    "race": pl.Utf8,
    "sex": pl.Utf8,
    "capital-gain": pl.Int64,
    "capital-loss": pl.Int64,
    "hours-per-week": pl.Int64,
    "native-country": pl.Utf8,
    "income": pl.Utf8,
}

renda_adulta = pl.read_csv(
    "renda_adulta.csv",
    has_header=False,
    new_columns=colunas,
    schema_overrides=esquema,
    null_values="?",
)

# 2. Apresente os tipos de cada uma das coluna.
print("--- Tipos das Colunas ---")
print(renda_adulta.schema)

# 3. Apresente as dimensões da tabela de dados.
print("\n--- Dimensões da Tabela (Linhas, Colunas) ---")
print(renda_adulta.shape)

# 4. Quantas pessoas recebem acima de $50.000 e quantas pessoas recebem abaixo deste limiar?
print("\n--- Estimativa Populacional Ponderada (fnlwgt) ---")
contagem_ponderada = renda_adulta.group_by("income").agg(
    pl.col("fnlwgt").sum().alias("total_populacao")
)
print(contagem_ponderada)

# 5. Crie um objeto chamado renda_longo, no qual você transforma as colunas capital-gain e 
# capital-loss (formato wide) para formato longo. Os valores destas variáveis devem ser armazenados numa 
# nova coluna chamada Valor e os tipos de valores (gain e loss) devem ser armazenados numa coluna chamada tipo.
cols_index = [
    c
    for c in renda_adulta.columns
    if c not in ["capital-gain", "capital-loss"]
]

renda_longo = renda_adulta.unpivot(
    index=cols_index,
    on=["capital-gain", "capital-loss"],
    variable_name="tipo",
    value_name="valor",
).with_columns(
    pl.col("tipo").replace({"capital-gain": "gain", "capital-loss": "loss"})
)

# 6. Quais são as médias de horas trabalhadas por classe salarial?
print("\n--- Média de Horas Trabalhadas Semanalmente por Classe Salarial ---")
hora_media = renda_adulta.group_by("income").agg(
    pl.col("hours-per-week").mean().alias("media_simples"),
    (
        (pl.col("hours-per-week") * pl.col("fnlwgt")).sum()
        / pl.col("fnlwgt").sum()
    ).alias("media_ponderada"),
)
print(hora_media)

# 7. Se cada linha representa uma pessoa, quantas pessoas foram amostradas em cada profissão?
print("\n--- Pessoas Amostradas por Profissão ---")
pessoas_por_profissao = renda_adulta.group_by("occupation").len().sort(
    "len", descending=True
)
print(pessoas_por_profissao)

# 8. Crie um gráfico de barras que apresente o número médio de horas trabalhadas semanalmente em função do nível salarial.
df_horas_plot = hora_media.to_pandas()

plt.figure(figsize=(7, 5))
sns.barplot(
    data=df_horas_plot, x="income", y="media_simples", palette="viridis"
)
plt.title("Média Semanal de Horas Trabalhadas por Classe Salarial")
plt.xlabel("Classe Salarial")
plt.ylabel("Média de Horas Semanais")
plt.ylim(0, 50)

for idx, row in df_horas_plot.iterrows():
    plt.text(
        idx,
        row["media_simples"] + 0.5,
        f"{row['media_simples']:.1f}h",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.show()

# 9. Desafio: existe alguma evidência de discriminação salarial entre gêneros biológicos?
print("\n--- Análise da Disparidade por Gênero ---")

proporcao_renda_genero = renda_adulta.group_by("sex").agg(
    pl.len().alias("total"),
    (pl.col("income") == ">50K").sum().alias("qtd_acima_50k"),
    ((pl.col("income") == ">50K").mean() * 100).alias("pct_acima_50k"),
    pl.col("hours-per-week").mean().alias("media_horas_semana"),
)
print(proporcao_renda_genero)

controlado = (
    renda_adulta.filter(
        (pl.col("hours-per-week") >= 40)
        & pl.col("education").is_in(
            ["Bachelors", "Masters", "Doctorate", "Prof-school"]
        )
    )
    .group_by("sex")
    .agg(
        pl.len().alias("total_qualificados_fulltime"),
        ((pl.col("income") == ">50K").mean() * 100).alias(
            "pct_acima_50k_controlado"
        ),
    )
)
print(
    "\n--- Proporção de >50K Controlando por Escolaridade Alta + Carga Horária Full-time (>=40h) ---"
)
print(controlado)

timestamp = Path(__file__).stat().st_mtime
data_modificacao = datetime.fromtimestamp(timestamp)
print(
    f"Script editado pela última vez em: {data_modificacao.strftime('%d/%m/%Y às %H:%M:%S')}"
)