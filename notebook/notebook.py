import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 0. Software
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    # Get the path to the “notebook/” directory
    notebook_dir = mo.notebook_dir()

    # Get the path to the "datasets/" directory 
    datasets_dir = notebook_dir.parent / "datasets"

    # Get the path to the "src/" directory
    src_dir = notebook_dir.parent / "src"

    # Add the source directory to the search Python path
    import sys
    sys.path.append(str(src_dir))
    return datasets_dir, mo


@app.cell
def _():
    import polars as pl
    import numpy as np

    import time

    from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples, homogeneity_completeness_v_measure, davies_bouldin_score
    from sklearn.metrics.cluster import contingency_matrix

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    return (
        ColumnTransformer,
        MinMaxScaler,
        StandardScaler,
        cm,
        np,
        pl,
        plt,
        silhouette_samples,
        silhouette_score,
        sns,
        time,
    )


@app.cell
def _():
    SEED = 42
    return (SEED,)


@app.cell
def _(mo):
    mo.md(r"""
    # 1. Introduction
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.1. Data loading
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    [ECV2024](https://www.ine.es/dyngs/Prensa/ECV2024.htm)
    """)
    return


@app.cell
def _(datasets_dir, pl):
    # Load the dataset "ECV" into a dataframe
    df_raw = pl.read_csv(datasets_dir / "ECV.csv", separator=',')
    # Display the dataframe
    df_raw
    return (df_raw,)


@app.cell
def _(df_raw):
    # Display information about the dataframe's columns
    df_raw.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.2. Exploratory Data Analysis
    """)
    return


@app.cell
def _(df_raw):
    print(f"Number of duplicated rows: {df_raw.is_duplicated().sum()}")
    return


@app.cell
def _(df_raw):
    {c: df_raw[c].unique() for c in df_raw.columns}
    return


@app.cell
def _():
    # constant_columns = [c for c in df_raw.columns if df_raw[c].unique().shape[0] == 1]
    # print(f"Columns with constant values: {constant_columns}")
    # indexlike_columns = [c for c in df_raw.columns if df_raw[c].unique().shape[0] == df_raw.height]
    # print(f"Columns with different values in each row: {indexlike_columns}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Basic information
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - HB010: Survey year.
    - HB020: Country.
    - HB030: Transverse identification of the household.
    - HB050: Month of the household interview.
    - HB060: Year of the household interview.
    - HB070: Personal identification of the respondent.
    - HB080: Identification of the primary household head.
    - HB100: Number of minutes spent completing the household questionnaire.
    - HB120: Number of household members.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(df_raw):
    df_raw['HB010','HB020','HB030','HB050','HB050_F','HB060','HB060_F','HB070','HB070_F','HB080','HB080_F','HB100','HB100_F','HB120','HB120_F']
    return


@app.cell
def _(df_raw, pl):
    df_raw.select(
        pl.col('HB120').cast(pl.Int8).alias("num_household_members")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Household income and wealth
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - HY020: Total disposable household income in the year prior to the survey (note: this variable includes income received from private pension schemes).
    - HY022: Total disposable household income before social transfers, excluding retirement and survivor benefits, in the year prior to the survey.
    - HY023: Total household disposable income before social transfers, including retirement and survivor benefits in the year prior to the survey.
    """)
    return


@app.cell
def _(df_raw):
    df_raw['HY020','HY020_F','HY022','HY022_F','HY023','HY023_F']
    return


@app.cell
def _(df_raw, pl):
    df_raw.select(
        pl.col('HY023').str.strip_chars().replace("","0").cast(pl.Float32).alias('net_disposable_income')
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **INCOME: SERIES OF NET INCOME VARIABLES**
    - HY030N: Imputed rent (Imputed rent applies to households that do not pay full rent because they own their home or occupy a rented dwelling at below market rates or free of charge. The value imputed is equivalent to the rent that would be paid on the market for a dwelling similar to the one occupied, minus any rent actually paid).
    - HY040N: Net income from renting a property or land in the year prior to the survey.
    - HY050N: Family/child support in the year prior to the survey.
    - HY060N: Income from social assistance in the year prior to the survey.
    - HY070N: Housing assistance in the year prior to the survey.
    - HY080N: Regular monetary transfers received from other households in the year prior to the survey.
    - HY081N: Regular monetary transfers received from other households in the year prior to the survey. (child support or spousal support)
    - HY090N: Interest, dividends, and net gains from capital investments in unincorporated businesses in the year prior to the survey
    - HY100N: Interest paid on the loan for the purchase of the main residence in the year prior to the survey.
    - HY110N: Net income received by children under 16 in the year prior to the survey.
    - HY120N: Income tax in the year prior to the survey (IRPF in Spain).
    - HY130N: Regular monetary transfers paid to other households in the year prior to the survey.
    - HY131N: Regular monetary transfers paid to other households in the year prior to the survey (child support or spousal support).
    - HY145N: Refunds/supplementary income due to tax adjustments in the year prior to the survey (income tax return).
    - HY170N: Own consumption in the year prior to the survey (value of goods and services that a household produces for itself).

    We'll consider the following variables: ''
    """)
    return


@app.cell
def _(df_raw):
    df_raw['HY030N','HY030N_F','HY040N','HY040N_F','HY050N','HY050N_F','HY060N','HY060N_F','HY070N','HY070N_F','HY080N','HY080N_F', 'HY081N','HY081N_F','HY090N','HY090N_F','HY100N','HY100N_F','HY110N','HY110N_F','HY120N','HY120N_F','HY130N','HY130N_F','HY131N','HY131N_F','HY145N','HY145N_F','HY170N','HY170N_F'] 
    return


@app.cell
def _(df_raw, pl):
    df_raw.select(
        pl.col('HY120N').str.strip_chars().replace("","0").cast(pl.Float32).alias('income_tax'),
        pl.col('HY040N').str.strip_chars().replace("","0").cast(pl.Float32).alias('net_income_from_renting')
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **INCOME: SERIES OF GROSS INCOME VARIABLES**
    - HY010: Total gross household income in the year prior to the survey.
    - HY040G: Gross income from renting a property or land in the year prior to the survey.
    - HY050G: Family/child support in the year prior to the survey.
    - HY060G: Social assistance income in the year prior to the survey.
    - HY070G: Housing assistance in the year prior to the survey.
    - HY080G: Regular cash transfers received from other households in the year prior to the survey.
    - HY081G: Regular cash transfers received from other households in the year prior to the survey (child support or spousal support).
    - HY090G: Interest, dividends, and gross profits from capital investments in unincorporated businesses in the year prior to the survey.
    - HY100G: Interest paid on the loan for the purchase of the main residence in the year prior to the survey.
    - HY110G: Gross income received by children under 16 in the year prior to the survey.
    - HY120G: Wealth tax in the year prior to the survey. From ECV2021 onwards, this variable includes property tax on the main residence, when the tenure is ownership.
    - HY130G: Regular monetary transfers paid to other households in the year prior to the survey.
    - HY131G: Regular cash transfers paid to other households in the year prior to the survey (child support or spousal support).
    - HY140G: Income tax and social security contributions.
    """)
    return


@app.cell
def _(df_raw, pl):
    df_raw.select(
        pl.col('HY010').str.strip_chars().replace("","0").cast(pl.Float32).alias('total_gross_household_income'),
        pl.col('HY140G').str.strip_chars().replace("","0").cast(pl.Float32).alias('income_tax_social_security'), 
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Social exclusion
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - HS011: Have there been any delays in paying the mortgage or rent for the home in the last 12 months?
    - HS021: Have there been any delays in paying bills?
    - HS022: Did the household benefit from any social subsidies to cover electricity, heating, gas, etc. expenses?
    - HS031: Have there been any delays in paying for deferred purchases or other loans (debts not related to the main residence) in the last 12 months?
    - HS040: Can the household afford to go on vacation away from home at least one week per year?
    - HS050: Can the household afford a meal of meat, chicken, or fish (or equivalent for vegetarians) at least every other day?
    - HS060: Does the household have the capacity to meet unexpected expenses?
    - HS090: Does the household have a computer?
    - HS110: Does the household have a car?
    - HS120: Household's ability to make ends meet
    - HS150: Expenditures for installment purchases or loan repayments not related to the main residence represent a burden for the household.
    - HD080: Could the household replace damaged or old furniture?
    """)
    return


@app.cell
def _(df_raw):
    df_raw['HS011', 'HS011_F', 'HS021','HS021_F','HS022','HS022_F','HS031','HS031_F','HS040','HS040_F','HS050','HS050_F','HS060','HS060_F','HS090','HS090_F','HS110','HS110_F', 'HS120','HS120_F','HS150','HS150_F','HD080','HD080_F']
    return


@app.cell
def _(df_raw, pl):
    T_2SiNo_code={
        1: 1,
        2: 2,
        3: 0,
        None: None
    }

    T_SiNo_code={
        1: True,
        2: False,
        None: None
    }

    df_raw.select(
        pl.col('HS011').str.strip_chars().replace("",None).cast(pl.Int8).replace_strict(T_2SiNo_code).alias('delay_mortgage_rent'),
        pl.col('HS021').replace_strict(T_2SiNo_code).cast(pl.Int8).alias('delay_bills'),
        pl.col('HS040').replace_strict(T_SiNo_code).cast(pl.Boolean).alias('afford_week_vacation_out'),
        pl.col('HS050').replace_strict(T_SiNo_code).cast(pl.Boolean).alias('afford_expensive_meal'),
    )
    return T_2SiNo_code, T_SiNo_code


@app.cell
def _(mo):
    mo.md(r"""
    ### Housing
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - HH010: Tipo de vivienda
    - HH021: Régimen de tenencia
    - HH030: Número de habitaciones de la vivienda
    - HH050: ¿Puede el hogar permitirse mantener la vivienda con una temperatura adecuada durante los meses de invierno?
    - HH060: Alquiler actual por la vivienda ocupada.
    - HH070: Gastos de la vivienda: Alquiler (si la vivienda se encuentra en régimen de alquiler), intereses de la hipoteca (para viviendas en propiedad con pagos pendientes) y otros gastos asociados (comunidad, agua, electricidad, gas, etc.)
    """)
    return


@app.cell
def _(df_raw):
    df_raw['HH010','HH010_F','HH021','HH021_F','HH030','HH030_F','HH050','HH050_F','HH060','HH060_F','HH070','HH070_F']
    return


@app.cell
def _(T_SiNo_code, df_raw, pl):
    TH010H_code={
        1: 'Detached single-family home',
        2: 'Semi-detached or attached single-family home',
        3: 'Flat or apartment in a building with fewer than 10 dwellings',
        4: 'Flat or apartment in a building with 10 or more dwellings',
        None: None
    }

    TH021H_code={
        1: 'Owned without mortgage',
        2: 'Owned with mortgage',
        3: 'Rented or sublet at market rate',
        4: 'Rented or sublet below market rate',
        5: 'Occupied rent-free',
        None: None,
    }

    df_raw.select(
        pl.col('HH010').replace_strict(TH010H_code).alias("housing_type"),
        pl.col('HH021').replace_strict(TH021H_code).alias("tenure_regime"),
        pl.col('HH030').alias('number_of_rooms'), 
        pl.col('HH050').replace_strict(T_SiNo_code).cast(pl.Boolean).alias('afford_heating_house_in_winter'),
        pl.col('HH060').str.strip_chars().replace("",None).cast(pl.Float32).alias('rent_price'), 
        pl.col('HH070').str.strip_chars().replace("",None).cast(pl.Float32).alias('housing_expenditure'),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Excessive debt
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - HI010: Has there been a change in your household income in the last 12 months?
    - HI020: What was the main reason for the increase in income?
    - HI030: What was the main reason for the decrease in income?
    - HI040: Thinking about your household income, what do you expect to happen in the next 12 months?
    """)
    return


@app.cell
def _(df_raw):
    df_raw['HI010','HI010_F','HI020','HI020_F','HI030','HI030_F','HI040','HI040_F']
    return


@app.cell
def _(df_raw, pl):
    TH010I_code={
        1: 'Household income has increased',
        2: 'Household income has remained',
        3: 'Household income has decreased',
        None: None,
    }

    TH020I_code={
        1: 'Annual salary increase',
        2: 'Increase in hours worked or salary at current job',
        3: 'Return to work after absence due to illness, maternity/paternity leave, childcare, or care for sick or elderly relatives',
        4: 'Start of new job or change of job',
        5: 'Changes in household composition',
        6: 'Receipt or increase in social benefits',
        7: 'Other reasons',
        None: None,
    }

    TH030I_code={
        1: 'Reduction in hours worked or salary from current job',
        2: 'Maternity/paternity, childcare, or care for sick or elderly relatives',
        3: 'Change of job',
        4: 'Job loss/unemployment',
        5: 'Inability to work due to illness or disability',
        6: 'Divorce, separation, or other changes in household composition',
        7: 'Retirement',
        8: 'Elimination or reduction of social benefits',
        9: 'Other reasons',
        None: None,
    }

    TH040I_code={
        1: 'Improve',
        2: 'Remain',
        3: 'Decrease',
        None: None,
    }

    df_raw.select(
        pl.col('HI010').replace_strict(TH010I_code).alias("income_change_last_year"),
        pl.col('HI020').str.strip_chars().replace("",None).cast(pl.Int8).replace_strict(TH020I_code).alias("main_reason_increase_income"),
        pl.col('HI030').str.strip_chars().replace("",None).cast(pl.Int8).replace_strict(TH030I_code).alias("main_reason_decrease_income"),
        pl.col('HI040').str.strip_chars().replace("",None).cast(pl.Int8).replace_strict(TH040I_code).alias("income_expectations_next_year")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Access to services
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - HC190: Does anyone in your household need help because they are elderly or have a chronic illness?
    - HC200: Do you receive home care from a paid caregiver?
    - HC221: Who pays for the cost of this care?
    - HC230: How can the household afford to pay for this care?
    - HC240: Does anyone in your household need care or more home care from a paid caregiver?
    - HC250: Main reason why you do not receive them.
    - HC300: Home charge for public transportation costs.
    """)
    return


@app.cell
def _(df_raw):
    df_raw['HC190','HC190_F','HC200','HC200_F','HC221','HC221_F','HC230','HC230_F','HC240','HC240_F','HC250','HC250_F','HC300','HC300_F']
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.3. Preproccesing
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Remove the columns 'HB010', 'HB020'.
    - Remove the columns 'HB030', 'HB090' (and therefore 'BH080_F').
    """)
    return


@app.cell
def _(df_raw):
    df_raw['HH010','HH010_F']
    return


@app.cell
def _(mo):
    mo.md(r"""
    - 'HY010' as 'total_gross_household_income': Total income household before taxes and deductions.
    - 'HY023' as 'net_disponable_income': The income that the household actually has to spend or save after all obligations (taxes, social security, transfers paid).
    - 'HY140G' as 'income_tax': It taxes incomes (IRPF in Spain).
    - 'HY120G' as 'wealth_tax': It taxes asset ownership (IBI in Spain).
    """)
    return


@app.cell
def _(df_raw, pl):

    df_final = df_raw.select(
        pl.col('HB120').cast(pl.Int8).alias('num_household_members'),
    )
    df_final
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Study case I: Brechas de Cuidados 2024: Segmentación de la Demanda y Solvencia en el Hogar
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(df_raw):
    df_raw['HC190']
    return


@app.cell
def _(df_raw, pl):
    TH190C_code={
        1: 1,  # Yes
        2: None   # No
    }

    TH221C_code={
    
    }

    df_caseI_raw = df_raw.select(
        pl.when(pl.col('HC190_F') < 0).then(None).otherwise(pl.col('HC190').replace_strict(TH190C_code)).alias('HY020'),
        pl.when(pl.col('HC200_F') < 0).then(None).otherwise(pl.col('HC200').replace_strict(TH190C_code)).alias('HY200'),
        pl.when(pl.col('HC210_F') < 0).then(None).otherwise(pl.col('HC210').replace_strict(TH190C_code)).alias('HY210'),
        pl.when(pl.col('HC220_F') < 0).then(None).otherwise(pl.col('HC220').replace_strict(TH190C_code)).alias('HY220'),
    )

    df_caseI_raw
    return (df_caseI_raw,)


@app.cell
def _(df_caseI_raw, pl):
    df_caseI = df_caseI_raw.select(
        (pl.col('HY020')/pl.col('HX240')).cast(pl.Float32).alias('equivalised_disposable_income'),
        pl.when(pl.col('HH021').is_in([3,4])).then(1).otherwise(0).cast(pl.Int8).alias('tenant'), # Codes 3 and 4 corresponds to rented
        pl.when(pl.col('HY040N')>0).then(1).otherwise(0).cast(pl.Int8).alias('landlord'),
        pl.col('HY040N').alias('net_income_from_renting'),
        pl.col('HH060').alias('occupied_dwelling_rent'),
        # pl.col('HH070').alias('housing_expenses')
    ).drop_nulls()

    df_caseI
    return (df_caseI,)


@app.cell
def _(df_caseI, np, plt, sns):
    from matplotlib.ticker import FuncFormatter

    sns.histplot(df_caseI['equivalised_disposable_income'], bins=50)
    plt.yscale("log", base=2)

    ymin, ymax = plt.ylim()
    exp_min = int(np.floor(np.log2(ymin))) if ymin > 0 else 0
    exp_max = int(np.ceil(np.log2(ymax)))

    ticks = [2**k for k in range(exp_min, exp_max + 1)]
    plt.yticks(ticks)

    # Formatear como potencias
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"$2^{{{int(np.log2(v))}}}$")
    )

    plt.show()
    return


@app.cell
def _(ColumnTransformer, MinMaxScaler, StandardScaler, df_caseI):
    transformer_caseI = ColumnTransformer(
        transformers=[
            ('equivalised_disposable_income', StandardScaler(), ['equivalised_disposable_income']),
            ('tenant', MinMaxScaler(), ['tenant']),
            ('landlord', MinMaxScaler(), ['landlord']),
            ('net_income_from_renting', StandardScaler(), ['net_income_from_renting']),
            ('occupied_dwelling_rent', StandardScaler(), ['occupied_dwelling_rent']),
            # ('housing_expenses', StandardScaler(), ['housing_expense    s']),
        ]
    )
    transformer_caseI.set_output(transform="polars")
    df_caseI_norm=transformer_caseI.fit_transform(df_caseI)
    df_caseI_norm
    return (df_caseI_norm,)


@app.cell
def _(SEED, resulsts, sklearn, time, training_time):
    def kmeans(df, n_clusters, n_init=None, random_state=SEED):
        if n_init is None:
            n_init = 10 * n_clusters
        kmeans = sklearn.cluster.KMeans(n_clusters)
        start_time = time.time()
        cluster_labels = kmeans.fit_predict(df)
        end_time = time.time()
        return cluster_labels, training_time

    def test_k_kmeans(df, range_n_clusters):
        results = {'n_clusters':[], 'silhouette_score': [], 'training_time': []}
        for n_clusters in range_n_clusters:
            resulsts['n_clusters'].append(n_clusters)
            cluster_labels, training_time = kmeans(df, n_clusters, 100*n_clusters)
    
    
    return


@app.cell
def _():
    # # Define los valores del hiperparámetro 'n_clusters'
    # range_n_clusters_1 = range(2,13)

    # # Almacena las métricas calculadas
    # results_1 = {'n_clusters':[], 'inertia': [], 'silhouette_score': [], 'training_time': []}

    # # Entrena los modelos KMeans y calcula y guarda las métricas con cada valor de 'n_clusters'
    # for n_clusters in range_n_clusters:

    #     results_1['n_clusters'].append(n_clusters)
    #     kmeans_1 = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10*n_clusters, random_state=42)

    #     # Medir el tiempo de entrenamiento
    #     start_time = time.time()
    #     cluster_labels = kmeans.fit_predict(df_normalized)
    #     end_time = time.time()

    #     # Calcular el tiempo de entrenamiento
    #     training_time = end_time - start_time
    #     results_1['training_time'].append(training_time)

    #     results_1['inertia'].append(kmeans.inertia_)
    #     results_1['silhouette_score'].append( silhouette_score(df_normalized, pred_labels) )
    return


@app.cell
def _():
    return


@app.cell
def _():
    # def evaluate_clustering(df, cluster_labels):

    #     silhouette = silhouette_score(df, cluster_labels)
    #     davies_bouldin = davies_bouldin_score(df, cluster_labels)
    #     calinski_harabasz = calinski_harabasz_score(df, cluster_labels)

    return


@app.cell
def _():
    # def choose_n_clusters_kmeans(range_n_clusters, df):

    #     results = {'n_clusters':[], 'inertia': [], 'silhouette_score': [], 'training_time': []}

    #     for n_clusters in range_n_clusters:

    #         results['n_clusters'].append(n_clusters)
    #         kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10*n_clusters, random_state=42)

    #         # Medir el tiempo de entrenamiento
    #         start_time = time.time()
    #         pred_labels = kmeans.fit_predict(df)
    #         end_time = time.time()

    #         # Calcular el tiempo de entrenamiento
    #         training_time = end_time - start_time
    #         results['training_time'].append(training_time)
    #         results['inertia'].append(kmeans.inertia_)
    #         results['silhouette_score'].append( silhouette_score(df, pred_labels) )

    #      # Convertimos resultados a Polars
    #     results_pl = pl.DataFrame(results)

    #     # Elegir el n_clusters con menor silhouette
    #     best_n_clusters = (
    #         results_pl
    #         .sort('silhouette_score', descending=False)
    #         .select('n_clusters')
    #         .item()  # devuelve el valor como Python scalar
    #     )

    #     return best_n_clusters, results_pl
    return


@app.cell
def _():
    # # n_clusters = choose_n_clusters_kmeans(range(2,11,1), df_caseI_norm)
    # n_clusters = 6
    return


@app.cell
def _():

    # kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10*n_clusters, random_state=42)

    # # Medir el tiempo de entrenamiento
    # cluster_labels = kmeans.fit_predict(df_caseI_norm)
    return


@app.cell
def _():
    # def plot_silhouette(X, cluster_labels, ax=None, width=1000, height=1000):

    #     if ax is None:
    #     # Convertimos width y height a pulgadas, ya que plt.subplots usa pulgadas
    #         fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    #         own_ax = True  # Indicador de que el eje fue creado dentro de esta función
    #     else:
    #         own_ax = False

    #     n_clusters = len(np.unique(cluster_labels))

    #     ax.set_xlim([-0.1, 1])
    #     ax.set_ylim([0, len(X) + (n_clusters + 1) * 5])

    #     #
    #     silhouette_avg = silhouette_score(X, cluster_labels)

    #     # Compute the silhouette scores for each sample
    #     sample_silhouette_values = silhouette_samples(X, cluster_labels)

    #     y_lower = 6
    #     for i in range(n_clusters):
    #         #
    #         ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

    #         ith_cluster_silhouette_values.sort()

    #         size_cluster_i = ith_cluster_silhouette_values.shape[0]
    #         y_upper = y_lower + size_cluster_i

    #         color = cm.nipy_spectral(float(i) / n_clusters)
    #         ax.fill_betweenx(
    #             np.arange(y_lower, y_upper),
    #             0,
    #             ith_cluster_silhouette_values,
    #             facecolor=color,
    #             edgecolor=color,
    #             alpha=0.7,
    #         )

    #         # Label the silhouette plots with their cluster numbers at the middle
    #         ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    #         # Compute the new y_lower for next plot
    #         y_lower = y_upper + 5

    #         ax.set_xlabel("Coeficiente de silueta", loc='right', fontweight='bold')
    #         ax.set_ylabel("Etiqueta del cluster", loc='top', fontweight='bold')

    #         # The vertical line for average silhouette score of all the values
    #         ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    #         ax.set_yticks([])  # Clear the yaxis labels / ticks
    #         ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    #     plt.show()
    return


@app.cell
def _(cluster_labels, df_caseI_norm, plot_silhouette):
    plot_silhouette(df_caseI_norm, cluster_labels)
    return


@app.cell
def _():
    # # Define los valores del hiperparámetro 'n_clusters'
    # range_n_clusters = range(2,11,1)

    # # Almacena las métricas calculadas
    # results = {'n_clusters':[], 'inertia': [], 'silhouette_score': [], 'training_time': []}

    # #
    # print(" Num. clusters | Training time ")

    # # Entrena los modelos KMeans y calcula y guarda las métricas con cada valor de 'n_clusters'
    # for n_clusters in range_n_clusters:

    #     results['n_clusters'].append(n_clusters)
    #     kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=30, random_state=42)

    #     # Medir el tiempo de entrenamiento
    #     start_time = time.time()
    #     pred_labels = kmeans.fit_predict(df_caseI_norm)
    #     end_time = time.time()

    #     # Calcular el tiempo de entrenamiento
    #     training_time = end_time - start_time
    #     results['training_time'].append(training_time)

    #     results['inertia'].append(kmeans.inertia_)
    #     results['silhouette_score'].append( silhouette_score(df_caseI_norm, pred_labels) )

    #     print(f"{n_clusters:>15}| {training_time:>15}" )
    return


@app.cell
def _():
    # # Define los valores del hiperparámetro 'n_clusters'
    # range_n_clusters = range(2,11,1)

    # # Almacena las métricas calculadas
    # results = {'n_clusters':[], 'inertia': [], 'silhouette_score': [], 'training_time': []}

    # #
    # print(" Num. clusters | Training time ")

    # # Entrena los modelos KMeans y calcula y guarda las métricas con cada valor de 'n_clusters'
    # for n_clusters in range_n_clusters:

    #     results['n_clusters'].append(n_clusters)
    #     kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=2**(2+n_clusters), random_state=42)

    #     # Medir el tiempo de entrenamiento
    #     start_time = time.time()
    #     pred_labels = kmeans.fit_predict(df_caseI_norm)
    #     end_time = time.time()

    #     # Calcular el tiempo de entrenamiento
    #     training_time = end_time - start_time
    #     results['training_time'].append(training_time)

    #     results['inertia'].append(kmeans.inertia_)
    #     results['silhouette_score'].append( silhouette_score(df_caseI_norm, pred_labels) )

    #     print(f"{n_clusters:>15}| {training_time:>15}" )
    return


@app.cell
def _(plt, range_n_clusters, results):
    # Grafica los resultados
    plt.figure(figsize=(12.5, 6.25))

    # Subplot para Inercia
    plt.subplot(1, 2, 1)
    plt.plot(range_n_clusters, results['inertia'], marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia')
    plt.title('Inercia para Diferentes Números de Clusters')
    plt.grid(True)
    plt.xticks(range(2, 16))

    # Subplot para Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(range_n_clusters, results['silhouette_score'], marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score para Diferentes Números de Clusters')
    plt.grid(True)
    plt.xticks(range(2, 16))

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 2. Study case I: Financial Vulnerability and Economic Stress Profiles
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This study seeks to go beyond simple income levels and group households according to their actual capacity for economic survival and financial stress. The aim is to differentiate between “income-poor,” “indebted households,” and “resilient households.”
    """)
    return


@app.cell
def _(T_2SiNo_code, T_SiNo_code, df_raw, pl):
    TH120S_code_caseIII={
        1: 1,   # 'Con mucha dificultad'
        2: 2,   # 'Con dificultad'
        3: 3,   # 'Con cierta dificultad'
        4: 4,   # 'Con cierta facilidad'
        5: 5,   # 'Con facilidad'
        6: 6,   # 'Con mucha facilidad'
        None: None
    }

    T_SiNo_code_caseIII={
        1: 1,
        2: 0,
        None: None
    }

    T_2SiNo_code_caseIII={
        1: 1,
        2: 2,
        3: 0,
        None: None
    }

    df_caseIII = df_raw.select(

        pl.when(pl.col('HY020_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY020').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('total_gross_income'),

        pl.when(pl.col('HB120_F') == -2)
            .then( None )
            .otherwise( pl.col('HS120').cast(pl.Int8) )
            .alias('num_household_members'),

        pl.when(pl.col('HS120_F') == -2)
            .then( None )
            .otherwise( pl.col('HS120').replace_strict(TH120S_code_caseIII).cast(pl.Int8) )
            .alias('ease_to_make_ends_meet'),

        pl.when(pl.col('HS060_F') == -2)
            .then( None )
            .otherwise( pl.col('HS060').replace_strict(T_SiNo_code_caseIII).cast(pl.Int8) )
            .alias('capacity_to_cover_unexpected_expenses'),

        pl.when(pl.col('HS011_F') == -2)
            .then( pl.lit(0).cast(pl.Int8) )
            .otherwise( pl.col('HS011').str.strip_chars().replace("",None).cast(pl.Int8).replace_strict(T_2SiNo_code).cast(pl.Int8) )
            .alias('delay_mortgage_rent'),

        pl.when(pl.col('HS021_F') == -2)
            .then( pl.lit(0).cast(pl.Int8) )
            .otherwise( pl.col('HS021').replace_strict(T_2SiNo_code).cast(pl.Int8) )
            .alias('delay_bills_residence'),

        pl.when(pl.col('HS031_F') == -2)
            .then( pl.lit(0).cast(pl.Int8) )
            .otherwise( pl.col('HS031').str.strip_chars().replace("",None).cast(pl.Int8).replace_strict(T_2SiNo_code).cast(pl.Int8) )
            .alias('delay_debts_not_related_primary_residence'),

        pl.when(pl.col('HS050_F') == -2)
            .then( None )
            .otherwise( pl.col('HS050').replace_strict(T_SiNo_code).cast(pl.Int8) )
            .alias('afford_expensive_ingredients_every_2days')

    ).drop_nulls()
    df_caseIII
    return


@app.cell
def _(ColumnTransformer, StandardScaler, df_caseIII_nonulls):
    transformer = ColumnTransformer(
        transformers=[
            ('total_gross_income', StandardScaler(), ['total_gross_income']),
            ('num_household_members', StandardScaler(), ['num_household_members']),
            ('ease_to_make_ends_meet', StandardScaler(), ['ease_to_make_ends_meet']),
            ('capacity_to_cover_unexpected_expenses', StandardScaler(), ['capacity_to_cover_unexpected_expenses']),
            ('delay_mortgage_rent', StandardScaler(), ['delay_mortgage_rent']),
            ('delay_bills_residence', StandardScaler(), ['delay_bills_residence']),
            ('delay_debts_not_related_primary_residence', StandardScaler(), ['delay_debts_not_related_primary_residence']),
            ('afford_expensive_ingredients_every_2days', StandardScaler(), ['afford_expensive_ingredients_every_2days'])
        ]
    )
    transformer.set_output(transform="polars")
    df_caseIII_norm=transformer.fit_transform(df_caseIII_nonulls)
    df_caseIII_norm
    return


@app.cell
def _():
    return


@app.cell
def _():
    # # Define los valores del hiperparámetro 'n_clusters'
    # range_n_clusters = range(2,11,1)

    # # Almacena las métricas calculadas
    # results = {'n_clusters':[], 'inertia': [], 'silhouette_score': [], 'training_time': []}

    # #
    # print(" Num. clusters | Training time ")

    # # Entrena los modelos KMeans y calcula y guarda las métricas con cada valor de 'n_clusters'
    # for n_clusters in range_n_clusters:

    #     results['n_clusters'].append(n_clusters)
    #     kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=30, random_state=42)

    #     # Medir el tiempo de entrenamiento
    #     start_time = time.time()
    #     pred_labels = kmeans.fit_predict(df_caseIII_norm)
    #     end_time = time.time()

    #     # Calcular el tiempo de entrenamiento
    #     training_time = end_time - start_time
    #     results['training_time'].append(training_time)

    #     results['inertia'].append(kmeans.inertia_)
    #     results['silhouette_score'].append( silhouette_score(df_caseIII_norm, pred_labels) )

    #     print(f"{n_clusters:>15}| {training_time:>15}" )
    return


@app.cell
def _():
    # # Grafica los resultados
    # plt.figure(figsize=(12.5, 6.25))

    # # Subplot para Inercia
    # plt.subplot(1, 2, 1)
    # plt.plot(range_n_clusters, results['inertia'], marker='o')
    # plt.xlabel('Número de Clusters')
    # plt.ylabel('Inercia')
    # plt.title('Inercia para Diferentes Números de Clusters')
    # plt.grid(True)
    # plt.xticks(range(2, 16))

    # # Subplot para Silhouette Score
    # plt.subplot(1, 2, 2)
    # plt.plot(range_n_clusters, results['silhouette_score'], marker='o')
    # plt.xlabel('Número de Clusters')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Score para Diferentes Números de Clusters')
    # plt.grid(True)
    # plt.xticks(range(2, 16))

    # plt.tight_layout()
    # plt.show()
    return


@app.cell
def _():
    # # Almacena las métricas calculadas
    # results_overall = {'algorithm': [], 'n_clusters':[],
    #            'inertia': [], 'calinski_harabasz': [],
    #            'silhouette_values': [],'silhouette_score': [],
    #            'training_time': []}


    # results_overall['algorithm'].append("KMeans (k=6)")
    # n_clustersI = 6
    # results_overall['n_clusters'].append(n_clustersI)
    # kmeansI = KMeans(n_clusters=n_clustersI, init='k-means++', n_init=40, random_state=42)

    # # Medir el tiempo de entrenamiento
    # start_time_I = time.time()
    # cluster_labels = kmeansI.fit_predict(df_caseIII_norm)
    # end_time_I = time.time()

    # # Calcular el tiempo de entrenamiento
    # training_time_I = end_time_I - start_time_I
    # results_overall['training_time'].append(training_time_I)

    # results_overall['inertia'].append(kmeansI.inertia_)
    # results_overall['calinski_harabasz'].append( calinski_harabasz_score(df_caseIII_norm, cluster_labels) )
    # results_overall['silhouette_score'].append( silhouette_score(df_caseIII_norm, cluster_labels) )

    # results_overall['silhouette_values'].append(silhouette_samples(df_caseIII_norm, cluster_labels))
    return


@app.cell
def _(cm, np, plt, silhouette_samples, silhouette_score):
    def plot_silhouette(X, cluster_labels, ax=None, width=1000, height=1000):

        if ax is None:
        # Convertimos width y height a pulgadas, ya que plt.subplots usa pulgadas
            fig, ax = plt.subplots(figsize=(width / 100, height / 100))
            own_ax = True  # Indicador de que el eje fue creado dentro de esta función
        else:
            own_ax = False

        n_clusters = len(np.unique(cluster_labels))

        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(X) + (n_clusters + 1) * 5])

        #
        silhouette_avg = silhouette_score(X, cluster_labels)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 6
        for i in range(n_clusters):
            #
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 5

            ax.set_xlabel("Coeficiente de silueta", loc='right', fontweight='bold')
            ax.set_ylabel("Etiqueta del cluster", loc='top', fontweight='bold')

            # The vertical line for average silhouette score of all the values
            ax.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax.set_yticks([])  # Clear the yaxis labels / ticks
            ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    return (plot_silhouette,)


@app.cell
def _(mo):
    mo.md(r"""
    # 3. Study case II: Income sources
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This study case aims to ...
    """)
    return


@app.cell
def _(df_raw, pl):
    df_caseII_raw = df_raw.select(

        pl.when(pl.col('HB120_F') == -2)
            .then( None )
            .otherwise( pl.col('HS120').cast(pl.Int8) )
            .alias('num_household_members'),

        pl.when(pl.col('HY020_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY020').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('net_disposable_income'),

        pl.when(pl.col('HY010_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY010').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('gross_total_income'),

        #----------------------------------------------------------------------------------------

        pl.when(pl.col('HY040N_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY040N').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('net_income_from_renting'),

        pl.when(pl.col('HY090N_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY090N').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('interests_dividends_net_gains_on_capital_investments'),

        pl.when(pl.col('HY030N_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY030N').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('imputed_rent'),

        pl.when(pl.col('HY100N_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY100N').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('interests_from_mortgage'),

        #----------------------------------------------------------------------------------------

        pl.when(pl.col('HY050N_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY050N').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('family_child_support'),

        pl.when(pl.col('HY060N_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY060N').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('social_assistance_income'),

        pl.when(pl.col('HY070N_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY070N').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('housing_support'),

        #----------------------------------------------------------------------------------------

        pl.when(pl.col('HY080N_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY080N').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('regular_monetary_transfers_received_from_other_households'),

        pl.when(pl.col('HY081N_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY081N').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('child_support_spousal_support_received'),

        pl.when(pl.col('HY130N_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY130N').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('regular_monetary_transfers_paid_to_other_households'),

        pl.when(pl.col('HY131N_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY131N').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('child_support_spousal_support_paid'),

        #----------------------------------------------------------------------------------------

        pl.when(pl.col('HY170N_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY170N').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('own_consumptions'),

        #----------------------------------------------------------------------------------------

        pl.when(pl.col('HY120N_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY120N').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('income_tax'),

        pl.when(pl.col('HY145N_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY145N').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('income_tax_return'),

        #----------------------------------------------------------------------------------------

        pl.when(pl.col('HY110N_F') % 10 == 9)
            .then( None )
            .otherwise( pl.col('HY110N').str.strip_chars().replace("","0").cast(pl.Float32) )
            .alias('tranfers_minors_under_16'),

    ).drop_nulls()

    df_caseII_raw
    return (df_caseII_raw,)


@app.cell
def _(df_caseII_raw, pl):

    df_caseII_raw.select(
        (pl.col('gross_total_income') -(
            - pl.col('income_tax')
            - pl.col('regular_monetary_transfers_paid_to_other_households')
            - pl.col('child_support_spousal_support_paid')
            - pl.col('interests_from_mortgage')
            + pl.col('family_child_support')
            + pl.col('social_assistance_income')
            + pl.col('housing_support')
            + pl.col('regular_monetary_transfers_received_from_other_households')
            + pl.col('child_support_spousal_support_received')
            + pl.col('tranfers_minors_under_16')
            + pl.col('income_tax_return')
            + pl.col('own_consumptions')
            + pl.col('net_income_from_renting')
            + pl.col('interests_dividends_net_gains_on_capital_investments')
        )
        ).alias('zero')
    )
    return


@app.cell
def _(df_raw, pl):
    df_tests_raw = df_raw.select(

        pl.col('HS120').cast(pl.Int8),
        pl.col('HY010').str.strip_chars().replace("","0").cast(pl.Float32),
        pl.col('HY020').str.strip_chars().replace("","0").cast(pl.Float32),
        pl.col('HY022').str.strip_chars().replace("","0").cast(pl.Float32),
        pl.col('HY023').str.strip_chars().replace("","0").cast(pl.Float32),

        pl.when(pl.col('HY030N_F') % 10 == 9).then(None).otherwise(pl.col('HY030N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY030N'),

        pl.when(pl.col('HY040G_F') % 10 == 9).then(None).otherwise(pl.col('HY040G').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY040G'),
        pl.when(pl.col('HY040N_F') % 10 == 9).then(None).otherwise(pl.col('HY040N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY040N'),

        pl.when(pl.col('HY050G_F') % 10 == 9).then(None).otherwise(pl.col('HY050G').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY050G'),
        pl.when(pl.col('HY050N_F') % 10 == 9).then(None).otherwise(pl.col('HY050N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY050N'),

        pl.when(pl.col('HY060G_F') % 10 == 9).then(None).otherwise(pl.col('HY060G').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY060G'),
        pl.when(pl.col('HY060N_F') % 10 == 9).then(None).otherwise(pl.col('HY060N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY060N'),

        pl.when(pl.col('HY070G_F') % 10 == 9).then(None).otherwise(pl.col('HY070G').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY070G'),
        pl.when(pl.col('HY070N_F') % 10 == 9).then(None).otherwise(pl.col('HY070N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY070N'),

        pl.when(pl.col('HY080G_F') % 10 == 9).then(None).otherwise(pl.col('HY080G').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY080G'),
        pl.when(pl.col('HY080N_F') % 10 == 9).then(None).otherwise(pl.col('HY080N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY080N'),

        pl.when(pl.col('HY081G_F') % 10 == 9).then(None).otherwise(pl.col('HY081G').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY081G'),
        pl.when(pl.col('HY081N_F') % 10 == 9).then(None).otherwise(pl.col('HY081N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY081N'),

        pl.when(pl.col('HY090G_F') % 10 == 9).then(None).otherwise(pl.col('HY090G').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY090G'),
        pl.when(pl.col('HY090N_F') % 10 == 9).then(None).otherwise(pl.col('HY090N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY090N'),

        pl.when(pl.col('HY100G_F') % 10 == 9).then(None).otherwise(pl.col('HY100G').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY100G'),
        pl.when(pl.col('HY100N_F') % 10 == 9).then(None).otherwise(pl.col('HY100N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY100N'),

        pl.when(pl.col('HY110G_F') % 10 == 9).then(None).otherwise(pl.col('HY110G').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY110G'),
        pl.when(pl.col('HY110N_F') % 10 == 9).then(None).otherwise(pl.col('HY110N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY110N'),

        pl.when(pl.col('HY120G_F') % 10 == 9).then(None).otherwise(pl.col('HY120G').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY120G'),
        pl.when(pl.col('HY120N_F') % 10 == 9).then(None).otherwise(pl.col('HY120N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY120N'),

        pl.when(pl.col('HY130G_F') % 10 == 9).then(None).otherwise(pl.col('HY130G').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY130G'),
        pl.when(pl.col('HY130N_F') % 10 == 9).then(None).otherwise(pl.col('HY130N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY130N'),

        pl.when(pl.col('HY131G_F') % 10 == 9).then(None).otherwise(pl.col('HY131G').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY131G'),
        pl.when(pl.col('HY131N_F') % 10 == 9).then(None).otherwise(pl.col('HY131N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY131N'),

        pl.when(pl.col('HY140G_F') % 10 == 9).then(None).otherwise(pl.col('HY140G').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY140G'),

        pl.when(pl.col('HY145N_F') % 10 == 9).then(None).otherwise(pl.col('HY145N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY145N'),

        pl.when(pl.col('HY170N_F') % 10 == 9).then(None).otherwise(pl.col('HY170N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY170N'),

    ).drop_nulls()

    df_tests_raw
    return (df_tests_raw,)


@app.cell
def _(df_tests_raw, pl):
    df_tests_raw.select(
        (pl.col('HY022')+pl.col('HY050N')+pl.col('HY060N')+pl.col('HY070N')) - pl.col('HY020')
    )
    return


@app.cell
def _():
    return


@app.cell
def _(df_tests_raw, pl):
    df_tests_raw.select(
        (pl.col('HY030N')+pl.col('HY040N')+pl.col('HY050N')+pl.col('HY060N')+pl.col('HY070N')+pl.col('HY080N')+pl.col('HY130N')+pl.col('HY090N')) - pl.col('HY020')
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    HY020 = HY010 - HY120G - HY130G - HY140G \\
    HY020 = HY022 + HY050G + HY060G + HY070G + ???
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Riqueza y recursos en familias con personas dependientes
    - Riqueza y rentismo
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
