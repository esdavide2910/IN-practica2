import marimo

__generated_with = "0.17.8"
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

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    return (pl,)


@app.cell
def _(pl):
    def analyze_missing_values(df: pl.DataFrame, show_all=False) -> pl.DataFrame:
        """
        Analyzes the missing values in a Polars DataFrame and returs an abstract.

        Parameters:
          df (pl.DataFrame): Dataframe to analyze.
          show_all (bool): If True, includes columns without missing values. False by default.

        Returns:
          pd.DataFrame: DataFrame con dos filas por columna:
                        - Número de valores faltantes.
                        - Porcentaje de valores faltantes.
        """

        # Verify that the input is a Dataframe 
        if not isinstance(df, pl.DataFrame):
            raise TypeError("The parameter 'df' must be a Polars DataFrame debe ser un DataFrame de Polars.")

        # Number of total values per column
        total_rows = df.height

        # Calculate missing values per column Calcular valores faltantes por columna
        missing_counts = df.null_count()

        # Convert Series to DataFrame 
        missing_df = (
            missing_counts
            .transpose(include_header=True)
            .melt(variable_name="column", value_name="Missing values (units)")
        )

        # Calculate the percentage of missing values
        missing_df = missing_df.with_columns(
            (pl.col("Missing values (units)") / total_rows * 100)
            .alias("Ratio missing vals. (%)")
        )

        # Filter
        if not show_all:
            missing_df = missing_df.filter(pl.col("Missing values (units)") > 0)

        return missing_df
    return


@app.cell
def _(pl):
    def analyze_unique_values(df: pl.DataFrame, col_name: str, dropna=False) -> pl.DataFrame:
        """
        Analiza los valores únicos en una columna de un DataFrame de Polars, calculando la frecuencia
        tanto en unidades como en porcentaje. También incluye los valores faltantes (NaN)
        si se especifica 'dropna=False'.

        Parameters:
            df (pl.DataFrame): El DataFrame que contiene los datos a analizar.
            col_name (str): El nombre de la columna de la que se desea obtener los valores únicos.
            dropna (bool): Si es True, excluye los valores faltantes (NaN) del análisis. Si es False,
                           incluye los valores faltantes. El valor por defecto es False.

        Returns:
            pd.DataFrame: Un DataFrame con dos columnas:
                           'Frecuencia (unidades)' con la cantidad de ocurrencias de cada valor único
                           y 'Frecuencia (%)' con el porcentaje de ocurrencias de cada valor único.
        """

        if not isinstance(df, pl.DataFrame):
            raise TypeError("El parámetro 'df' debe ser un DataFrame de Polars.")

        # Opción: excluir nulos antes del conteo
        if dropna:
            working_df = df.drop_nulls(subset=[col_name])
        else:
            working_df = df

        total_count = working_df.height()

        # Obtener frecuencia
        freq_df = (
            working_df
            .group_by(col_name)
            .agg(
                pl.count().alias("Frequency (units)")
            )
            .with_columns(
                (pl.col("Frequency (units)") / total_count * 100)
                .alias("Frequency (%)")
            )
        )

        return freq_df
    return


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
def _(df_raw):
    df_raw['HB010','HB020','HB030','HB050','HB050_F','HB060','HB060_F','HB070','HB070_F','HB080','HB080_F','HB100','HB100_F',
        'HB120','HB120_F']
    return


@app.cell
def _(df_raw, pl):
    df_step1 = df_raw.select(
        pl.col("HB120").cast(pl.Int8).alias("num_household_members")
    )
    df_step1
    return (df_step1,)


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
def _(df_raw, df_step1, pl):
    df_step2 = df_step1.with_columns(
        df_raw['HY023'].str.strip_chars().replace("","0").cast(pl.Float32).alias('net_disposable_income')
    )
    df_step2
    return (df_step2,)


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
    df_raw['HY030N','HY030N_F','HY040N','HY040N_F','HY050N','HY050N_F','HY060N','HY060N_F','HY070N','HY070N_F','HY080N','HY080N_F', 
        'HY081N','HY081N_F','HY090N','HY090N_F','HY100N','HY100N_F','HY110N','HY110N_F','HY120N','HY120N_F','HY130N','HY130N_F',
        'HY131N','HY131N_F','HY145N','HY145N_F','HY170N','HY170N_F']
    return


@app.cell
def _(df_raw, df_step2, pl):
    df_step3 = df_step2.with_columns(
        df_raw['HY120N'].str.strip_chars().replace("","0").cast(pl.Float32).alias('income_tax'),
        df_raw['HY040N'].str.strip_chars().replace("","0").cast(pl.Float32).alias('net_income_from_renting')
    )
    df_step3
    return (df_step3,)


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
def _(df_raw, df_step3, pl):
    df_step4 = df_step3.with_columns(
        df_raw['HY010'].str.strip_chars().replace("","0").cast(pl.Float32).alias('total_gross_household_income'), 
        df_raw['HY140G'].str.strip_chars().replace("","0").cast(pl.Float32).alias('income_tax_social_security'), 
    )
    df_step4
    return (df_step4,)


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
    df_raw['HS011', 'HS011_F', 'HS021','HS021_F','HS022','HS022_F','HS031','HS031_F','HS040','HS040_F','HS050','HS050_F',
        'HS060','HS060_F','HS090','HS090_F','HS110','HS110_F', 'HS120','HS120_F','HS150','HS150_F']
    return


@app.cell
def _(df_raw, df_step4, pl):
    T_2SiNo_code={
        "1": 1,
        "2": 2,
        "3": 0,
        "": None
    }

    T_2SiNo_code_from_i64={
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

    df_step5 = df_step4.with_columns(
        # df_raw['HS011'].str.strip_chars().replace("","0").cast(pl.Float32)    
        df_raw['HS011'].str.strip_chars().replace_strict(T_2SiNo_code).cast(pl.Int8).alias('delay_mortgage_rent'),
        df_raw['HS021'].replace_strict(T_2SiNo_code_from_i64).cast(pl.Int8).alias('delay_bills'),
        df_raw['HS040'].replace_strict(T_SiNo_code).cast(pl.Boolean).alias('afferd_week_vacation_out'),
        df_raw['HS050'].replace_strict(T_SiNo_code).cast(pl.Boolean).alias('afford_expensive_meal'),
    
    )
    df_step5
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


app._unparsable_cell(
    r"""
    TH010H_code = {
        1: 'Detached single-family home',
        2: 'Semi-detached or attached single-family home',
        3: 'Flat or apartment in a building with fewer than 10 dwellings',
        4: 'Flat or apartment in a building with 10 or more dwellings',
        None: None
    }

    TH021H_code={
        1: 'En propiedad sin hipoteca',
        2: 'En propiedad con hipoteca'
        3: 'En alquiler o realquiler a precio de mercado'
        4: 'En alquiler o realquiler a precio inferior al de mercado'
        5: 'En cesion gratuita'
    }

    df_final = df_raw.select(
        pl.col('HB120').cast(pl.Int8).alias('num_household_members'),
        pl.col('HY010').str.strip_chars().replace(\"\", \"0\").cast(pl.Float32).alias('total_gross_houshold_income'),
        pl.col('HY023').str.strip_chars().replace(\"\", \"0\").cast(pl.Float32).alias('net_disposable_income'),
        pl.col('HY030N').str.strip_chars().replace(\"\", \"0\").cast(pl.Float32).alias('net_imputed_rent'),
        pl.col('HY140G').str.strip_chars().replace(\"\", \"0\").cast(pl.Float32).alias('income_tax'),
        pl.col('HY120G').str.strip_chars().replace(\"\", \"0\").cast(pl.Float32).alias('wealth_tax'),
        pl.col('')
        pl.col('HH010').replace_strict(TH010H_code).alias(\"housing_type\")
    )
    df_final
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md(r"""
    # 2. Study case 1:
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
