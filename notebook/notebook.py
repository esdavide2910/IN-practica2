import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 0. Software usado
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    # Obtiene la ruta del directorio 'notebook/'
    notebook_dir = mo.notebook_dir()

    # Obtiene la ruta del directorio 'src/'
    src_dir = notebook_dir.parent / "src"

    # Obtiene la ruta del directorio 'datasets/'
    datasets_dir = notebook_dir.parent / "datasets"

    # Añade este directorio a la ruta de búsqueda de Python
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
def _(mo):
    mo.md(r"""
    ## 0.2 Funciones propias
    """)
    return


@app.cell
def _(pl):
    def analyze_missing_values(df: pl.DataFrame, show_all=False) -> pl.DataFrame:
        """
        Analiza los valores faltantes en un DataFrame de Polars y devuelve un resumen.

        Parameters:
          df (pl.DataFrame): El DataFrame a analizar.
          show_all (bool): Si True, incluye columnas sin valores faltantes. Por defecto es False.

        Returns:
          pd.DataFrame: DataFrame con dos filas por columna:
                        - Número de valores faltantes.
                        - Porcentaje de valores faltantes.
        """

        # Verificar que el input sea un DataFrame
        if not isinstance(df, pl.DataFrame):
            raise TypeError("El parámetro 'df' debe ser un DataFrame de Polars.")

        # Número de valores totales por columna
        total_rows = df.height

        # Calcular valores faltantes por columna
        missing_counts = df.null_count()

        # Convertir a Series -> DataFrame
        missing_df = (
            missing_counts
            .transpose(include_header=True)
            .melt(variable_name="column", value_name="Missing values (units)")
        )

        # Calcular porcentaje
        missing_df = missing_df.with_columns(
            (pl.col("Missing values (units)") / total_rows * 100)
            .alias("Ratio missing vals. (%)")
        )

        # Filtrar
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
    # 1. Introducción
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.1 Carga de datos
    """)
    return


@app.cell
def _(datasets_dir, pl):
    # Carga en un dataframe los datos del archivo "ECV"
    df_raw = pl.read_csv(datasets_dir / "ECV.csv", separator=',')
    # Muestra el dataframe
    df_raw
    return (df_raw,)


@app.cell
def _(df_raw):
    # Muestra información de las columnas del dataframe
    df_raw.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""
    **VIVIENDA**
    - TH010H: Tipo de vivienda.
    - TH021H: Régimen de tenencia.
    - TH030H: Número de habitaciones de la vivienda.
    - HH050: ¿Puede el hogar permitirse mantener la vivienda con una temperatura adecuada durante los meses de invierno?
    - HH060: Alquiler actual por la vivienda ocupada.
    - HH070: Gastos de la vivienda: Alquiler (si la vivienda se encuentra en régimen de alquiler), intereses de la hipoteca (para viviendas en propiedad con pagos pendientes) y otros gastos asociados (comunidad, agua, electricidad, gas, etc.).

    **ENDEUDAMIENTO EXCESIVO**


    **INFANCIA**



    **ACCESO A SERVICIOS: CUIDADO DE NIÑOS**
    - HC040: ¿Cómo puede permitirse el hogar el gasto por asistencia a centros de cuidado de niños?
    - HC040B: ¿Cómo puede permitirse el hogar el gasto por asistencia a centros de cuidado de niños? Se incluyen los niños de 0 a 3 años

    **ACCESO A SERVICIOS: CUIDADO DE PERSONAS DEPENDIENTES**
    - HC190: ¿Vive en su hogar alguna persona que necesite ayuda por ser mayor o por tener una dolencia crónica?
    - HC200: ¿Reciben cuidados a domicilio por parte de un cuidador remunerado?
    - HC221: ¿Quién paga el coste de estos cuidados?
    - HC230: ¿Cómo puede permitirse el hogar el pago de estos cuidados?
    - HC240: ¿Necesita algún miembro del hogar cuidados o más cuidados a domicilio por parte de un cuidador remunerado?
    - HC250: Razón principal por la que no los recibe

    **ACCESO A SERVICIOS: TRANSPORTE PÚBLICO**
    - HC300: Carga para el hogar del coste del transporte público.
    """)
    return


@app.cell
def _(df_raw):
    print(f"Número de filas duplicadas: {df_raw.is_duplicated().sum()}")
    return


@app.cell
def _(df_raw):
    {c: df_raw[c].unique() for c in df_raw.columns}
    return


@app.cell
def _(df_raw):
    constant_columns = [c for c in df_raw.columns if df_raw[c].unique().shape[0] == 1]
    print(f"Columnas con valor constante: {constant_columns}")
    indexlike_columns = [c for c in df_raw.columns if df_raw[c].unique().shape[0] == df_raw.height]
    print(f"Columnas con valores distintos en cada fila: {indexlike_columns}")
    return constant_columns, indexlike_columns


@app.cell
def _(constant_columns, df_raw, indexlike_columns):
    df_step1 = df_raw.drop(constant_columns, indexlike_columns)
    df_step1
    return


@app.cell
def _():
    #df_final = df_raw ...
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 2. Caso de estudio
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.1. Estudio de la información básica.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **INFORMACIÓN BÁSICA**
    - HB010: Año de la encuesta.
    - HB020: País.
    - HB030: Identificación transversal del hogar.
    - HB050: Mes de la entrevista al hogar.
    - HB060: Año de la entrevista al hogar.
    - HB070: Identificación personal del informante.
    - HB080: Identificación de la primera persona responsable de la vivienda.
    - HB0100: Número de minutos empleados en cumplimentar el cuestionario sobre el hogar.
    - HB0120: Número de miembros del hogar.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.2. Estudio de la renta.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **RENTA: GENERAL**
    - HY020: Renta disponible total del hogar en el año anterior al de encuesta (nota: esta variable contiene las rentas percibidas de esquemas privados de pensiones).
    - HY022: Renta disponible total del hogar antes de transferencias sociales excepto prestaciones por jubilación y por supervivencia en el año anterior al de encuesta.
    - HY023: Renta disponible total del hogar antes de transferencias sociales incluidas prestaciones por jubilación y por supervivencia en el año anterior al de encuesta.

    **RENTA: SERIE DE VARIABLES DE RENTA NETAS**
    - HY030N: Alquiler imputado (El alquiler imputado se aplica a los hogares que no pagan un alquiler completo por ser propietarios o por ocupar una vivienda alquilada a un precio inferior al de mercado o a título gratuito. El valor que se imputa es el equivalente al alquiler que se pagaría en el mercado por una vivienda similar a la ocupada, menos cualquier alquiler realmente abonado).
    - HY040N: Renta neta procedente del alquiler de una propiedad o terreno en el año anterior al de encuesta.
    - HY050N: Ayuda por familia/hijos en el año anterior al de la encuesta.
    - HY060N: Ingresos por asistencia social en el año anterior al de encuesta
    - HY070N: Ayuda para vivienda en el año anterior al de encuesta
    - HY080N: Transferencias periódicas monetarias percibidas de otros hogares en el año anterior al de encuesta
    - HY081N: Transferencias periódicas monetarias percibidas de otros hogares en el año anterior al de encuesta (pensiones alimenticias a hijos o compensatorias a cónyuges)
    - HY090N: Intereses, dividendos y ganancias netos de inversiones de capital en empresas no constituidas en sociedad en el año anterior al de encuesta
    - HY100N: Intereses pagados del préstamo para la compra de la vivienda principal, en el año anterior al de encuesta
    - HY110N: Renta neta percibida por los menores de 16 años en el año anterior al de encuesta
    - HY120N: Impuesto sobre el patrimonio en el año anterior al de la encuesta.
    - HY130N: Transferencias periódicas monetarias abonadas a otros hogares en el año anterior al de encuesta.
    - HY131N: Transferencias periódicas monetarias abonadas a otros hogares en el año anterior al de encuesta (pensiones alimenticias a hijos o compensatorias a cónyuges).
    - HY145N: Devoluciones/ingresos complementarios por ajustes En impuestos en el año anterior al de encuesta (declaración del IRPF).
    - HY170N: Autoconsumo en el año anterior al de encuesta.

    **RENTA: SERIE DE VARIABLES DE RENTA BRUTAS**
    - HY010: Renta bruta total del hogar en el año anterior al de encuesta.
    - HY040G: Renta bruta procedente del alquiler de una propiedad o terreno en el año anterior al de encuesta.
    - HY050G: Ayuda por familia/hijos en el año anterior al de encuesta.
    - HY060G: Ingresos por asistencia social en el año anterior al de encuesta.
    - HY070G: Ayuda para vivienda en el año anterior al de encuesta.
    - HY080G: Transferencias periódicas monetarias percibidas de otros hogares en el año anterior al de encuesta.
    - HY081G: Transferencias periódicas monetarias percibidas de otros hogares en el año anterior al de encuesta (pensiones alimenticias a hijos o compensatorias a cónyuges).
    - HY090G: Intereses, dividendos y ganancias brutos de inversiones de capital en empresas no constituidas en sociedad en el año anterior al de encuesta.
    - HY100G: Intereses pagados del préstamo para la compra de la vivienda principal, en el año anterior al de encuesta.
    - HY110G: Renta bruta percibida por los menores de 16 años en el año anterior al de encuesta.
    - HY120G: Impuesto sobre el patrimonio en el año anterior al de encuesta. A partir de ECV2021 se incluye en esta variable el Impuesto sobre Bienes Inmuebles de la vivienda principal, cuando el régimen de tenencia sea en propiedad.
    - HY130G: Transferencias periódicas monetarias abonadas a otros hogares en el año anterior al de encuesta.
    - HY131G: Transferencias periódicas monetarias abonadas a otros hogares en el año anterior al de encuesta (pensiones alimenticias a hijos o compensatorias a cónyuges).
    - HY140G: Impuesto sobre la renta y cotizaciones sociales.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
