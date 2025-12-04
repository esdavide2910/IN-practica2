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

    import sklearn
    import hdbscan

    import time

    import matplotlib.pyplot as plt
    return hdbscan, pl, plt, sklearn, time


@app.cell
def _():
    SEED=42
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
    ## 1.1. Data Loading
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
def _(mo):
    mo.md(r"""
    # 2. Study case I:
    """)
    return


@app.cell
def _(df_raw, pl):
    df_caseI_raw = df_raw.select(
        pl.when(pl.col('HY020_F') < 0).then(None).otherwise(pl.col('HY020').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY020'),
        pl.col('HX240'),
        pl.when(pl.col('HY040N_F') % 10 == 9).then(None).otherwise(pl.col('HY040N').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HY040N'),
        pl.when(pl.col('HH060_F') < 0).then(0).otherwise(pl.col('HH060').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HH060'),
        pl.when(pl.col('HH070_F') < 0).then(None).otherwise(pl.col('HH070').str.strip_chars().replace("","0").cast(pl.Float32)).alias('HH070'),
    )

    df_caseI_raw
    return (df_caseI_raw,)


@app.cell
def _(df_caseI_raw, pl):
    df_caseI = df_caseI_raw.select(
        (pl.col('HY020')/pl.col('HX240')).cast(pl.Float32).alias('equivalised_disposable_income'),
        pl.col('HY040N').alias('net_income_from_renting'),
        pl.col('HH060').alias('occupied_dwelling_rent'),
        pl.col('HH070').alias('housing_expenses')
    ).drop_nulls()

    df_caseI
    return (df_caseI,)


@app.cell
def _(df_caseI, sklearn):
    from sklearn.preprocessing import PowerTransformer

    transformer_caseI = sklearn.compose.ColumnTransformer(
        transformers=[
            ('equivalised_disposable_income', PowerTransformer(method='yeo-johnson'), ['equivalised_disposable_income']),
            ('net_income_from_renting', PowerTransformer(method='yeo-johnson'), ['net_income_from_renting']),
            ('occupied_dwelling_rent', PowerTransformer(method='yeo-johnson'), ['occupied_dwelling_rent']),
            ('housing_expenses', PowerTransformer(method='yeo-johnson'), ['housing_expenses']),
        ],
        verbose_feature_names_out=False
    )

    transformer_caseI.set_output(transform="polars")

    df_caseI_norm = transformer_caseI.fit_transform(df_caseI)
    df_caseI_norm
    return (df_caseI_norm,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.1. KMeans
    """)
    return


@app.cell
def _(SEED, df_caseI_norm, pl, sklearn, time):
    range_k_clusters = range(2,10)

    results_kmeans = {'n_clusters':[], 'inertia': [], 'silhouette_score': [], 'training_time': []}

    for k in range_k_clusters:

        kmeans = sklearn.cluster.KMeans(n_clusters=k, init='k-means++', n_init=10*k, random_state=SEED)

        start_time = time.time()
        cluster_labels = kmeans.fit_predict(df_caseI_norm)
        training_time = time.time()-start_time

        results_kmeans['n_clusters'].append(k)
        results_kmeans['inertia'].append(kmeans.inertia_)
        results_kmeans['silhouette_score'].append(sklearn.metrics.silhouette_score(df_caseI_norm, cluster_labels))
        results_kmeans['training_time'].append(training_time)

    results_kmeans = pl.DataFrame(results_kmeans)
    results_kmeans
    return k, range_k_clusters, results_kmeans


@app.cell
def _(plt, range_k_clusters, results_kmeans):
    # Grafica los resultados
    plt.figure(figsize=(12.5, 6.25))

    # Subplot para Inercia
    plt.subplot(1, 2, 1)
    plt.plot(range_k_clusters, results_kmeans['inertia'], marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia')
    plt.title('Inercia para Diferentes Números de Clusters')
    plt.grid(True)
    plt.xticks(range(2, 10))

    # Subplot para Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(range_k_clusters, results_kmeans['silhouette_score'], marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score para Diferentes Números de Clusters')
    plt.grid(True)
    plt.xticks(range(2, 10))

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(SEED, df_caseI_norm, k, results_kmeans, sklearn):
    best_k = results_kmeans.sort('silhouette_score', descending=True).select('n_clusters')[0].item()
    e_kmeans = sklearn.cluster.KMeans(n_clusters=best_k, init='k-means++', n_init=100*k, random_state=SEED)
    cluster_labels_kmeans = e_kmeans.fit_predict(df_caseI_norm)
    return best_k, cluster_labels_kmeans


@app.cell
def _():
    return


@app.cell
def _(cluster_labels_kmeans, df_caseI_norm, plot_silhouette):
    plot_silhouette(df_caseI_norm, cluster_labels_kmeans)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.2. HDBSCAN
    """)
    return


@app.cell
def _(df_caseI_norm, hdbscan, pl, sklearn, time):
    range_min_cluster_size = range(5,105,5)

    results_hdbscan = {'min_cluster_size':[], 'inertia': [], 'silhouette_score': [], 'training_time': []}

    for min_cluster_size in range_min_cluster_size:

        hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

        start_time_1 = time.time()
        hdb.fit(df_caseI_norm)
        hdbscan_labels = hdb.labels_
        training_time_1 = time.time()-start_time_1

        results_hdbscan['min_cluster_size'].append(min_cluster_size)
        results_hdbscan['silhouette_score'].append(sklearn.metrics.silhouette_score(df_caseI_norm, hdbscan_labels))
        results_hdbscan['training_time'].append(training_time_1)

    results_hdbscan = pl.DataFrame(results_hdbscan)
    results_hdbscan
    return range_min_cluster_size, results_hdbscan


@app.cell
def _(plt, range_min_cluster_size, results_hdbscan):
    # Grafica los resultados
    plt.figure(figsize=(6.5, 6.25))

    # Subplot para Silhouette Score
    plt.plot(range_min_cluster_size, results_hdbscan['silhouette_score'], marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score para Diferentes Números de Clusters')
    plt.grid(True)
    plt.xticks(range(5,105,5))

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(SEED, best_k, df_caseI_norm, hdbscan, k, results_hdbscan, sklearn):
    best_min_cluster_size = results_hdbscan.sort('silhouette_score', descending=True).select('n_clusters')[0].item()

    hdb = hdbscan.HDBSCAN(min_cluster_size=best_min_cluster_size)
    hdb.fit(df_caseI_norm)
    cluster_labels_hdbscan = hdb.labels_

    e_kmeans = sklearn.cluster.KMeans(n_clusters=best_k, init='k-means++', n_init=100*k, random_state=SEED)
    cluster_labels_kmeans = e_kmeans.fit_predict(df_caseI_norm)
    return (cluster_labels_kmeans,)


if __name__ == "__main__":
    app.run()
