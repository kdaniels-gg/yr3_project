import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
import umap
import hdbscan
import os 


OUT_DIR = Path('clustering_results')
OUT_DIR.mkdir(exist_ok=True)


data_dir = Path(os.path.join(os.getcwd(),'plasmid_motif_network/intermediate'))
files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))

df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()

df_merged = df_merged.with_columns(
    pl.col('strand').cast(pl.Int32).alias('strand')
)

plasmid_domains = (
    df_merged.group_by('plasmid')
    .agg(pl.col('target_name').unique().alias('domains'))
    .sort('plasmid')
)
plasmid_list = plasmid_domains['plasmid'].to_list()
domain_lists = plasmid_domains['domains'].to_list()

mlb = MultiLabelBinarizer(sparse_output=True)
X = mlb.fit_transform(domain_lists).astype(np.float32)
print(f'matrix {X.shape[0]} plasmids x {X.shape[1]} domains')


embedding = umap.UMAP(
    metric='jaccard',
    n_neighbors=10,
    min_dist=0.05,
    n_components=2,
    random_state=42,
    low_memory=True,
    verbose=True
).fit_transform(X)

mask = ~np.isnan(embedding).any(axis=1)
embedding_clean = embedding[mask]
plasmids_clean = np.array(plasmid_list)[mask]

labels = hdbscan.HDBSCAN(
    min_cluster_size=30,
    min_samples=5,
    cluster_selection_method='leaf'
).fit_predict(embedding_clean)


result_df = pd.DataFrame({
    'plasmid': plasmids_clean,
    'cluster': labels,
    'umap_x':  embedding_clean[:, 0],
    'umap_y':  embedding_clean[:, 1],
})

print(result_df['cluster'].value_counts().sort_index())
result_df.to_csv(OUT_DIR / 'umap_hdbscan_clusters.csv', index=False)
