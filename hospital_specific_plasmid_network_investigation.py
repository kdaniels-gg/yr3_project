import os
import sys
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import networkx as nx
import matplotlib.pyplot as plt


RECOMB_THRESHOLD = 3
N_NULL = 100

def _eb(ax, x, y, ye, **kw):
    ax.plot(x, y, **kw)
    if np.any(np.array(ye) > 0):
        ax.fill_between(x, np.array(y) - np.array(ye),
                        np.array(y) + np.array(ye),
                        alpha=0.2, color=kw.get('color', 'blue'))


def _std_arr(df, col):
    c = col + '_std'
    return df[c].to_numpy() if c in df.columns else np.zeros(len(df))


def find_unbranching_paths(G):
    succ  = {n: list(set(G.successors(n)))   for n in G.nodes()}
    pred  = {n: list(set(G.predecessors(n))) for n in G.nodes()}
    in_d  = {n: len(pred[n]) for n in G.nodes()}
    out_d = {n: len(succ[n]) for n in G.nodes()}
    simple  = {n for n in G.nodes() if in_d[n] == 1 and out_d[n] == 1}
    paths   = []
    visited = set()
    for start in G.nodes():
        if start in simple:
            continue
        for nxt in succ[start]:
            if nxt not in simple or nxt in visited:
                continue
            path = [start, nxt]
            visited.add(nxt)
            cur = nxt
            while True:
                nxt2 = succ[cur][0]
                path.append(nxt2)
                if nxt2 not in simple or nxt2 in visited:
                    break
                visited.add(nxt2)
                cur = nxt2
            paths.append(path)
    return paths


def path_stats_from_graph(G):
    paths   = find_unbranching_paths(G)
    n_nodes = G.number_of_nodes()
    if paths:
        lengths = [len(p) for p in paths]
        return {
            'n_paths':  len(paths),
            'max_len':  max(lengths),
            'mean_len': float(np.mean(lengths)),
            'coverage': len(set(n for p in paths for n in p)) / n_nodes if n_nodes > 0 else 0.0,
        }
    return {'n_paths': 0, 'max_len': 0, 'mean_len': 0.0, 'coverage': 0.0}


def recomb_null_graph(G_obs, threshold=RECOMB_THRESHOLD, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    nodes = list(G_obs.nodes())
    n = len(nodes)
    if n < 2:
        return nx.DiGraph()
    budget = sum(max(0, d - 2) for _, d in G_obs.degree())
    H = nx.DiGraph()
    H.add_nodes_from(nodes)
    for i in range(n):
        H.add_edge(nodes[i], nodes[(i + 1) % n])
    existing = set(H.edges())
    idx = list(range(n))
    while budget >= 2:
        rng.shuffle(idx)
        added = False
        for ui in idx:
            vi = int(rng.integers(n))
            if vi == ui:
                vi = (ui + 1) % n
            u, v = nodes[ui], nodes[vi]
            if (u, v) not in existing:
                H.add_edge(u, v)
                existing.add((u, v))
                budget -= 2
                added = True
                break
        if not added:
            break
    return H


def config_model_stats(G_obs, n_null=N_NULL, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    in_seq  = [d for _, d in G_obs.in_degree()]
    out_seq = [d for _, d in G_obs.out_degree()]
    n_nodes = G_obs.number_of_nodes()
    if n_nodes < 2 or sum(in_seq) == 0:
        return {k: 0 for k in ('null_edges_mean', 'null_edges_std',
                               'null_degree_mean', 'null_degree_std',
                               'null_density_mean', 'null_density_std')}
    null_edges, null_degrees, null_densities = [], [], []
    for _ in range(n_null):
        H = nx.directed_configuration_model(in_seq, out_seq,
                create_using=nx.DiGraph(), seed=int(rng.integers(1e9)))
        H.remove_edges_from(nx.selfloop_edges(H))
        e = H.number_of_edges()
        n = H.number_of_nodes()
        null_edges.append(e)
        null_degrees.append(2 * e / n if n > 0 else 0)
        max_e = n * (n - 1)
        null_densities.append(e / max_e if max_e > 0 else 0)
    return {
        'null_edges_mean':   float(np.mean(null_edges)),
        'null_edges_std':    float(np.std(null_edges)),
        'null_degree_mean':  float(np.mean(null_degrees)),
        'null_degree_std':   float(np.std(null_degrees)),
        'null_density_mean': float(np.mean(null_densities)),
        'null_density_std':  float(np.std(null_densities)),
    }


def recomb_model_stats(G_obs, n_null=N_NULL, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n_nodes = G_obs.number_of_nodes()
    if n_nodes < 2:
        return {k: 0 for k in ('recomb_edges_mean', 'recomb_edges_std',
                               'recomb_degree_mean', 'recomb_degree_std',
                               'recomb_density_mean', 'recomb_density_std')}
    null_edges, null_degrees, null_densities = [], [], []
    max_e = n_nodes * (n_nodes - 1)
    for _ in range(n_null):
        H = recomb_null_graph(G_obs, threshold=RECOMB_THRESHOLD, rng=rng)
        e = H.number_of_edges()
        null_edges.append(e)
        null_degrees.append(2 * e / n_nodes if n_nodes > 0 else 0)
        null_densities.append(e / max_e if max_e > 0 else 0)
    return {
        'recomb_edges_mean':   float(np.mean(null_edges)),
        'recomb_edges_std':    float(np.std(null_edges)),
        'recomb_degree_mean':  float(np.mean(null_degrees)),
        'recomb_degree_std':   float(np.std(null_degrees)),
        'recomb_density_mean': float(np.mean(null_densities)),
        'recomb_density_std':  float(np.std(null_densities)),
    }


def config_null_path_stats(G_obs, n_null=50, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    in_seq  = [d for _, d in G_obs.in_degree()]
    out_seq = [d for _, d in G_obs.out_degree()]
    n_nodes = G_obs.number_of_nodes()
    if n_nodes < 2 or sum(in_seq) == 0:
        return {k: (0.0, 0.0) for k in ('n_paths', 'max_len', 'mean_len', 'coverage')}
    results = []
    for _ in range(n_null):
        H = nx.directed_configuration_model(in_seq, out_seq,
                create_using=nx.DiGraph(), seed=int(rng.integers(1e9)))
        H.remove_edges_from(nx.selfloop_edges(H))
        results.append(path_stats_from_graph(H))
    out = {}
    for key in ('n_paths', 'max_len', 'mean_len', 'coverage'):
        vals = [r[key] for r in results]
        out[key] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def recomb_null_path_stats(G_obs, n_null=50, threshold=RECOMB_THRESHOLD, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n_nodes = G_obs.number_of_nodes()
    if n_nodes < 2:
        return {k: (0.0, 0.0) for k in ('n_paths', 'max_len', 'mean_len', 'coverage')}
    results = []
    for _ in range(n_null):
        H = recomb_null_graph(G_obs, threshold=threshold, rng=rng)
        results.append(path_stats_from_graph(H))
    out = {}
    for key in ('n_paths', 'max_len', 'mean_len', 'coverage'):
        vals = [r[key] for r in results]
        out[key] = (float(np.mean(vals)), float(np.std(vals)))
    return out


# ── graph builder (shared edge-construction logic) ───────────────────────────

def build_graphs_from_plasmids(plasmids, df_merged):
    """Return (G_multi, F_signed) from a list of plasmid accessions."""
    df_filt = df_merged.filter(pl.col('plasmid').is_in(plasmids))
    ordered = (df_filt
               .sort(['plasmid', 'start', 'ali_from'])
               .select(['plasmid', 'query_name', 'target_name', 'start', 'ali_from', 'strand']))
    df_shifted = ordered.with_columns([
        pl.col('target_name').shift(-1).over('plasmid').alias('domain2'),
        pl.col('strand').shift(-1).over('plasmid').alias('strand2'),
    ]).rename({'target_name': 'domain1', 'strand': 'strand1'})
    df_shifted = df_shifted.filter(pl.col('domain2').is_not_null())
    wrap = ordered.group_by('plasmid').agg([
        pl.col('target_name').last().alias('domain1'),
        pl.col('target_name').first().alias('domain2'),
        pl.col('strand').last().alias('strand1'),
        pl.col('strand').first().alias('strand2'),
    ])
    df_edges = pl.concat([
        df_shifted.select(['plasmid', 'domain1', 'domain2', 'strand1', 'strand2']),
        wrap.select(['plasmid', 'domain1', 'domain2', 'strand1', 'strand2']),
    ])
    df_edges = df_edges.with_columns(
        pl.when((pl.col('strand1') == 1)  & (pl.col('strand2') == 1)).then(pl.lit('PP'))
          .when((pl.col('strand1') == -1) & (pl.col('strand2') == -1)).then(pl.lit('MM'))
          .when((pl.col('strand1') == 1)  & (pl.col('strand2') == -1)).then(pl.lit('PM'))
          .otherwise(pl.lit('MP'))
          .alias('orientation')
    )
    adj_df = (df_edges
              .group_by(['domain1', 'domain2', 'orientation'])
              .agg(pl.len().cast(pl.Int64).alias('weight')))
    signed_contrib = adj_df.with_columns(
        pl.when(pl.col('orientation').is_in(['PP', 'MM']))
          .then(pl.col('weight'))
          .otherwise(-pl.col('weight'))
          .alias('signed_contribution')
    )
    collapsed = (signed_contrib
                 .group_by(['domain1', 'domain2'])
                 .agg([pl.sum('signed_contribution').alias('signed_weight'),
                       pl.sum('weight').alias('total_weight')]))
    domain_to_plasmids = defaultdict(set)
    for row in ordered.iter_rows(named=True):
        domain_to_plasmids[row['target_name']].add(row['plasmid'])
    # unsigned multi-di-graph
    G = nx.MultiDiGraph()
    for row in adj_df.iter_rows(named=True):
        G.add_edge(row['domain1'], row['domain2'],
                   weight=row['weight'], orientation=row['orientation'])
    for node in G.nodes():
        ps = domain_to_plasmids.get(node, set())
        G.nodes[node]['label'] = node
        G.nodes[node]['plasmid_count'] = len(ps)
        G.nodes[node]['plasmids'] = ';'.join(sorted(ps))
    for gid, (u, v, k) in enumerate(G.edges(keys=True)):
        G[u][v][k]['id'] = str(gid)
    # signed simple di-graph
    F = nx.DiGraph()
    for row in collapsed.iter_rows(named=True):
        F.add_edge(row['domain1'], row['domain2'],
                   signed_weight=row['signed_weight'],
                   total_weight=row['total_weight'])
    for node in F.nodes():
        ps = domain_to_plasmids.get(node, set())
        F.nodes[node]['label'] = node
        F.nodes[node]['plasmid_count'] = len(ps)
        F.nodes[node]['plasmids'] = ';'.join(sorted(ps))
    return G, F

cwd = Path(os.getcwd())
out_dir = cwd / 'hospital_analysis'
out_dir.mkdir(exist_ok=True)
gml_dir = out_dir / 'graphml'
gml_dir.mkdir(exist_ok=True)

#get motif data
data_dir = cwd / 'plasmid_motif_network' / 'intermediate'
files    = sorted(data_dir.glob('parsed_selected_nonoverlap_*.parquet'))
df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
df_merged = df_merged.with_columns(pl.col('strand').cast(pl.Int32))


#get hospital sample
plasmid_files_path = Path(os.path.join(os.getcwd(), 'plasmids'))

plasmid_files = os.listdir(plasmid_files_path)
plasmid_nuccore_list = [''.join(x.split('.fa')[:-1]) for x in plasmid_files]

plsdb_meta_path = Path(os.path.join(os.getcwd(), 'plsdb_meta'))
plsdb_meta_files = os.listdir(plsdb_meta_path)

nucpath = plsdb_meta_path / 'nuccore.csv'
nucout_path = plsdb_meta_path / f'nuccore_only.csv'
nuc_df = pd.read_csv(nucout_path)
nuc_bio = dict(zip(nuc_df['NUCCORE_ACC'].tolist(), nuc_df['BIOSAMPLE_UID'].tolist()))


biopath = plsdb_meta_path / 'biosample.csv'
bio_df = pd.read_csv(biopath, low_memory=False)

bio_loc = dict(zip(bio_df['BIOSAMPLE_UID'].tolist(), bio_df['LOCATION_name'].tolist()))
bio_lat = dict(zip(bio_df['BIOSAMPLE_UID'].tolist(), bio_df['LOCATION_lat'].tolist()))
bio_lng = dict(zip(bio_df['BIOSAMPLE_UID'].tolist(), bio_df['LOCATION_lng'].tolist()))


nuc_loc = {k:bio_loc.get(v) for k,v in nuc_bio.items()}
nuc_lat = {k:bio_lat.get(v) for k,v in nuc_bio.items()}
nuc_lng = {k:bio_lng.get(v) for k,v in nuc_bio.items()}


nuc_loc_down = {k:v for k,v in nuc_loc.items() if type(v) == str}
nuc_loc_hosp = {k:v for k,v in nuc_loc_down.items() if 'ospital' in v}
nuc_loc_hosp_vals = list(set(list(nuc_loc_hosp.values())))


hospital_plasmids = {}
for val in nuc_loc_hosp_vals:
    v_plas = [k for k,v in nuc_loc_hosp.items() if v == val]
    hospital_plasmids[val] = v_plas


hospital_plasmids = dict(sorted(hospital_plasmids.items(), key=lambda x: len(x[1]), reverse=True))
print(f'Max number of plasmids from single hospital "{list(hospital_plasmids.keys())[0]}": \n{len(hospital_plasmids[list(hospital_plasmids.keys())[0]])}\n')

selected_hospital_plasmid_sample = hospital_plasmids[list(hospital_plasmids.keys())[0]]


label = 'hospital'
print(f'Hospital plasmids: {len(selected_hospital_plasmid_sample)}')
# ── build batch series (geomspace over 1..55, 10 runs) ───────────────────
all_run_stats_F, all_run_stats_G = [], []
max_size    = len(selected_hospital_plasmid_sample)
num_batches = 50
batch_sizes = np.unique(np.geomspace(1, max_size, num=num_batches, dtype=int))
print('Building batch graphs …')
sys.stdout.flush()
for run in range(1, 11):
    plasmids_shuf = selected_hospital_plasmid_sample.copy()
    random.shuffle(plasmids_shuf)
    F_run, G_run = [], []
    for size in batch_sizes:
        batch = plasmids_shuf[:size]
        G_multi, F_signed = build_graphs_from_plasmids(batch, df_merged)
        # save graphml only for the last run (matches chud_species.py)
        if run == 10:
            nx.write_graphml(
                G_multi,
                str(gml_dir / f'batch_{size}_{label}_domain_architecture_network.graphml'),
                edge_id_from_attribute='id')
            nx.write_graphml(
                F_signed,
                str(gml_dir / f'batch_{size}_{label}_domain_architecture_signed_network.graphml'))
        fn = F_signed.number_of_nodes()
        fe = F_signed.number_of_edges()
        ge = G_multi.number_of_edges()
        F_run.append({'plasmid_number': size,
                      'node_number':   fn,
                      'edge_number':   fe,
                      'average_degree': 2 * fe / fn if fn > 0 else 0.0})
        G_run.append({'plasmid_number': size,
                      'node_number':   fn,
                      'edge_number':   ge,
                      'average_degree': 2 * ge / fn if fn > 0 else 0.0})
    all_run_stats_F.append(F_run)
    all_run_stats_G.append(G_run)
    print(f'  run {run}/10 done')
    sys.stdout.flush()



# average stats across runs
Fdf_all = pl.DataFrame([r for run in all_run_stats_F for r in run])
avg_F   = Fdf_all.group_by('plasmid_number').mean().sort('plasmid_number')
std_F   = (Fdf_all.group_by('plasmid_number')
                  .agg([pl.std('edge_number').alias('edge_number_std'),
                        pl.std('node_number').alias('node_number_std'),
                        pl.std('average_degree').alias('average_degree_std')])
                  .sort('plasmid_number'))
Gdf_all = pl.DataFrame([r for run in all_run_stats_G for r in run])
avg_G   = Gdf_all.group_by('plasmid_number').mean().sort('plasmid_number')
std_G   = (Gdf_all.group_by('plasmid_number')
                  .agg([pl.std('edge_number').alias('edge_number_std'),
                        pl.std('node_number').alias('node_number_std'),
                        pl.std('average_degree').alias('average_degree_std')])
                  .sort('plasmid_number'))
F_stats = avg_F.join(std_F, on='plasmid_number', how='left')
G_stats = avg_G.join(std_G, on='plasmid_number', how='left')
F_stats.write_csv(out_dir / 'F_graph_statistics.csv')
G_stats.write_csv(out_dir / 'G_graph_statistics.csv')
print('Graph statistics saved.')



# ── edges / degree / density plots ───────────────────────────────────────
fdf = pd.read_csv(out_dir / 'F_graph_statistics.csv')
gdf = pd.read_csv(out_dir / 'G_graph_statistics.csv')
batch_x   = fdf['plasmid_number'].tolist()
f_nodes   = fdf['node_number'].tolist()
f_edges   = fdf['edge_number'].tolist()
g_edges   = gdf['edge_number'].tolist()
f_edges_std = fdf.get('edge_number_std', pd.Series([0]*len(f_edges))).fillna(0).tolist()
g_edges_std = gdf.get('edge_number_std', pd.Series([0]*len(g_edges))).fillna(0).tolist()
complete_edges  = [n*(n-1) for n in f_nodes]
f_density = [o/c if c > 0 else 0 for o, c in zip(f_edges, complete_edges)]
g_density = [o/c if c > 0 else 0 for o, c in zip(g_edges, complete_edges)]
f_degree  = [2*e/n if n > 0 else 0 for e, n in zip(f_edges, f_nodes)]
g_degree  = [2*e/n if n > 0 else 0 for e, n in zip(g_edges, f_nodes)]
f_density_std = [s/c if c > 0 else 0 for s, c in zip(f_edges_std, complete_edges)]
g_density_std = [s/c if c > 0 else 0 for s, c in zip(g_edges_std, complete_edges)]
f_degree_std  = [2*s/n if n > 0 else 0 for s, n in zip(f_edges_std, f_nodes)]
g_degree_std  = [2*s/n if n > 0 else 0 for s, n in zip(g_edges_std, f_nodes)]
batch_arr = np.array(batch_x)
print('Computing config-model & recombination null stats …')
sys.stdout.flush()
null_rows, recomb_rows = [], []
_rng_null  = np.random.default_rng(42)
_rng_recomb = np.random.default_rng(7)
signed_gml_files = sorted(
    gml_dir.glob(f'batch_*_{label}_domain_architecture_signed_network.graphml'),
    key=lambda p: int(p.name.split('_')[1]))
for gml_path in signed_gml_files:
    bn     = int(gml_path.name.split('_')[1])
    G_obs  = nx.read_graphml(str(gml_path))
    st_cfg = config_model_stats(G_obs, n_null=N_NULL, rng=_rng_null)
    st_cfg['plasmid_number'] = bn
    null_rows.append(st_cfg)
    st_rec = recomb_model_stats(G_obs, n_null=N_NULL, rng=_rng_recomb)
    st_rec['plasmid_number'] = bn
    recomb_rows.append(st_rec)
null_df   = pd.DataFrame(null_rows).sort_values('plasmid_number')
recomb_df = pd.DataFrame(recomb_rows).sort_values('plasmid_number')
null_df.to_csv(out_dir / 'null_model_statistics.csv', index=False)
recomb_df.to_csv(out_dir / 'recomb_null_model_statistics.csv', index=False)
ref = pd.DataFrame({'plasmid_number': batch_x})
nm  = ref.merge(null_df,   on='plasmid_number', how='left').fillna(0)
rm  = ref.merge(recomb_df, on='plasmid_number', how='left').fillna(0)
null_edges_mean   = nm['null_edges_mean'].tolist()
null_edges_std    = nm['null_edges_std'].tolist()
null_degree_mean  = nm['null_degree_mean'].tolist()
null_degree_std   = nm['null_degree_std'].tolist()
null_density_mean = nm['null_density_mean'].tolist()
null_density_std  = nm['null_density_std'].tolist()
r_edges_m  = rm['recomb_edges_mean'].tolist()
r_edges_s  = rm['recomb_edges_std'].tolist()
r_deg_m    = rm['recomb_degree_mean'].tolist()
r_deg_s    = rm['recomb_degree_std'].tolist()
r_dens_m   = rm['recomb_density_mean'].tolist()
r_dens_s   = rm['recomb_density_std'].tolist()
# edges plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
fig.suptitle(f'No. Edges / No. plasmids in batch (Duke University Hospital)')
_eb(ax1, batch_arr, np.array(f_edges), np.array(f_edges_std),
    linestyle='--', label='data', color='blue')
_eb(ax1, batch_arr, np.array(null_edges_mean), np.array(null_edges_std),
    linestyle=':', label='config-model null', color='purple')
_eb(ax1, batch_arr, np.array(r_edges_m), np.array(r_edges_s),
    linestyle='-.', label='recombination null', color='seagreen')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('No. plasmids in batch'); ax1.set_ylabel('No. edges')
ax1.set_title('Signed architecture network'); ax1.legend()
_eb(ax2, batch_arr, np.array(g_edges), np.array(g_edges_std),
    linestyle='--', label='data', color='green')
_eb(ax2, batch_arr, np.array(null_edges_mean), np.array(null_edges_std),
    linestyle=':', label='config-model null', color='purple')
_eb(ax2, batch_arr, np.array(r_edges_m), np.array(r_edges_s),
    linestyle='-.', label='recombination null', color='seagreen')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('No. plasmids in batch'); ax2.set_ylabel('No. edges')
ax2.set_title('Unsigned architecture network'); ax2.legend()
plt.tight_layout()
plt.savefig(out_dir / 'hospital_edges_3.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved hospital_edges_3.png')
# degree plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
fig.suptitle(f'Avg. node degree / No. plasmids in batch (Duke University Hospital)')
_eb(ax1, batch_arr, np.array(f_degree), np.array(f_degree_std),
    linestyle='--', label='data', color='blue')
_eb(ax1, batch_arr, np.array(null_degree_mean), np.array(null_degree_std),
    linestyle=':', label='config-model null', color='purple')
_eb(ax1, batch_arr, np.array(r_deg_m), np.array(r_deg_s),
    linestyle='-.', label='recombination null', color='seagreen')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('No. plasmids in batch'); ax1.set_ylabel('Avg. node degree')
ax1.set_title('Signed architecture network'); ax1.legend()
_eb(ax2, batch_arr, np.array(g_degree), np.array(g_degree_std),
    linestyle='--', label='data', color='green')
_eb(ax2, batch_arr, np.array(null_degree_mean), np.array(null_degree_std),
    linestyle=':', label='config-model null', color='purple')
_eb(ax2, batch_arr, np.array(r_deg_m), np.array(r_deg_s),
    linestyle='-.', label='recombination null', color='seagreen')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('No. plasmids in batch'); ax2.set_ylabel('Avg. node degree')
ax2.set_title('Unsigned architecture network'); ax2.legend()
plt.tight_layout()
plt.savefig(out_dir / 'hospital_degrees_3.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved hospital_degrees_3.png')
# density plot
fig, ax1 = plt.subplots(figsize=(10, 12))
fig.suptitle(f'Density / No. plasmids in batch (Duke University Hospital)')
_eb(ax1, batch_arr, np.array(f_density), np.array(f_density_std),
    linestyle='--', label='signed data', color='blue')
_eb(ax1, batch_arr, np.array(g_density), np.array(g_density_std),
    linestyle=':', label='unsigned data', color='red')
_eb(ax1, batch_arr, np.array(null_density_mean), np.array(null_density_std),
    linestyle='-.', label='config-model null', color='purple')
_eb(ax1, batch_arr, np.array(r_dens_m), np.array(r_dens_s),
    linestyle=(0, (3, 1, 1, 1)), label='recombination null', color='seagreen')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('No. plasmids in batch'); ax1.set_ylabel('No. Edges / No. possible edges')
ax1.legend()
plt.tight_layout()
plt.savefig(out_dir / 'hospital_density_3.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved hospital_density_3.png')



# ── paths plot ────────────────────────────────────────────────────────────
print('Computing unbranching paths …')
sys.stdout.flush()
all_path_stats, path_csv_rows, null_path_rows = [], [], []
unsigned_gml_files = sorted(
    gml_dir.glob(f'batch_*_{label}_domain_architecture_network.graphml'),
    key=lambda p: int(p.name.split('_')[1]))
_rng_paths = np.random.default_rng(42)
for gml_path in unsigned_gml_files:
    batch_num = int(gml_path.name.split('_')[1])
    G = nx.read_graphml(str(gml_path))
    paths = find_unbranching_paths(G)
    if paths:
        lengths = [len(p) for p in paths]
        max_len, mean_len = max(lengths), sum(lengths)/len(lengths)
        unique_nodes = len(set(n for p in paths for n in p))
    else:
        lengths, max_len, mean_len, unique_nodes = [], 0, 0.0, 0
    n_nodes  = G.number_of_nodes()
    coverage = unique_nodes / n_nodes if n_nodes > 0 else 0.0
    all_path_stats.append({'batch': batch_num, 'n_nodes': n_nodes,
                            'n_paths': len(paths), 'max_path_len': max_len,
                            'mean_path_len': round(mean_len, 2),
                            'coverage': round(coverage, 4)})
    for i, path in enumerate(sorted(paths, key=len, reverse=True)):
        path_csv_rows.append({'batch': batch_num, 'path_index': i,
                              'length': len(path), 'path': ' -> '.join(path)})
    null_ps   = config_null_path_stats(G, n_null=50, rng=_rng_paths)
    recomb_ps = recomb_null_path_stats(G, n_null=50, rng=_rng_paths)
    null_path_rows.append({
        'batch': batch_num, 'n_nodes': n_nodes,
        'null_n_paths_mean':   null_ps['n_paths'][0],
        'null_n_paths_std':    null_ps['n_paths'][1],
        'null_max_len_mean':   null_ps['max_len'][0],
        'null_max_len_std':    null_ps['max_len'][1],
        'null_mean_len_mean':  null_ps['mean_len'][0],
        'null_mean_len_std':   null_ps['mean_len'][1],
        'null_cov_mean':       null_ps['coverage'][0] * 100,
        'null_cov_std':        null_ps['coverage'][1] * 100,
        'recomb_n_paths_mean': recomb_ps['n_paths'][0],
        'recomb_n_paths_std':  recomb_ps['n_paths'][1],
        'recomb_max_len_mean': recomb_ps['max_len'][0],
        'recomb_max_len_std':  recomb_ps['max_len'][1],
        'recomb_mean_len_mean':recomb_ps['mean_len'][0],
        'recomb_mean_len_std': recomb_ps['mean_len'][1],
        'recomb_cov_mean':     recomb_ps['coverage'][0] * 100,
        'recomb_cov_std':      recomb_ps['coverage'][1] * 100,
    })
pathstats_df = pd.DataFrame(all_path_stats)
null_path_df = (pd.DataFrame(null_path_rows)
                  .sort_values('batch')
                  .set_index('batch')
                  .reindex(pathstats_df['batch'])
                  .reset_index())
pd.DataFrame(path_csv_rows).to_csv(out_dir / 'unbranching_paths.csv', index=False)
pathstats_df.to_csv(out_dir / 'unbranching_paths_stats.csv', index=False)
pd.DataFrame(null_path_rows).to_csv(out_dir / 'null_path_statistics.csv', index=False)
pathstats_df['coverage_percentage'] = pathstats_df['coverage'] * 100
batch_n      = pathstats_df['batch'].to_numpy()
n_nodes2     = pathstats_df['n_nodes'].to_numpy()
n_paths      = pathstats_df['n_paths'].to_numpy()
max_pathlen  = pathstats_df['max_path_len'].to_numpy()
mean_pathlen = pathstats_df['mean_path_len'].to_numpy()
path_coverage = pathstats_df['coverage_percentage'].to_numpy()
n_paths_std      = _std_arr(pathstats_df, 'n_paths')
max_pathlen_std  = _std_arr(pathstats_df, 'max_path_len')
mean_pathlen_std = _std_arr(pathstats_df, 'mean_path_len')
coverage_std     = _std_arr(pathstats_df, 'coverage_percentage')
np_mean  = null_path_df['null_n_paths_mean'].to_numpy()
np_std   = null_path_df['null_n_paths_std'].to_numpy()
ml_mean  = null_path_df['null_max_len_mean'].to_numpy()
ml_std   = null_path_df['null_max_len_std'].to_numpy()
mnl_mean = null_path_df['null_mean_len_mean'].to_numpy()
mnl_std  = null_path_df['null_mean_len_std'].to_numpy()
cov_mean = null_path_df['null_cov_mean'].to_numpy()
cov_std  = null_path_df['null_cov_std'].to_numpy()
rp_mean   = null_path_df['recomb_n_paths_mean'].to_numpy()
rp_std    = null_path_df['recomb_n_paths_std'].to_numpy()
rml_mean  = null_path_df['recomb_max_len_mean'].to_numpy()
rml_std   = null_path_df['recomb_max_len_std'].to_numpy()
rmnl_mean = null_path_df['recomb_mean_len_mean'].to_numpy()
rmnl_std  = null_path_df['recomb_mean_len_std'].to_numpy()
rcov_mean = null_path_df['recomb_cov_mean'].to_numpy()
rcov_std  = null_path_df['recomb_cov_std'].to_numpy()
null_n_nodes = null_path_df['n_nodes'].to_numpy()
fig, axes = plt.subplots(7, 1, figsize=(10, 70))
ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes
fig.suptitle('Unbranching paths statistics — Duke University Hospital')
_eb(ax1, batch_n, n_paths, n_paths_std, linestyle='--', label='data', color='blue')
_eb(ax1, batch_n, np_mean, np_std, linestyle=':', label='config-model null', color='purple')
_eb(ax1, batch_n, rp_mean, rp_std, linestyle='-.', label='recombination null', color='seagreen')
ax1.set_xlabel('Plasmid No. in batch'); ax1.set_ylabel('No. unbranching paths')
ax1.set_title('No. paths with increasing batch size')
ax1.set_xscale('log'); ax1.set_yscale('log'); ax1.legend()
ratio_paths       = n_paths       / np.maximum(n_nodes2, 1)
ratio_paths_std   = n_paths_std   / np.maximum(n_nodes2, 1)
null_ratio_paths  = np_mean       / np.maximum(null_n_nodes, 1)
nrp_std           = np_std        / np.maximum(null_n_nodes, 1)
recomb_rp         = rp_mean       / np.maximum(null_n_nodes, 1)
recomb_rp_std     = rp_std        / np.maximum(null_n_nodes, 1)
_eb(ax2, batch_n, ratio_paths, ratio_paths_std, linestyle='--', label='data', color='blue')
_eb(ax2, batch_n, null_ratio_paths, nrp_std, linestyle=':', label='config-model null', color='purple')
_eb(ax2, batch_n, recomb_rp, recomb_rp_std, linestyle='-.', label='recombination null', color='seagreen')
ax2.set_xlabel('Plasmid No. in batch'); ax2.set_ylabel('No. paths / No. nodes')
ax2.set_title('No. paths relative to No. nodes'); ax2.set_xscale('log'); ax2.set_yscale('log'); ax2.legend()
_eb(ax3, batch_n, max_pathlen, max_pathlen_std, linestyle='--', label='data', color='blue')
_eb(ax3, batch_n, ml_mean, ml_std, linestyle=':', label='config-model null', color='purple')
_eb(ax3, batch_n, rml_mean, rml_std, linestyle='-.', label='recombination null', color='seagreen')
ax3.set_xlabel('Plasmid No. in batch'); ax3.set_ylabel('Max path length')
ax3.set_title('Max length of unbranching paths'); ax3.set_xscale('log'); ax3.set_yscale('log'); ax3.legend()
ratio_max        = max_pathlen     / np.maximum(n_nodes2, 1)
ratio_max_std    = max_pathlen_std / np.maximum(n_nodes2, 1)
null_ratio_max   = ml_mean         / np.maximum(null_n_nodes, 1)
null_ratio_max_s = ml_std          / np.maximum(null_n_nodes, 1)
recomb_rmax      = rml_mean        / np.maximum(null_n_nodes, 1)
recomb_rmax_std  = rml_std         / np.maximum(null_n_nodes, 1)
_eb(ax4, batch_n, ratio_max, ratio_max_std, linestyle='--', label='data', color='blue')
_eb(ax4, batch_n, null_ratio_max, null_ratio_max_s, linestyle=':', label='config-model null', color='purple')
_eb(ax4, batch_n, recomb_rmax, recomb_rmax_std, linestyle='-.', label='recombination null', color='seagreen')
ax4.set_xlabel('Plasmid No. in batch'); ax4.set_ylabel('Max length / No. nodes')
ax4.set_title('Max path length relative to No. nodes'); ax4.set_xscale('log'); ax4.set_yscale('log'); ax4.legend()
_eb(ax5, batch_n, mean_pathlen, mean_pathlen_std, linestyle='--', label='data', color='blue')
_eb(ax5, batch_n, mnl_mean, mnl_std, linestyle=':', label='config-model null', color='purple')
_eb(ax5, batch_n, rmnl_mean, rmnl_std, linestyle='-.', label='recombination null', color='seagreen')
ax5.set_xlabel('Plasmid No. in batch'); ax5.set_ylabel('Mean path length')
ax5.set_title('Mean length of unbranching paths'); ax5.set_xscale('log'); ax5.set_yscale('log'); ax5.legend()
ratio_mean        = mean_pathlen     / np.maximum(n_nodes2, 1)
ratio_mean_std    = mean_pathlen_std / np.maximum(n_nodes2, 1)
null_ratio_mean   = mnl_mean         / np.maximum(null_n_nodes, 1)
null_ratio_mean_s = mnl_std          / np.maximum(null_n_nodes, 1)
recomb_rmean      = rmnl_mean        / np.maximum(null_n_nodes, 1)
recomb_rmean_std  = rmnl_std         / np.maximum(null_n_nodes, 1)
_eb(ax6, batch_n, ratio_mean, ratio_mean_std, linestyle='--', label='data', color='blue')
_eb(ax6, batch_n, null_ratio_mean, null_ratio_mean_s, linestyle=':', label='config-model null', color='purple')
_eb(ax6, batch_n, recomb_rmean, recomb_rmean_std, linestyle='-.', label='recombination null', color='seagreen')
ax6.set_xlabel('Plasmid No. in batch'); ax6.set_ylabel('Mean length / No. nodes')
ax6.set_title('Mean path length relative to No. nodes'); ax6.set_xscale('log'); ax6.set_yscale('log'); ax6.legend()
_eb(ax7, batch_n, path_coverage, coverage_std, linestyle='--', label='data', color='blue')
_eb(ax7, batch_n, cov_mean, cov_std, linestyle=':', label='config-model null', color='purple')
_eb(ax7, batch_n, rcov_mean, rcov_std, linestyle='-.', label='recombination null', color='seagreen')
ax7.set_xlabel('Plasmid No. in batch'); ax7.set_ylabel('Coverage (%)')
ax7.set_title('Coverage of unbranching paths'); ax7.set_xscale('log'); ax7.set_yscale('log'); ax7.legend()
plt.tight_layout()
plt.savefig(out_dir / 'hospital_paths_3.png', bbox_inches='tight', dpi=150)
plt.close()
print('Saved hospital_paths_3.png')
# ── degree-distribution histograms (3 batches: min, mid, max) ─────────────
print('Computing degree distribution histograms …')
sys.stdout.flush()
all_batches = sorted(
    int(p.name.split('_')[1])
    for p in gml_dir.glob(f'batch_*_{label}_domain_architecture_signed_network.graphml'))
nsp = len(all_batches)
hist_batches = [all_batches[0], all_batches[nsp // 2], all_batches[-1]]
cutoffs      = [30, 50, 100]
rng_hist  = np.random.default_rng(0)
_rng_hist = np.random.default_rng(int(rng_hist.integers(1e9)))
for batch_num, cutoff in zip(hist_batches, cutoffs):
    gml_path = gml_dir / f'batch_{batch_num}_{label}_domain_architecture_signed_network.graphml'
    if not gml_path.exists():
        print(f'  skipping batch {batch_num}'); continue
    G_obs = nx.read_graphml(str(gml_path))
    obs_degrees = [d for _, d in G_obs.degree()]
    n, m = G_obs.number_of_nodes(), G_obs.number_of_edges()
    in_seq  = [d for _, d in G_obs.in_degree()]
    out_seq = [d for _, d in G_obs.out_degree()]
    cfg_degrees, er_degrees, recomb_degrees = [], [], []
    _r = np.random.default_rng(int(_rng_hist.integers(1e9)))
    for _ in range(20):
        H = nx.directed_configuration_model(in_seq, out_seq,
                create_using=nx.DiGraph(), seed=int(_r.integers(1e9)))
        H.remove_edges_from(nx.selfloop_edges(H))
        cfg_degrees.extend([d for _, d in H.degree()])
    for _ in range(20):
        H = nx.gnm_random_graph(n, m, directed=True, seed=int(_r.integers(1e9)))
        er_degrees.extend([d for _, d in H.degree()])
    _rr = np.random.default_rng(7)
    for _ in range(20):
        H = recomb_null_graph(G_obs, threshold=RECOMB_THRESHOLD, rng=_rr)
        recomb_degrees.extend([d for _, d in H.degree()])
    obs_arr    = np.array(obs_degrees) if obs_degrees else np.array([0])
    cfg_arr    = np.array(cfg_degrees)
    er_arr     = np.array(er_degrees)
    recomb_arr = np.array(recomb_degrees)
    max_deg = max(int(np.percentile(obs_arr, 99)) + 2, 5)
    bins    = np.arange(0, max_deg + 2) - 0.5
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    fig.suptitle(f'Degree distribution — Duke University Hospital, batch {batch_num}\n'
                 f'(n={n} nodes, m={m} edges)', fontsize=13, y=1.03)
    datasets = [
        ('Observed',            obs_arr,    'darkred'),
        ('Configuration model', cfg_arr,    'steelblue'),
        ('Erdős–Rényi',         er_arr,     'darkorange'),
        ('Recombination null',  recomb_arr, 'seagreen'),
    ]
    for ax, (title, data, colour) in zip(axes.flat, datasets):
        counts, edges_ = np.histogram(data, bins=bins, density=True)
        centers = 0.5 * (edges_[:-1] + edges_[1:])
        ax.bar(centers, counts, width=0.85, color=colour, alpha=0.75,
               linewidth=0, align='center')
        if len(data) > 0:
            ax.axvline(np.mean(data), color='k', lw=1.2, ls='--',
                       label=f'mean={np.mean(data):.1f}')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Degree', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_yscale('log')
        ax.set_xlim(-0.5, max_deg + 0.5)
        ax.tick_params(axis='x', labelbottom=True, labelsize=8)
        ax.tick_params(axis='y', which='both', labelleft=True, labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    out_path = out_dir / f'hospital_degree_distribution_batch{batch_num}_3.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved hospital_degree_distribution_batch{batch_num}_3.png')
    # _4 variant: obs vs recomb bar chart
    obs_filtered    = obs_arr[obs_arr <= cutoff]
    recomb_filtered = recomb_arr[recomb_arr <= cutoff]
    bins2   = np.arange(0, cutoff + 2)
    obs_counts,    obs_edges2    = np.histogram(obs_filtered,    bins=bins2, density=True)
    recomb_counts, recomb_edges2 = np.histogram(recomb_filtered, bins=bins2, density=True)
    centers2 = obs_edges2[:-1]
    plt.figure(figsize=(10, 6))
    plt.bar(centers2, obs_counts,    width=1, alpha=0.9, color='darkgreen',  label='plasmid data')
    plt.bar(centers2, recomb_counts, width=1, alpha=0.7, color='darkorange', label='recombination null model')
    plt.xlabel('Degree', fontsize=12); plt.ylabel('Density', fontsize=12)
    plt.title(f'Degree distribution — Duke University Hospital, batch {batch_num}', fontsize=14)
    plt.legend(fontsize=10); plt.grid(True, alpha=0.3, linestyle='--'); plt.tight_layout()
    out_path4 = out_dir / f'hospital_degree_distribution_batch{batch_num}_4.png'
    plt.savefig(out_path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved hospital_degree_distribution_batch{batch_num}_4.png')
# ── scale-free plots (max-batch graphml) ──────────────────────────────────
max_batch_gml = sorted(
    gml_dir.glob(f'batch_*_{label}_domain_architecture_signed_network.graphml'),
    key=lambda p: int(p.name.split('_')[1]))[-1]
print(f'Scale-free plots using: {max_batch_gml.name}')
G_max    = nx.read_graphml(str(max_batch_gml))
degrees  = np.array([d for _, d in G_max.degree()])
deg_vals, counts = np.unique(degrees, return_counts=True)
pk = counts / counts.sum()
mask = deg_vals > 0
deg_vals, pk = deg_vals[mask], pk[mask]
plt.figure(figsize=(6, 5))
plt.scatter(deg_vals, pk, c='black', s=4)
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Degree'); plt.ylabel('Fraction of nodes')
plt.title('Degree distribution — Duke University Hospital')
plt.tight_layout()
plt.savefig(out_dir / 'scale_free_linear_plot_hospital.png', dpi=150)
plt.close()
print('Saved scale_free_linear_plot_hospital.png')
cutoff_linear = 200
degrees2 = np.array([d for _, d in G_max.degree()])
deg_lin  = degrees2[degrees2 <= cutoff_linear]
bins3    = np.arange(0, cutoff_linear + 2)
cnt3, edges3 = np.histogram(deg_lin, bins=bins3)
frac3  = cnt3 / cnt3.sum()
ctrs3  = edges3[:-1]
plt.figure(figsize=(7, 5))
plt.bar(ctrs3, frac3, width=2, color='darkred')
plt.xlabel('degree'); plt.ylabel('fraction of nodes')
plt.title('Degree distribution — Duke University Hospital')
plt.tight_layout()
plt.savefig(out_dir / 'scale_free_hist_plot_hospital.png', dpi=150)
plt.close()
print('Saved scale_free_hist_plot_hospital.png')
print(f'\nAll plots written to: {out_dir}')

