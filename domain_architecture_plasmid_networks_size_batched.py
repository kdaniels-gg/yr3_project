import os
import csv
import argparse
from collections import defaultdict
from pathlib import Path
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
import pickle
import igraph as ig
import leidenalg
import random
import sys 

############################################################################################################################
############################################################################################################################
############################################################################################################################
#PLASMID BATCHED DEGREE, EDGES, DENSITY ETC.


data_dir = Path(os.path.join(os.getcwd(),'plasmid_motif_network/intermediate'))
files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))

df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()

df_merged = df_merged.with_columns(
    pl.col('strand').cast(pl.Int32).alias('strand')
)


output_path = Path(os.path.join(os.getcwd(), 'plasmid_batched_graphs'))


all_plasmids = list(set(df_merged['plasmid']))
                    
domain_df = pd.read_csv('Pfam-A.clans.tsv', sep='\t', header=None)
domain_dict = dict(zip(domain_df[3].tolist(), domain_df[2].tolist()))

random.seed(42)

F_stats_csv_path = os.path.join(output_path, 'F_graph_statistics.csv')

G_stats_csv_path = os.path.join(output_path, 'G_graph_statistics.csv')


from multiprocessing import Pool
import signal


N_NULL = 100

def config_model_stats(G_obs, n_null=N_NULL, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    in_seq  = [d for _, d in G_obs.in_degree()]
    out_seq = [d for _, d in G_obs.out_degree()]
    n_nodes = G_obs.number_of_nodes()
    if n_nodes < 2 or sum(in_seq) == 0:
        return {
            'null_edges_mean': 0,
            'null_edges_std': 0,
            'null_degree_mean': 0,
            'null_degree_std': 0,
            'null_density_mean': 0,
            'null_density_std': 0
        }
    null_edges = np.empty(n_null)
    null_degrees = np.empty(n_null)
    null_densities = np.empty(n_null)
    max_e = n_nodes * (n_nodes - 1)
    for i in range(n_null):
        H = nx.directed_configuration_model(
            in_seq,
            out_seq,
            create_using=nx.DiGraph(),
            seed=int(rng.integers(1e9))
        )
        H.remove_edges_from(nx.selfloop_edges(H))
        e = H.number_of_edges()
        null_edges[i] = e
        null_degrees[i] = 2 * e / n_nodes
        null_densities[i] = e / max_e if max_e > 0 else 0
    return {
        'null_edges_mean': float(null_edges.mean()),
        'null_edges_std': float(null_edges.std()),
        'null_degree_mean': float(null_degrees.mean()),
        'null_degree_std': float(null_degrees.std()),
        'null_density_mean': float(null_densities.mean()),
        'null_density_std': float(null_densities.std())
    }





RECOMB_THRESHOLD = 3

def recomb_null_graph(G_obs, threshold=RECOMB_THRESHOLD, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    nodes = list(G_obs.nodes())
    n = len(nodes)
    if n < 2:
        return nx.DiGraph()
    recomb_budget = sum(max(0, d - 2) for _, d in G_obs.degree())
    H = nx.DiGraph()
    H.add_nodes_from(nodes)
    for i in range(n):
        H.add_edge(nodes[i], nodes[(i + 1) % n])
    existing_edges = set(H.edges())
    node_indices = list(range(n))
    while recomb_budget >= 2:
        rng.shuffle(node_indices)
        added = False
        for ui in node_indices:
            vi = int(rng.integers(n))
            if vi == ui:
                vi = (ui + 1) % n
            u = nodes[ui]
            v = nodes[vi]
            if (u, v) not in existing_edges:
                H.add_edge(u, v)
                existing_edges.add((u, v))
                recomb_budget -= 2
                added = True
                break
        if not added:
            break
    return H



def recomb_model_stats(G_obs, n_null=N_NULL, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n_nodes = G_obs.number_of_nodes()
    if n_nodes < 2:
        return {
            'recomb_edges_mean': 0,
            'recomb_edges_std': 0,
            'recomb_degree_mean': 0,
            'recomb_degree_std': 0,
            'recomb_density_mean': 0,
            'recomb_density_std': 0
        }
    null_edges = np.empty(n_null)
    null_degrees = np.empty(n_null)
    null_densities = np.empty(n_null)
    max_e = n_nodes * (n_nodes - 1)
    for i in range(n_null):
        H = recomb_null_graph(G_obs, threshold=RECOMB_THRESHOLD, rng=rng)
        e = H.number_of_edges()
        null_edges[i] = e
        null_degrees[i] = 2 * e / n_nodes
        null_densities[i] = e / max_e if max_e > 0 else 0
    return {
        'recomb_edges_mean': float(null_edges.mean()),
        'recomb_edges_std': float(null_edges.std()),
        'recomb_degree_mean': float(null_degrees.mean()),
        'recomb_degree_std': float(null_degrees.std()),
        'recomb_density_mean': float(null_densities.mean()),
        'recomb_density_std': float(null_densities.std())
    }





def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def compute_null_for_graph(gml_path):
    batch_num = int(gml_path.name.split('_')[0])
    G_obs = nx.read_graphml(str(gml_path))
    rng_null_gen = np.random.default_rng(42 + batch_num)
    stats = config_model_stats(G_obs, rng=rng_null_gen)
    stats.update(recomb_model_stats(G_obs, rng=rng_null_gen))
    stats['plasmid_number'] = batch_num
    print(
        f'  batch {batch_num}: config null_edges={stats["null_edges_mean"]:.1f} '
        f'recomb null_edges={stats["recomb_edges_mean"]:.1f}'
    )
    return stats




batch_files_for_null = sorted(
    Path(output_path).glob('*_domain_architecture_signed_network.graphml'),
    key=lambda p: int(p.name.split('_')[0])
)

null_rows = []
try:
    with Pool(4, initializer=init_worker) as pool:
        for result in pool.imap_unordered(compute_null_for_graph, batch_files_for_null):
            null_rows.append(result)
except KeyboardInterrupt:
    pool.terminate()
    pool.join()




null_df = pd.DataFrame(null_rows).sort_values('plasmid_number')
null_csv_path = os.path.join(output_path, 'null_model_statistics_3.csv')
null_df.to_csv(null_csv_path, index=False)
print(f'Null model stats -> {null_csv_path}')


null_df = pd.read_csv(null_csv_path)


graph_dir = Path(os.path.join(os.getcwd(), 'plasmid_batched_graphs'))
f_csv   = graph_dir / 'F_graph_statistics.csv'
g_csv   = graph_dir / 'G_graph_statistics.csv'
null_csv = graph_dir / 'null_model_statistics_3.csv'

fdf     = pd.read_csv(f_csv)
gdf     = pd.read_csv(g_csv)
null_df = pd.read_csv(null_csv)

batch_x  = fdf['plasmid_number'].tolist()
f_nodes  = fdf['node_number'].tolist()
f_edges  = fdf['edge_number'].tolist()
g_edges  = gdf['edge_number'].tolist()

#per-run std (stored alongside mean in csvs)
f_edges_std = fdf['edge_number_std'].tolist() if 'edge_number_std' in fdf.columns else [0]*len(f_edges)
g_edges_std = gdf['edge_number_std'].tolist() if 'edge_number_std' in gdf.columns else [0]*len(g_edges)

complete_edges  = [n*(n-1) for n in f_nodes]
complete_degree = [(n-1)   for n in f_nodes]

f_density   = [o/c if c > 0 else 0 for o, c in zip(f_edges, complete_edges)]
g_density   = [o/c if c > 0 else 0 for o, c in zip(g_edges, complete_edges)]
f_degree    = [2*e/n if n > 0 else 0 for e, n in zip(f_edges, f_nodes)]
g_degree    = [2*e/n if n > 0 else 0 for e, n in zip(g_edges, f_nodes)]

#propagate std to derived quantities (approximate, first-order)
f_density_std = [s/c if c > 0 else 0 for s, c in zip(f_edges_std, complete_edges)]
g_density_std = [s/c if c > 0 else 0 for s, c in zip(g_edges_std, complete_edges)]
f_degree_std  = [2*s/n if n > 0 else 0 for s, n in zip(f_edges_std, f_nodes)]
g_degree_std  = [2*s/n if n > 0 else 0 for s, n in zip(g_edges_std, f_nodes)]

#null model series (aligned to batch_x)
null_merged = pd.merge(pd.DataFrame({'plasmid_number': batch_x}), null_df,
                        on='plasmid_number', how='left').fillna(0)
#config-model null
null_edges_mean   = null_merged['null_edges_mean'].tolist()
null_edges_std    = null_merged['null_edges_std'].tolist()
null_degree_mean  = null_merged['null_degree_mean'].tolist()
null_degree_std   = null_merged['null_degree_std'].tolist()
null_density_mean = null_merged['null_density_mean'].tolist()
null_density_std  = null_merged['null_density_std'].tolist()
#recombination null (may be zeros if CSV predates recomb model)
recomb_edges_mean   = null_merged['recomb_edges_mean'].tolist()   if 'recomb_edges_mean'   in null_merged.columns else [0]*len(batch_x)
recomb_edges_std    = null_merged['recomb_edges_std'].tolist()    if 'recomb_edges_std'    in null_merged.columns else [0]*len(batch_x)
recomb_degree_mean  = null_merged['recomb_degree_mean'].tolist()  if 'recomb_degree_mean'  in null_merged.columns else [0]*len(batch_x)
recomb_degree_std   = null_merged['recomb_degree_std'].tolist()   if 'recomb_degree_std'   in null_merged.columns else [0]*len(batch_x)
recomb_density_mean = null_merged['recomb_density_mean'].tolist() if 'recomb_density_mean' in null_merged.columns else [0]*len(batch_x)
recomb_density_std  = null_merged['recomb_density_std'].tolist()  if 'recomb_density_std'  in null_merged.columns else [0]*len(batch_x)

batch_arr = np.array(batch_x)



#edges plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
fig.suptitle('No. Edges / No. plasmids in batch')

ax1.plot(batch_arr, f_edges, '--', label='data', color='blue')
ax1.fill_between(batch_arr,
                 np.array(f_edges) - np.array(f_edges_std),
                 np.array(f_edges) + np.array(f_edges_std),
                 alpha=0.2, color='blue', label='±1 SD (10 runs)')
ax1.plot(batch_arr, null_edges_mean, ':', label='config-model null', color='purple')
ax1.fill_between(batch_arr,
                 np.array(null_edges_mean) - np.array(null_edges_std),
                 np.array(null_edges_mean) + np.array(null_edges_std),
                 alpha=0.15, color='purple')
ax1.plot(batch_arr, recomb_edges_mean, '-.', label='recombination null', color='seagreen')
ax1.fill_between(batch_arr,
                 np.array(recomb_edges_mean) - np.array(recomb_edges_std),
                 np.array(recomb_edges_mean) + np.array(recomb_edges_std),
                 alpha=0.15, color='seagreen')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('No. plasmids in batch'); ax1.set_ylabel('No. edges')
ax1.set_title("Signed architecture network"); ax1.legend()

ax2.plot(batch_arr, g_edges, '--', label='data', color='green')
ax2.fill_between(batch_arr,
                 np.array(g_edges) - np.array(g_edges_std),
                 np.array(g_edges) + np.array(g_edges_std),
                 alpha=0.2, color='green', label='±1 SD (10 runs)')
ax2.plot(batch_arr, null_edges_mean, ':', label='config-model null', color='purple')
ax2.fill_between(batch_arr,
                 np.array(null_edges_mean) - np.array(null_edges_std),
                 np.array(null_edges_mean) + np.array(null_edges_std),
                 alpha=0.15, color='purple')
ax2.plot(batch_arr, recomb_edges_mean, '-.', label='recombination null', color='seagreen')
ax2.fill_between(batch_arr,
                 np.array(recomb_edges_mean) - np.array(recomb_edges_std),
                 np.array(recomb_edges_mean) + np.array(recomb_edges_std),
                 alpha=0.15, color='seagreen')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('No. plasmids in batch'); ax2.set_ylabel('No. edges')
ax2.set_title("Unsigned architecture network"); ax2.legend()

plt.savefig('edges4.png', dpi=150, bbox_inches='tight'); plt.close()



#degree plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
fig.suptitle('Avg. node degree / No. plasmids in batch')

ax1.plot(batch_arr, f_degree, '--', label='data', color='blue')
ax1.fill_between(batch_arr,
                 np.array(f_degree) - np.array(f_degree_std),
                 np.array(f_degree) + np.array(f_degree_std),
                 alpha=0.2, color='blue', label='±1 SD (10 runs)')
ax1.plot(batch_arr, null_degree_mean, ':', label='config-model null', color='purple')
ax1.fill_between(batch_arr,
                 np.array(null_degree_mean) - np.array(null_degree_std),
                 np.array(null_degree_mean) + np.array(null_degree_std),
                 alpha=0.15, color='purple')
ax1.plot(batch_arr, recomb_degree_mean, '-.', label='recombination null', color='seagreen')
ax1.fill_between(batch_arr,
                 np.array(recomb_degree_mean) - np.array(recomb_degree_std),
                 np.array(recomb_degree_mean) + np.array(recomb_degree_std),
                 alpha=0.15, color='seagreen')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('No. plasmids in batch'); ax1.set_ylabel('Avg. node degree')
ax1.set_title("Signed architecture network"); ax1.legend()

ax2.plot(batch_arr, g_degree, '--', label='data', color='green')
ax2.fill_between(batch_arr,
                 np.array(g_degree) - np.array(g_degree_std),
                 np.array(g_degree) + np.array(g_degree_std),
                 alpha=0.2, color='green', label='±1 SD (10 runs)')
ax2.plot(batch_arr, null_degree_mean, ':', label='config-model null', color='purple')
ax2.fill_between(batch_arr,
                 np.array(null_degree_mean) - np.array(null_degree_std),
                 np.array(null_degree_mean) + np.array(null_degree_std),
                 alpha=0.15, color='purple')
ax2.plot(batch_arr, recomb_degree_mean, '-.', label='recombination null', color='seagreen')
ax2.fill_between(batch_arr,
                 np.array(recomb_degree_mean) - np.array(recomb_degree_std),
                 np.array(recomb_degree_mean) + np.array(recomb_degree_std),
                 alpha=0.15, color='seagreen')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('No. plasmids in batch'); ax2.set_ylabel('Avg. node degree')
ax2.set_title("Unsigned architecture network"); ax2.legend()

plt.savefig('degrees4.png', dpi=150, bbox_inches='tight'); plt.close()



#density plots
fig, ax1 = plt.subplots(figsize=(10, 12))
fig.suptitle('Density / No. plasmids in batch')

ax1.plot(batch_arr, f_density, '--', label='signed data', color='blue')
ax1.fill_between(batch_arr,
                 np.array(f_density) - np.array(f_density_std),
                 np.array(f_density) + np.array(f_density_std),
                 alpha=0.2, color='blue')
ax1.plot(batch_arr, g_density, ':', label='unsigned data', color='red')
ax1.fill_between(batch_arr,
                 np.array(g_density) - np.array(g_density_std),
                 np.array(g_density) + np.array(g_density_std),
                 alpha=0.2, color='red')
ax1.plot(batch_arr, null_density_mean, '-.', label='config-model null', color='purple')
ax1.fill_between(batch_arr,
                 np.array(null_density_mean) - np.array(null_density_std),
                 np.array(null_density_mean) + np.array(null_density_std),
                 alpha=0.15, color='purple')
ax1.plot(batch_arr, recomb_density_mean, linestyle=(0, (3, 1, 1, 1)), label='recombination null', color='seagreen')
ax1.fill_between(batch_arr,
                 np.array(recomb_density_mean) - np.array(recomb_density_std),
                 np.array(recomb_density_mean) + np.array(recomb_density_std),
                 alpha=0.15, color='seagreen')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('No. plasmids in batch')
ax1.set_ylabel('No. Edges / No. possible edges')
ax1.legend()
yticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
ax1.set_yticks(yticks); ax1.set_yticklabels([str(y) for y in yticks])

plt.savefig('density4.png', dpi=150, bbox_inches='tight'); plt.close()


############################################################################################################################
############################################################################################################################
############################################################################################################################
#UNBRANCHING PATHS


graph_dir = Path(os.path.join(os.getcwd(), 'plasmid_batched_graphs'))

def find_unbranching_paths(G):
    #gets max unbranching paths based on if nodes have only one unique node/domain in front and behind
    #trace back through the assigned 'simple nodes' to get a path
    succ   = {n: list(set(G.successors(n)))   for n in G.nodes()}
    pred   = {n: list(set(G.predecessors(n))) for n in G.nodes()}
    in_deg  = {n: len(pred[n]) for n in G.nodes()}
    out_deg = {n: len(succ[n]) for n in G.nodes()}
    simple = {n for n in G.nodes() if in_deg[n] == 1 and out_deg[n] == 1}
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



#ER MODEL - DOESN'T RETAIN DEGREE DISTRIBUTION, EDGES ADDED RANDOMLY BASED ON NUMBER NODES AND EDGES FROM ORIGINAL NETWORK
N_NULL_ER = 50 
def er_null_degree_stats(G_obs, n_null=N_NULL_ER, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = G_obs.number_of_nodes()
    m = G_obs.number_of_edges()
    if n < 2 or m == 0:
        return 0.0, 0.0, []
    degrees = []
    all_deg_seqs = []
    for _ in range(n_null):
        H = nx.gnm_random_graph(n, m, directed=True,
                                seed=int(rng.integers(1e9)))
        degs = [d for _, d in H.degree()]
        degrees.append(np.mean(degs))
        all_deg_seqs.append(degs)
    return float(np.mean(degrees)), float(np.std(degrees)), all_deg_seqs


def path_stats_from_graph(G):
    paths = find_unbranching_paths(G)
    n_nodes = G.number_of_nodes()
    if paths:
        lengths = [len(p) for p in paths]
        return {
            'n_paths':   len(paths),
            'max_len':   max(lengths),
            'mean_len':  float(np.mean(lengths)),
            'coverage':  len(set(n for p in paths for n in p)) / n_nodes if n_nodes > 0 else 0.0,
        }
    return {'n_paths': 0, 'max_len': 0, 'mean_len': 0.0, 'coverage': 0.0}



#CONFIG MODEL - RETAINS DEGREE DISTRIBUTION, RANDOMLY REASSIGNS EDGES BETWEEN NODES
def config_null_path_stats(G_obs, n_null=50, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    in_seq  = [d for _, d in G_obs.in_degree()]
    out_seq = [d for _, d in G_obs.out_degree()]
    n_nodes = G_obs.number_of_nodes()
    if n_nodes < 2 or sum(in_seq) == 0:
        return {k: (0.0, 0.0) for k in ('n_paths','max_len','mean_len','coverage')}
    results = []
    for _ in range(n_null):
        H = nx.directed_configuration_model(
            in_seq, out_seq, create_using=nx.DiGraph(),
            seed=int(rng.integers(1e9))
        )
        H.remove_edges_from(nx.selfloop_edges(H))
        results.append(path_stats_from_graph(H))
    out = {}
    for key in ('n_paths','max_len','mean_len','coverage'):
        vals = [r[key] for r in results]
        out[key] = (float(np.mean(vals)), float(np.std(vals)))
    return out



#RECOMB MODEL
#DEFINE PLASMID NETWORK NODES WITH DEGREE >2 as RECOMBINATION EVENTS 
#IE DEGREE 3 -> 1 RECOMBINATION, DEGREE 4 -> 2 RECOMBINATIONS 
#DEFINE A RECOMBINATION BUDGET FROM ORIGINAL PLASMID NETWORK
#MAKE DIRECTED CIRCULAR NETWORK AS START POINT FOR NULL (ALL NODES HAVE DEGREE 2), THEN RANDOMLY PLACE EDGES AS WITHIN RECOMBINATION BUDGET 

##!!! WORKING



def recomb_null_path_stats(G_obs, n_null=50, threshold=RECOMB_THRESHOLD, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n_nodes = G_obs.number_of_nodes()
    if n_nodes < 2:
        return {k: (0.0, 0.0) for k in ('n_paths','max_len','mean_len','coverage')}
    results = []
    for _ in range(n_null):
        H = recomb_null_graph(G_obs, threshold=threshold, rng=rng)
        results.append(path_stats_from_graph(H))
    out = {}
    for key in ('n_paths','max_len','mean_len','coverage'):
        vals = [r[key] for r in results]
        out[key] = (float(np.mean(vals)), float(np.std(vals)))
    return out

#make csvs for paths plots with config and recomb nulls 
all_path_stats = []
batch_files = sorted(graph_dir.glob('*_domain_architecture_network.graphml'),
                     key=lambda p: int(p.name.split('_')[0]))

path_csv_rows = []
null_path_rows = []




for gml_path in batch_files:
    batch_num = int(gml_path.name.split('_')[0])
    G = nx.read_graphml(str(gml_path))
    paths = find_unbranching_paths(G)
    if paths:
        lengths = [len(p) for p in paths]
        max_len = max(lengths)
        mean_len = sum(lengths) / len(lengths)
        unique_nodes_in_paths = set(n for p in paths for n in p)
        total_nodes_in_paths = len(unique_nodes_in_paths)
    else:
        lengths, max_len, mean_len, total_nodes_in_paths = [], 0, 0.0, 0
    n_nodes = G.number_of_nodes()
    coverage = total_nodes_in_paths / n_nodes if n_nodes > 0 else 0.0
    all_path_stats.append({
        'batch':            batch_num,
        'n_nodes':          n_nodes,
        'n_paths':          len(paths),
        'max_path_len':     max_len,
        'mean_path_len':    round(mean_len, 2),
        'coverage':         round(coverage, 4),
    })
    for i, path in enumerate(sorted(paths, key=len, reverse=True)):
        path_csv_rows.append({
            'batch':      batch_num,
            'path_index': i,
            'length':     len(path),
            'path':       ' -> '.join(path),
        })
    null_ps = config_null_path_stats(G, n_null=50)
    recomb_ps = recomb_null_path_stats(G, n_null=50)
    null_path_rows.append({
        'batch':                   batch_num,
        'n_nodes':                 n_nodes,
        'null_n_paths_mean':       null_ps['n_paths'][0],
        'null_n_paths_std':        null_ps['n_paths'][1],
        'null_max_len_mean':       null_ps['max_len'][0],
        'null_max_len_std':        null_ps['max_len'][1],
        'null_mean_len_mean':      null_ps['mean_len'][0],
        'null_mean_len_std':       null_ps['mean_len'][1],
        'null_cov_mean':           null_ps['coverage'][0] * 100,
        'null_cov_std':            null_ps['coverage'][1] * 100,
        'recomb_n_paths_mean':     recomb_ps['n_paths'][0],
        'recomb_n_paths_std':      recomb_ps['n_paths'][1],
        'recomb_max_len_mean':     recomb_ps['max_len'][0],
        'recomb_max_len_std':      recomb_ps['max_len'][1],
        'recomb_mean_len_mean':    recomb_ps['mean_len'][0],
        'recomb_mean_len_std':     recomb_ps['mean_len'][1],
        'recomb_cov_mean':         recomb_ps['coverage'][0] * 100,
        'recomb_cov_std':          recomb_ps['coverage'][1] * 100,
    })
    print(f'  batch={batch_num:6d}: {n_nodes:5d} nodes, '
          f'{len(paths):4d} paths, max_len={max_len}, coverage={coverage:.1%}')



#!!! WORKING HERE
out_dir = Path(os.path.join(os.getcwd(), 'plasmid_graphs_analysis'))
os.makedirs(out_dir, exist_ok=True)
paths_csv_path = out_dir / 'unbranching_paths_3.csv'

path_df = pd.DataFrame.from_dict(path_csv_rows)
path_df.to_csv(paths_csv_path, index=False)

pathstats_csv_path = out_dir / 'unbranching_paths_stats_3.csv'
pathstats_df = pd.DataFrame.from_dict(all_path_stats)
pathstats_df['coverage_percentage'] = [x*100 for x in pathstats_df['coverage']]
pathstats_df.to_csv(pathstats_csv_path, index=False)


null_path_df = pd.DataFrame(null_path_rows).sort_values('batch')
null_path_df.to_csv(out_dir / 'null_path_statistics_3.csv', index=False)
null_path_df = null_path_df.set_index('batch').reindex(pathstats_df['batch']).reset_index()


#PLOT PATHS WITH CONFIG AND RECOOMB NULL MODELS FOR COMPARISON
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(10, 70))
fig.suptitle('Unbranching paths statistics', y=1.0)
fig.subplots_adjust(top=0.98)

#base arrays
batch_n      = pathstats_df['batch'].to_numpy()
n_nodes2     = pathstats_df['n_nodes'].to_numpy()
n_paths      = pathstats_df['n_paths'].to_numpy()
max_pathlen  = pathstats_df['max_path_len'].to_numpy()
mean_pathlen = pathstats_df['mean_path_len'].to_numpy()
path_coverage = pathstats_df['coverage_percentage'].to_numpy()

def _std(df, col):
    c = col + '_std'
    return df[c].to_numpy() if c in df.columns else np.zeros(len(df))


n_paths_std      = _std(pathstats_df, 'n_paths')
max_pathlen_std  = _std(pathstats_df, 'max_path_len')
mean_pathlen_std = _std(pathstats_df, 'mean_path_len')
coverage_std     = _std(pathstats_df, 'coverage_percentage')


def _eb(ax, x, y, ye, **kw):
    ax.plot(x, y, **kw)
    if np.any(ye > 0):
        ax.fill_between(x, y - ye, y + ye, alpha=0.2, color=kw.get('color','blue'))


#config null arrays
np_mean  = null_path_df['null_n_paths_mean'].to_numpy()
np_std   = null_path_df['null_n_paths_std'].to_numpy()
ml_mean  = null_path_df['null_max_len_mean'].to_numpy()
ml_std   = null_path_df['null_max_len_std'].to_numpy()
mnl_mean = null_path_df['null_mean_len_mean'].to_numpy()
mnl_std  = null_path_df['null_mean_len_std'].to_numpy()
cov_mean = null_path_df['null_cov_mean'].to_numpy()
cov_std  = null_path_df['null_cov_std'].to_numpy()

#recombination null arrays
rp_mean   = null_path_df['recomb_n_paths_mean'].to_numpy()
rp_std    = null_path_df['recomb_n_paths_std'].to_numpy()
rml_mean  = null_path_df['recomb_max_len_mean'].to_numpy()
rml_std   = null_path_df['recomb_max_len_std'].to_numpy()
rmnl_mean = null_path_df['recomb_mean_len_mean'].to_numpy()
rmnl_std  = null_path_df['recomb_mean_len_std'].to_numpy()
rcov_mean = null_path_df['recomb_cov_mean'].to_numpy()
rcov_std  = null_path_df['recomb_cov_std'].to_numpy()
null_n_nodes = null_path_df['n_nodes'].to_numpy()

#plot shit
_eb(ax1, batch_n, n_paths, n_paths_std, linestyle='--', label='data', color='blue')
_eb(ax1, batch_n, np_mean, np_std, linestyle=':', label='config-model null', color='purple')
_eb(ax1, batch_n, rp_mean, rp_std, linestyle='-.', label='recombination null', color='seagreen')
ax1.set_xlabel('Plasmid No. in batch'); ax1.set_ylabel('No. unbranching paths')
ax1.set_title('No. paths with increasing batch size')
ax1.set_xscale('log'); ax1.set_yscale('log'); ax1.legend()

ratio_paths = n_paths / np.maximum(n_nodes2, 1)
ratio_paths_std = n_paths_std / np.maximum(n_nodes2, 1)
null_ratio_paths = np_mean / np.maximum(null_n_nodes, 1)
null_ratio_paths_std = np_std / np.maximum(null_n_nodes, 1)
recomb_ratio_paths = rp_mean / np.maximum(null_n_nodes, 1)
recomb_ratio_paths_std = rp_std / np.maximum(null_n_nodes, 1)
_eb(ax2, batch_n, ratio_paths, ratio_paths_std, linestyle='--', label='data', color='blue')
_eb(ax2, batch_n, null_ratio_paths, null_ratio_paths_std, linestyle=':', label='config-model null', color='purple')
_eb(ax2, batch_n, recomb_ratio_paths, recomb_ratio_paths_std, linestyle='-.', label='recombination null', color='seagreen')
ax2.set_xlabel('Plasmid No. in batch'); ax2.set_ylabel('No. unbranching paths relative to node number')
ax2.set_title('No. paths relative to No. nodes with increasing batch size')
ax2.set_xscale('log'); ax2.set_yscale('log'); ax2.legend()

_eb(ax3, batch_n, max_pathlen, max_pathlen_std, linestyle='--', label='data', color='blue')
_eb(ax3, batch_n, ml_mean, ml_std, linestyle=':', label='config-model null', color='purple')
_eb(ax3, batch_n, rml_mean, rml_std, linestyle='-.', label='recombination null', color='seagreen')
ax3.set_xlabel('Plasmid No. in batch'); ax3.set_ylabel('Max length of unbranching paths')
ax3.set_title('Max length of unbranching paths with increasing batch size')
ax3.set_xscale('log'); ax3.set_yscale('log'); ax3.legend()

ratio_max = max_pathlen / np.maximum(n_nodes2, 1)
ratio_max_std = max_pathlen_std / np.maximum(n_nodes2, 1)
null_ratio_max = ml_mean / np.maximum(null_n_nodes, 1)
null_ratio_max_std = ml_std / np.maximum(null_n_nodes, 1)
recomb_ratio_max = rml_mean / np.maximum(null_n_nodes, 1)
recomb_ratio_max_std = rml_std / np.maximum(null_n_nodes, 1)
_eb(ax4, batch_n, ratio_max, ratio_max_std, linestyle='--', label='data', color='blue')
_eb(ax4, batch_n, null_ratio_max, null_ratio_max_std, linestyle=':', label='config-model null', color='purple')
_eb(ax4, batch_n, recomb_ratio_max, recomb_ratio_max_std, linestyle='-.', label='recombination null', color='seagreen')
ax4.set_xlabel('Plasmid No. in batch'); ax4.set_ylabel('Max length relative to No. nodes')
ax4.set_title('Max path length relative to No. nodes with increasing batch size')
ax4.set_xscale('log'); ax4.set_yscale('log'); ax4.legend()

_eb(ax5, batch_n, mean_pathlen, mean_pathlen_std, linestyle='--', label='data', color='blue')
_eb(ax5, batch_n, mnl_mean, mnl_std, linestyle=':', label='config-model null', color='purple')
_eb(ax5, batch_n, rmnl_mean, rmnl_std, linestyle='-.', label='recombination null', color='seagreen')
ax5.set_xlabel('Plasmid No. in batch'); ax5.set_ylabel('Mean length of unbranching paths')
ax5.set_title('Mean length of unbranching paths with increasing batch size')
ax5.set_xscale('log'); ax5.set_yscale('log'); ax5.legend()

ratio_mean = mean_pathlen / np.maximum(n_nodes2, 1)
ratio_mean_std = mean_pathlen_std / np.maximum(n_nodes2, 1)
null_ratio_mean = mnl_mean / np.maximum(null_n_nodes, 1)
null_ratio_mean_std = mnl_std / np.maximum(null_n_nodes, 1)
recomb_ratio_mean = rmnl_mean / np.maximum(null_n_nodes, 1)
recomb_ratio_mean_std = rmnl_std / np.maximum(null_n_nodes, 1)
_eb(ax6, batch_n, ratio_mean, ratio_mean_std, linestyle='--', label='data', color='blue')
_eb(ax6, batch_n, null_ratio_mean, null_ratio_mean_std, linestyle=':', label='config-model null', color='purple')
_eb(ax6, batch_n, recomb_ratio_mean, recomb_ratio_mean_std, linestyle='-.', label='recombination null', color='seagreen')
ax6.set_xlabel('Plasmid No. in batch'); ax6.set_ylabel('Mean length relative to No. nodes')
ax6.set_title('Mean path length relative to No. nodes with increasing batch size')
ax6.set_xscale('log'); ax6.set_yscale('log'); ax6.legend()

_eb(ax7, batch_n, path_coverage, coverage_std, linestyle='--', label='data', color='blue')
_eb(ax7, batch_n, cov_mean, cov_std, linestyle=':', label='config-model null', color='purple')
_eb(ax7, batch_n, rcov_mean, rcov_std, linestyle='-.', label='recombination null', color='seagreen')
ax7.set_xlabel('Plasmid No. in batch'); ax7.set_ylabel('Coverage of unbranching paths')
ax7.set_title('Coverage of unbranching paths with increasing batch size')
ax7.set_xscale('log'); ax7.set_yscale('log'); ax7.legend()

plt.savefig('paths5.png', bbox_inches='tight')
plt.close()


############################################################################################################################
############################################################################################################################
############################################################################################################################

#DEGREE DISTRIBUTION HISTS
#looks at a single small medium and max batch as defined for E.coli (biggest species group, done so you can compare in-species vs overall) because I can't be bothered to deal with more than that. uses plasmid, recomb, ER and config nulls.


HIST_BATCH_SIZES = [1, 10, 213, 7867]

_all_batch_nums = sorted(int(p.name.split('_')[0])
                         for p in graph_dir.glob('*_domain_architecture_signed_network.graphml'))


if HIST_BATCH_SIZES is None:
    _n = len(_all_batch_nums)
    HIST_BATCH_SIZES = [
        _all_batch_nums[0],
        _all_batch_nums[_n // 2],
        _all_batch_nums[-1],
    ]
print(f'Degree histogram batches: {HIST_BATCH_SIZES}')


rng_hist = np.random.default_rng(0)



for batch_num in HIST_BATCH_SIZES:
    gml_path = graph_dir / f'{batch_num}_domain_architecture_signed_network.graphml'
    if not gml_path.exists():
        print(f'  skipping {batch_num}: graphml not found')
        continue
    G_obs = nx.read_graphml(str(gml_path))
    obs_degrees = [d for _, d in G_obs.degree()]
    n, m = G_obs.number_of_nodes(), G_obs.number_of_edges()
    in_seq  = [d for _, d in G_obs.in_degree()]
    out_seq = [d for _, d in G_obs.out_degree()]
    cfg_degrees = []
    for _ in range(20):
        H = nx.directed_configuration_model(
            in_seq, out_seq, create_using=nx.DiGraph(),
            seed=int(rng_hist.integers(1e9))
        )
        H.remove_edges_from(nx.selfloop_edges(H))
        cfg_degrees.extend([d for _, d in H.degree()])
    er_degrees = []
    for _ in range(20):
        H = nx.gnm_random_graph(n, m, directed=True,
                                seed=int(rng_hist.integers(1e9)))
        er_degrees.extend([d for _, d in H.degree()])
    recomb_degrees = []
    rng_recomb = np.random.default_rng(7)
    for _ in range(20):
        H = recomb_null_graph(G_obs, threshold=RECOMB_THRESHOLD, rng=rng_recomb)
        recomb_degrees.extend([d for _, d in H.degree()])
    obs_arr    = np.array(obs_degrees) if obs_degrees else np.array([0])
    cfg_arr    = np.array(cfg_degrees)
    er_arr     = np.array(er_degrees)
    recomb_arr = np.array(recomb_degrees)
    #use integer bins up to 99th-pct of *observed* degrees only (nulls can have
    #very long tails that crush the interesting low-degree region)
    max_deg = max(int(np.percentile(obs_arr, 99)) + 2, 5)
    bins = np.arange(0, max_deg + 2) - 0.5      #integer-centred bins
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    fig.suptitle(f'Degree distribution — batch {batch_num}', fontsize=13, y=1.01)
    datasets = [
        ('Observed',            obs_arr,    'darkred'),
        ('Configuration model', cfg_arr,    'steelblue'),
        ('Erdős–Rényi',        er_arr,     'darkorange'),
        ('Recombination null',  recomb_arr, 'seagreen'),
    ]
    for ax, (title, data, colour) in zip(axes.flat, datasets):
        counts, edges_ = np.histogram(data, bins=bins, density=True)
        centers = 0.5 * (edges_[:-1] + edges_[1:])
        #bar chart without distracting per-bar outlines; alpha helps overlap reads
        ax.bar(centers, counts, width=0.85, color=colour, alpha=0.75,
               linewidth=0, align='center')
        #vertical line at mean for quick comparison across panels
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
    plt.savefig(out_dir / f'degree_distribution_batch{batch_num}_5.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'  Degree hist saved for batch {batch_num}')




#!!!
cutoffs = [20, 40, 140, 200]

out_dir = Path(os.path.join(os.getcwd(), 'plasmid_graphs_analysis'))

for batch_num in HIST_BATCH_SIZES:
    gml_path = graph_dir / f'{batch_num}_domain_architecture_signed_network.graphml'
    G_obs = nx.read_graphml(str(gml_path))
    obs_degrees = [d for _, d in G_obs.degree()]
    n, m = G_obs.number_of_nodes(), G_obs.number_of_edges()
    in_seq  = [d for _, d in G_obs.in_degree()]
    out_seq = [d for _, d in G_obs.out_degree()]
    recomb_degrees = []
    rng_recomb = np.random.default_rng(7)
    for _ in range(20):
        H = recomb_null_graph(G_obs, threshold=RECOMB_THRESHOLD, rng=rng_recomb)
        recomb_degrees.extend([d for _, d in H.degree()])
    obs_arr    = np.array(obs_degrees) if obs_degrees else np.array([0])
    recomb_arr = np.array(recomb_degrees)
    cutoff = cutoffs[HIST_BATCH_SIZES.index(batch_num)]
    obs_filtered = obs_arr[obs_arr <= cutoff]
    recomb_filtered = recomb_arr[recomb_arr <= cutoff]
    bins = np.arange(0, cutoff + 2) #include last bin edge
    obs_counts, obs_edges = np.histogram(obs_filtered, bins=bins, density=True)
    recomb_counts, recomb_edges = np.histogram(recomb_filtered, bins=bins, density=True)
    #bin centres
    centers = obs_edges[:-1]
    plt.figure(figsize=(10, 6))
    plt.bar(centers, obs_counts, width=1, alpha=0.9, color='darkgreen', label=f'plasmid data')
    plt.bar(centers, recomb_counts, width=1, alpha=0.7, color='darkorange', label=f'recombination null model')
    plt.xlabel('Degree', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Degree distribution', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_dir / f'degree_distribution_batch{batch_num}_5.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Degree hist saved for batch {batch_num}')


#use integer bins up to 99th-pct of *observed* degrees only (nulls can have
#very long tails that crush the interesting low-degree region)
#max_deg = max(int(np.percentile(obs_arr, 99)) + 2, 5)
#bins = np.arange(0, max_deg + 2) - 0.5      #integer-centred bins
#fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
#fig.suptitle(f'Degree distribution — batch {batch_num}', fontsize=13, y=1.01)
#datasets = [
#    ('Observed',            obs_arr,    'darkred'),
#    ('Configuration model', cfg_arr,    'steelblue'),
#    ('Erdős–Rényi',        er_arr,     'darkorange'),
#    ('Recombination null',  recomb_arr, 'seagreen'),
#]
#for ax, (title, data, colour) in zip(axes.flat, datasets):
#    counts, edges_ = np.histogram(data, bins=bins, density=True)
#    centers = 0.5 * (edges_[:-1] + edges_[1:])
#    #bar chart without distracting per-bar outlines; alpha helps overlap reads
#    ax.bar(centers, counts, width=0.85, color=colour, alpha=0.75,
#           linewidth=0, align='center')
#    #vertical line at mean for quick comparison across panels
#    if len(data) > 0:
#        ax.axvline(np.mean(data), color='k', lw=1.2, ls='--',
#                   label=f'mean={np.mean(data):.1f}')
#    ax.set_title(title, fontsize=11)
#    ax.set_xlabel('Degree', fontsize=9)
#    ax.set_ylabel('Density', fontsize=9)
#    ax.set_yscale('log')
#    ax.set_xlim(-0.5, max_deg + 0.5)
#    ax.tick_params(axis='x', labelbottom=True, labelsize=8)
#    ax.tick_params(axis='y', which='both', labelleft=True, labelsize=8)
#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    ax.legend(fontsize=8, frameon=False)
#plt.tight_layout()
#plt.savefig(out_dir / f'degree_distribution_batch{batch_num}_4.png', dpi=150,
#            bbox_inches='tight')
#plt.close()
#print(f'  Degree hist saved for batch {batch_num}')







graph_dir = Path(os.path.join(os.getcwd(), 'plasmid_batched_graphs'))

batch_files = sorted(
    graph_dir.glob('*_domain_architecture_signed_network.graphml'),
    key=lambda p: int(p.name.split('_')[0])
)

max_graph_path = batch_files[-1]
print("Using:", max_graph_path)

G = nx.read_graphml(max_graph_path)

degrees = np.array([d for _, d in G.degree()])
deg_vals, counts = np.unique(degrees, return_counts=True)

pk = counts / counts.sum()
mask = deg_vals > 0
deg_vals = deg_vals[mask]
pk = pk[mask]

plt.figure(figsize=(6,5))
plt.scatter(deg_vals, pk, c='black', s=4)
plt.xscale('log')
plt.yscale('log')

plt.xlabel('Degree')
plt.ylabel('Fraction of nodes')
plt.title('Degree distribution')
plt.tight_layout()

plt.savefig('scale_free_linear_plot.png', dpi=150)
plt.close()





G = nx.read_graphml(max_graph_path)
degrees = np.array([d for _, d in G.degree()])

cutoff = 300 

deg_linear = degrees[degrees <= cutoff]

bins = np.arange(0, cutoff + 2)
counts, edges = np.histogram(deg_linear, bins=bins)

fraction = counts / counts.sum()
centers = edges[:-1]

plt.figure(figsize=(7,5))
plt.bar(centers, fraction, width=2, color='darkred')

plt.xlabel('degree')
plt.ylabel('fraction of nodes')

plt.tight_layout()

plt.savefig('scale_free_hist_plot.png', dpi=150)
plt.close()
