from multiprocessing import Pool, cpu_count
import signal
import sys
import pandas as pd
import numpy as np
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


# 1. Define the safe worker initializer to handle KeyboardInterrupts
def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# 2. Package the loop logic into a single function for the workers
def compute_path_stats_for_graph(gml_path):
    batch_num = int(gml_path.name.split('_')[0])
    G = nx.read_graphml(str(gml_path))
    n_nodes = G.number_of_nodes()
    
    # -- Empirical Paths --
    paths = find_unbranching_paths(G)
    if paths:
        lengths = [len(p) for p in paths]
        max_len = max(lengths)
        mean_len = sum(lengths) / len(lengths)
        unique_nodes_in_paths = set(n for p in paths for n in p)
        total_nodes_in_paths = len(unique_nodes_in_paths)
    else:
        max_len, mean_len, total_nodes_in_paths = 0, 0.0, 0
        
    coverage = total_nodes_in_paths / n_nodes if n_nodes > 0 else 0.0
    
    empirical_stats = {
        'batch': batch_num,
        'n_nodes': n_nodes,
        'n_paths': len(paths),
        'max_path_len': max_len,
        'mean_path_len': round(mean_len, 2),
        'coverage': round(coverage, 4),
    }
    
    path_csv_rows_local = []
    for i, path in enumerate(sorted(paths, key=len, reverse=True)):
        path_csv_rows_local.append({
            'batch': batch_num,
            'path_index': i,
            'length': len(path),
            'path': ' -> '.join(path),
        })
        
    # -- Null Models (This runs in parallel now) --
    null_ps = config_null_path_stats(G, n_null=50)
    recomb_ps = recomb_null_path_stats(G, n_null=50)
    
    null_row = {
        'batch': batch_num,
        'n_nodes': n_nodes,
        'null_n_paths_mean': null_ps['n_paths'][0],
        'null_n_paths_std': null_ps['n_paths'][1],
        'null_max_len_mean': null_ps['max_len'][0],
        'null_max_len_std': null_ps['max_len'][1],
        'null_mean_len_mean': null_ps['mean_len'][0],
        'null_mean_len_std': null_ps['mean_len'][1],
        'null_cov_mean': null_ps['coverage'][0] * 100,
        'null_cov_std': null_ps['coverage'][1] * 100,
        'recomb_n_paths_mean': recomb_ps['n_paths'][0],
        'recomb_n_paths_std': recomb_ps['n_paths'][1],
        'recomb_max_len_mean': recomb_ps['max_len'][0],
        'recomb_max_len_std': recomb_ps['max_len'][1],
        'recomb_mean_len_mean': recomb_ps['mean_len'][0],
        'recomb_mean_len_std': recomb_ps['mean_len'][1],
        'recomb_cov_mean': recomb_ps['coverage'][0] * 100,
        'recomb_cov_std': recomb_ps['coverage'][1] * 100,
    }
    
    print(f' batch={batch_num:6d}: {n_nodes:5d} nodes, '
          f'{len(paths):4d} paths, max_len={max_len}, coverage={coverage:.1%}')
          
    return empirical_stats, path_csv_rows_local, null_row

# 3. Main Parallel Execution Block
if __name__ == '__main__':
    all_path_stats = []
    path_csv_rows = []
    null_path_rows = []

    batch_files = sorted(graph_dir.glob('*_domain_architecture_network.graphml'), 
                         key=lambda p: int(p.name.split('_')[0]))
    
    # --- WORKER CAP ADDED HERE ---
    # Change MAX_WORKERS to 4, 6, or 8 depending on your server's RAM. 
    # 4 is usually a very safe sweet spot for graph processing.
    MAX_WORKERS = 4
    num_workers = min(MAX_WORKERS, cpu_count())
    
    print(f"Starting parallel processing of paths with {num_workers} workers...")
    
    try:
        with Pool(num_workers, initializer=init_worker) as pool:
            for emp_stats, path_rows, null_row in pool.imap_unordered(compute_path_stats_for_graph, batch_files):
                all_path_stats.append(emp_stats)
                path_csv_rows.extend(path_rows)
                null_path_rows.append(null_row)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught! Terminating all background workers gracefully...")
        pool.terminate()
        pool.join()
        sys.exit(1)

    # Sort everything back in order since imap_unordered returns as soon as a job finishes
    all_path_stats.sort(key=lambda x: x['batch'])
    null_path_rows.sort(key=lambda x: x['batch'])

    # --- SAVE OUTPUTS ---
    out_dir = Path(os.path.join(os.getcwd(), 'plasmid_graphs_analysis'))
    os.makedirs(out_dir, exist_ok=True)

    paths_csv_path = out_dir / 'unbranching_paths_3.csv'
    pd.DataFrame(path_csv_rows).to_csv(paths_csv_path, index=False)