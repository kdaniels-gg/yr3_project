import polars as pl
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import networkx as nx
from datasketch import MinHash, MinHashLSH
import pickle
import igraph as ig
import leidenalg
import pandas as pd
import random
import os
import sys 

############################################################################################################################
############################################################################################################################
############################################################################################################################
#PROCESS QUALITIES OF NETWORKS FOR DIFFERENT PLASMID BATCH SIZES


#data_dir = Path('/home/kd541/rds/hpc-work/plasmid_motif_network/intermediate')
data_dir = Path(os.path.join(os.getcwd(),'plasmid_motif_network/intermediate'))
files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))

df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()

df_merged = df_merged.with_columns(
    pl.col('strand').cast(pl.Int32).alias('strand')
)


output_path = Path(os.path.join(os.getcwd(), 'plasmid_batched_graphs'))
os.makedirs(output_path, exist_ok=True)

all_plasmids = list(set(df_merged['plasmid']))
                    
domain_df = pd.read_csv('Pfam-A.clans.tsv', sep='\t', header=None)
domain_dict = dict(zip(domain_df[3].tolist(), domain_df[2].tolist()))

random.seed(42)


all_run_stats_F = []
all_run_stats_G = []

sys.stdout = open(os.devnull, 'w')
number_of_runs = list(range(1,11))
for run in number_of_runs:
    random.shuffle(all_plasmids) 
    max_size = len(all_plasmids)
    num_of_batches = 100 
    batch_sizes = np.unique(np.geomspace(1, max_size, num=num_of_batches, dtype=int))
    batch_num_to_plasmids = {}
    for size in batch_sizes:
        batch_num_to_plasmids[size] = all_plasmids[:size]
    F_graph_stats = []
    G_graph_stats = []
    for num, plasmids in batch_num_to_plasmids.items():
        df_filt = df_merged.filter(pl.col('plasmid').is_in(plasmids))
        df = df_filt
        df = df.sort(['plasmid', 'start', 'ali_from'])
        #make ordered list of domain hits per gene/protein
        ordered = df.select(['plasmid','query_name','target_name','start','ali_from','strand'])
        df = ordered
        #make adj
        adjacency = defaultdict(int)
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
            df_shifted.select(['plasmid','domain1','domain2','strand1','strand2']),
            wrap.select(['plasmid','domain1','domain2','strand1','strand2'])
        ])
        df_edges = df_edges.with_columns(
            pl.when((pl.col('strand1') == 1) & (pl.col('strand2') == 1)).then(pl.lit('PP'))
             .when((pl.col('strand1') == -1) & (pl.col('strand2') == -1)).then(pl.lit('MM'))
             .when((pl.col('strand1') == 1) & (pl.col('strand2') == -1)).then(pl.lit('PM'))
             .otherwise(pl.lit('MP'))
             .alias('orientation')
        )
        adj_df = (
            df_edges.group_by(['domain1', 'domain2', 'orientation'])
            .agg(pl.len().cast(pl.Int64).alias('weight'))
        )
        df = adj_df
        df = df.with_columns(pl.when(pl.col('orientation').is_in(['PP', 'MM'])).then(pl.col('weight')).otherwise(-pl.col('weight')).alias('signed_contribution'))
        collapsed = (
            df.group_by(['domain1', 'domain2'])
              .agg([
                  pl.sum('signed_contribution').alias('signed_weight'),
                  pl.sum('weight').alias('total_weight')
              ])
        )
        #make graph for adj
        df = adj_df
        domain_to_plasmids = defaultdict(set)
        for row in ordered.iter_rows(named=True):
            domain_to_plasmids[row['target_name']].add(row['plasmid'])
        #G = nx.DiGraph()
        G= nx.MultiDiGraph()
        for row in df.iter_rows(named=True):
            d1 = row['domain1']
            d2 = row['domain2']
            weight = row['weight']
            orientation = row['orientation']
            G.add_edge(
                d1,
                d2,
                weight=weight,
                orientation=orientation
            )
        for node in G.nodes():
            plasmid_set = domain_to_plasmids.get(node, set())
            G.nodes[node]['label'] = node
            G.nodes[node]['plasmid_count'] = len(plasmid_set)
            G.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
        for global_id, (u, v, k) in enumerate(G.edges(keys=True)):
            G[u][v][k]['id'] = str(global_id)
        nx.write_graphml(G, os.path.join(output_path, f'{num}_domain_architecture_network.graphml'), edge_id_from_attribute='id')
        #make graph for signed adj
        df = collapsed
        F = nx.DiGraph()
        for row in df.iter_rows(named=True):
            d1 = row['domain1']
            d2 = row['domain2']
            signed_weight = row['signed_weight']
            total_weight = row['total_weight']
            F.add_edge(
                d1,
                d2,
                signed_weight=signed_weight,
                total_weight=total_weight
            )
        for node in F.nodes():
            plasmid_set = domain_to_plasmids.get(node, set())
            F.nodes[node]['label'] = node
            F.nodes[node]['plasmid_count'] = len(plasmid_set)
            F.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
        nx.write_graphml(F, os.path.join(output_path, f'{num}_domain_architecture_signed_network.graphml'))
        F_node_number = F.number_of_nodes()
        F_edge_number = F.number_of_edges()
        F_average_degree = (2 * F_edge_number / F_node_number) if F_node_number > 0 else 0.0
        F_graph_stats.append({
            'plasmid_number': num,
            'node_number': F_node_number,
            'edge_number': F_edge_number,
            'average_degree': F_average_degree
        })
        G_node_number = G.number_of_nodes()
        G_edge_number = G.number_of_edges()
        G_average_degree = (2 * G_edge_number / G_node_number) if G_node_number > 0 else 0.0
        G_graph_stats.append({
            'plasmid_number': num,
            'node_number': G_node_number,
            'edge_number': G_edge_number,
            'average_degree': G_average_degree
        })
    all_run_stats_F.append(F_graph_stats)
    all_run_stats_G.append(G_graph_stats)
    

Fdf = pl.DataFrame([row for run in all_run_stats_F for row in run])
avg_F = (Fdf.group_by('plasmid_number').mean().sort('plasmid_number'))

Gdf = pl.DataFrame([row for run in all_run_stats_G for row in run])
avg_G = (Gdf.group_by('plasmid_number').mean().sort('plasmid_number'))
 
sys.stdout.close()
sys.stdout = sys.__stdout__


F_stats_df = pl.DataFrame(avg_F)
F_stats_csv_path = os.path.join(output_path, 'F_graph_statistics.csv')
F_stats_df.write_csv(F_stats_csv_path)
G_stats_df = pl.DataFrame(avg_G)
G_stats_csv_path = os.path.join(output_path, 'G_graph_statistics.csv')
G_stats_df.write_csv(G_stats_csv_path)


############################################################################################################################
############################################################################################################################
############################################################################################################################
#NETWORK PROPERTY PLOTS


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



graph_dir = Path(os.path.join(os.getcwd(), 'plasmid_batched_graphs'))
f_csv = graph_dir / 'F_graph_statistics.csv'
g_csv = graph_dir / 'G_graph_statistics.csv'

fdf = pd.read_csv(f_csv)
gdf = pd.read_csv(g_csv)


batch_x  = fdf['plasmid_number'].tolist() 
f_nodes = fdf['node_number'].tolist()
f_edges = fdf['edge_number'].tolist()
g_edges = gdf['edge_number'].tolist()

complete_edges = [n*(n-1) for n in f_nodes]
complete_degree = [(n-1) for n in f_nodes]
f_density = [o/c if c > 0 else 0 for o, c in zip(f_edges, complete_edges)]
g_density = [o/c if c > 0 else 0 for o, c in zip(g_edges, complete_edges)]
f_degree = [2*e/n if n > 0 else 0 for e, n in zip(f_edges, f_nodes)]
g_degree = [2*e/n if n > 0 else 0 for e, n in zip(g_edges, f_nodes)]


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
fig.suptitle('No. Edges / No. plasmids in batch')

ax1.plot(batch_x, f_edges, '--', label='data', color='blue')
ax1.plot(batch_x, complete_edges, ':', label='fully connected', color='orange')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('No. plasmids in batch')
ax1.set_ylabel('No. edges')
ax1.set_title("Signed architecture network")
ax1.legend() 

ax2.plot(batch_x, g_edges, '--', label='data', color='green')
ax2.plot(batch_x, complete_edges, ':', label='fully connected', color='orange')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('No. plasmids in batch')
ax2.set_ylabel('No. edges')
ax2.set_title("Unsigned architecture network")
ax2.legend() 

plt.savefig('edges2.png')
plt.close()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
fig.suptitle('Avg. node degree / No. plasmids in batch')

ax1.plot(batch_x, f_degree, '--', label='data', color='blue')
ax1.plot(batch_x, complete_degree, ':', label='fully connected', color='orange')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('No. plasmids in batch')
ax1.set_ylabel('Avg. node degree')
ax1.set_title("Signed architecture network")
ax1.legend() 

ax2.plot(batch_x, g_degree, '--', label='data', color='green')
ax2.plot(batch_x, complete_degree, ':', label='fully connected', color='orange')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('No. plasmids in batch')
ax2.set_ylabel('Avg. node degree')
ax2.set_title("Unsigned architecture network")
ax2.legend() 

plt.savefig('degrees2.png')
plt.close()

fig, ax1 = plt.subplots(figsize=(10, 12))
fig.suptitle('Density / No. plasmids in batch')
ax1.plot(batch_x, f_density, '--', label='signed data', color='blue')
ax1.plot(batch_x, g_density, ':', label='unsigned data', color='red')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('No. plasmids in batch')
ax1.set_ylabel('No. Edges / No. possible edges')
ax1.legend() 
yticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
ax1.set_yticks(yticks)
ax1.set_yticklabels([str(y) for y in yticks])

plt.savefig('density2.png')
plt.close()

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


all_path_stats = []
batch_files = sorted(graph_dir.glob('*_domain_architecture_network.graphml'),
                     key=lambda p: int(p.name.split('_')[0]))



path_csv_rows = []

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
    print(f'  batch={batch_num:6d}: {n_nodes:5d} nodes, '
          f'{len(paths):4d} paths, max_len={max_len}, coverage={coverage:.1%}')




out_dir = Path(os.path.join(os.getcwd(), 'plasmid_graphs_analysis'))
os.makedirs(out_dir, exist_ok=True)
paths_csv_path = out_dir / 'unbranching_paths.csv'

path_df = pd.DataFrame.from_dict(path_csv_rows)
path_df.to_csv(paths_csv_path, index=False)

pathstats_csv_path =  out_dir / 'unbranching_paths_stats.csv'
pathstats_df = pd.DataFrame.from_dict(all_path_stats)
pathstats_df['coverage_percentage'] = [x*100 for x in pathstats_df['coverage']]

pathstats_df.to_csv(pathstats_csv_path, index=False)

fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(10, 70))
fig.suptitle('Unbranching paths statistics')

batch_n = pathstats_df['batch'].to_numpy()
n_nodes2 = pathstats_df['n_nodes'].to_numpy()  
n_paths = pathstats_df['n_paths'].to_numpy()    
max_pathlen = pathstats_df['max_path_len'].to_numpy()
mean_pathlen = pathstats_df['mean_path_len'].to_numpy()
path_coverage = pathstats_df['coverage_percentage'].to_numpy()  
 

ax1.plot(batch_n, n_paths, '--', label='data', color='blue')
ax1.set_xlabel('Plasmid No. in batch')
ax1.set_ylabel('No. unbranching paths')
ax1.set_title('No. paths with increasing batch size')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend() 

ax2.plot(batch_n, n_paths/n_nodes2, '--', label='data', color='blue')
ax2.set_xlabel('Plasmid No. in batch')
ax2.set_ylabel('No. unbranching paths relative to node number')
ax2.set_title('No. paths relative to No. nodes with increasing batch size')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend() 

ax3.plot(batch_n, max_pathlen, '--', label='data', color='blue')
ax3.set_xlabel('Plasmid No. in batch')
ax3.set_ylabel('Max length of unbranching paths')
ax3.set_title('Max length of unbranching paths with increasing batch size')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.legend() 

ax4.plot(batch_n, max_pathlen/n_nodes2, '--', label='data', color='blue')
ax4.set_xlabel('Plasmid No. in batch')
ax4.set_ylabel('Max length of unbranching paths relative to No. nodes')
ax4.set_title('Max path length relative to No. nodes with increasing batch size')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.legend() 

ax5.plot(batch_n, mean_pathlen, '--', label='data', color='blue')
ax5.set_xlabel('Plasmid No. in batch')
ax5.set_ylabel('Mean length of unbranching paths')
ax5.set_title('Mean length of unbranching paths with increasing batch size')
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.legend() 

ax6.plot(batch_n, mean_pathlen/n_nodes2, '--', label='data', color='blue')
ax6.set_xlabel('Plasmid No. in batch')
ax6.set_ylabel('Mean length of unbranching paths relative to No. nodes')
ax6.set_title('Mean path length relative to No. nodes with increasing batch size')
ax6.set_xscale('log')
ax6.set_yscale('log')
ax6.legend() 

ax7.plot(batch_n, path_coverage, '--', label='data', color='blue')
ax7.set_xlabel('Plasmid No. in batch')
ax7.set_ylabel('Coverage of unbranching paths')
ax7.set_title('Coverage of unbranching paths with increasing batch size')
ax7.set_xscale('log')
ax7.set_yscale('log')
ax7.legend() 

plt.savefig('paths2.png')
plt.close()

############################################################################################################################
############################################################################################################################
############################################################################################################################
#REPEAT ANALYSIS BUT FOR PLASMIDS GROUPED BY LOCATION, PATHOGEN SPECIES, AND PATHOGEN STRAIN

from collections import Counter
from sklearn.cluster import DBSCAN
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

plasmid_files_path = Path(os.path.join(os.getcwd(), 'plasmids'))

plasmid_files = os.listdir(plasmid_files_path)
plasmid_nuccore_list = [''.join(x.split('.fa')[:-1]) for x in plasmid_files]

plsdb_meta_path = Path(os.path.join(os.getcwd(), 'plsdb_meta'))
plsdb_meta_files = os.listdir(plsdb_meta_path)


#general linker 'nuccore.csv'

#typing 'typing.csv'

#biosample 'biosample.csv'

#taxonomy 'taxonomy.csv'


nucpath = plsdb_meta_path / 'nuccore.csv'
nucout_path = plsdb_meta_path / f'nuccore_only.csv'

try:
    chunks = pd.read_csv(nucpath, sep=',', low_memory=False, chunksize=50_000)
    first = True
    total_in, total_out = 0, 0
    for chunk in chunks:
        if 'NUCCORE_ACC' not in chunk.columns:
            print(f'NUCCORE_ACC not found in nuccore.csv, skipping')
            break
        total_in += len(chunk)
        filt = chunk.loc[chunk['NUCCORE_ACC'].isin(plasmid_nuccore_list),]
        total_out += len(filt)
        filt.to_csv(nucout_path, sep=',', index=False, mode='w' if first else 'a', header=first)
        first = False
    print(f'nuccore_only.csv: {total_in} -> {total_out} rows')
except Exception as e:
    print(f'Error processing nuccore.csv: {e}')



nuc_df = pd.read_csv(nucout_path)
nuc_tax = dict(zip(nuc_df['NUCCORE_ACC'].tolist(), nuc_df['TAXONOMY_UID'].tolist()))
nuc_bio = dict(zip(nuc_df['NUCCORE_ACC'].tolist(), nuc_df['BIOSAMPLE_UID'].tolist()))



typpath = plsdb_meta_path / 'typing.csv'
typout_path = plsdb_meta_path / f'typing_only.csv'

try:
    chunks = pd.read_csv(typpath, sep=',', low_memory=False, chunksize=50_000)
    first = True
    total_in, total_out = 0, 0
    for chunk in chunks:
        if 'NUCCORE_ACC' not in chunk.columns:
            print(f'NUCCORE_ACC not found in typing.csv, skipping')
            break
        total_in += len(chunk)
        filt = chunk.loc[chunk['NUCCORE_ACC'].isin(plasmid_nuccore_list),]
        total_out += len(filt)
        filt.to_csv(typout_path, sep=',', index=False, mode='w' if first else 'a', header=first)
        first = False
    print(f'typing_only.csv: {total_in} -> {total_out} rows')
except Exception as e:
    print(f'Error processing typing.csv: {e}')


#Link plasmids to taxonomy; family, species, strain
#Link plasmids to location and time of collection where possible
#Link plasmids to pathogenicity status
#Link plasmids to MOBSUITE element

typpath = plsdb_meta_path / 'typing_only.csv'
biopath = plsdb_meta_path / 'biosample.csv'
taxpath = plsdb_meta_path / 'taxonomy.csv'

typ_df = pd.read_csv(typpath)
bio_df = pd.read_csv(biopath, low_memory=False)
tax_df = pd.read_csv(taxpath)

nuc_mob = dict(zip(typ_df['NUCCORE_ACC'].tolist(), typ_df['predicted_mobility'].tolist()))

bio_loc = dict(zip(bio_df['BIOSAMPLE_UID'].tolist(), bio_df['LOCATION_name'].tolist()))
bio_lat = dict(zip(bio_df['BIOSAMPLE_UID'].tolist(), bio_df['LOCATION_lat'].tolist()))
bio_lng = dict(zip(bio_df['BIOSAMPLE_UID'].tolist(), bio_df['LOCATION_lng'].tolist()))
bio_pth = dict(zip(bio_df['BIOSAMPLE_UID'].tolist(), bio_df['BIOSAMPLE_pathogenicity'].tolist()))


nuc_loc = {k:bio_loc.get(v) for k,v in nuc_bio.items()}
nuc_lat = {k:bio_lat.get(v) for k,v in nuc_bio.items()}
nuc_lng = {k:bio_lng.get(v) for k,v in nuc_bio.items()}
#nuc_pth = {k:bio_pth.get(v) for k,v in nuc_bio.items()} 

#known_pathogen_classifications =  [ 'Enterotoxins', 'animal', 'human', 'acute cholangitis', 'sepsis', 'Bacteremia', 'persistent bacteremia', 'stomachache, vomit, fever, watery and bloody diarrhea, and HUS']
#known_pathogens = {k:v for k, v in nuc_pth.items() if v in known_pathogen_classifications}
#Very few

#test = list(nuc_lat.keys())[:50]
#test2 = [nuc_loc.get(x) for x in test]
#test3 = [nuc_lat.get(x) for x in test]
#test4 = [nuc_lng.get(x) for x in test]
#test5 = {k:v for k, v in nuc_loc.items() if type(v) == str}


#cluster plasids by location

valid_nuc = [(k, nuc_lat[k], nuc_lng[k]) 
             for k in nuc_lat 
             if nuc_lat[k] is not None 
             and not (isinstance(nuc_lat[k], float) and np.isnan(nuc_lat[k]))
             and nuc_lng[k] is not None
             and not (isinstance(nuc_lng[k], float) and np.isnan(nuc_lng[k]))]

nucs, lats, lngs = zip(*valid_nuc)
coords = np.radians(np.array(list(zip(lats, lngs))))

# eps in radians: 500km / 6371km earth radius
eps_km = 500
db = DBSCAN(eps=eps_km/6371, min_samples=5, algorithm='ball_tree', metric='haversine')
labels = db.fit_predict(coords)

nuc_cluster = dict(zip(nucs, labels))

print(Counter(labels))


#note a lot of these are default USA values
#USA_DEFAULT = (39.7837304, -100.4458825)
#nuc_is_default = {k: (round(nuc_lat[k], 4), round(nuc_lng[k], 4)) == (39.7837, -100.4459) 
#                  for k in nuc_lat if nuc_lat.get(k) is not None}'


#Now cluster by taxonomy


tax_spc = dict(zip(tax_df['TAXONOMY_UID'].tolist(), tax_df['TAXONOMY_species'].tolist()))
nuc_spc = {k:tax_spc.get(v) for k, v in nuc_tax.items()}

tax_str = dict(zip(tax_df['TAXONOMY_UID'].tolist(), tax_df['TAXONOMY_strain'].tolist()))
tax_str = {k:v for k, v in tax_str.items() if type(v) == str}
nuc_str = {k:tax_str.get(v) for k, v in nuc_tax.items()}
nuc_str = {k:v for k, v in nuc_str.items() if type(v) == str}


#for shared species 
all_spec = list(set(list(nuc_spc.values())))
spc_inc = dict(sorted(dict(Counter(list(nuc_spc.values()))).items(), key=lambda item: item[1]))

nuc_spc_oi = {k:v for k,v in nuc_spc.items() if spc_inc.get(v) > 1}

species_analysis_path = Path(os.path.join(os.getcwd(), 'species_specific_plasmid_analysis'))
os.makedirs(species_analysis_path, exist_ok=True)

spec_oi = list(set(list(nuc_spc_oi.values())))


###################################################################################################################
#first just look at overall graphs

sys.stdout = open(os.devnull, 'w')
for species in spec_oi:
    plasmids_of_species = [k for k,v in nuc_spc_oi.items() if v == species]
    number_of_loc_plasmids = len(plasmids_of_species)
    df_filt = df_merged.filter(pl.col('plasmid').is_in(plasmids_of_species))
    df = df_filt
    df = df_filt.sort(['plasmid', 'start', 'ali_from']).select(['plasmid','query_name','target_name','start','ali_from','strand'])
    ordered = df.select(['plasmid','query_name','target_name','start','ali_from','strand'])
    adjacency = defaultdict(int)
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
        df_shifted.select(['plasmid','domain1','domain2','strand1','strand2']),
        wrap.select(['plasmid','domain1','domain2','strand1','strand2'])
    ])
    df_edges = df_edges.with_columns(
        pl.when((pl.col('strand1') == 1) & (pl.col('strand2') == 1)).then(pl.lit('PP'))
         .when((pl.col('strand1') == -1) & (pl.col('strand2') == -1)).then(pl.lit('MM'))
         .when((pl.col('strand1') == 1) & (pl.col('strand2') == -1)).then(pl.lit('PM'))
         .otherwise(pl.lit('MP'))
         .alias('orientation')
    )
    adj_df = (
        df_edges.group_by(['domain1', 'domain2', 'orientation'])
        .agg(pl.len().cast(pl.Int64).alias('weight'))
    )
    df = adj_df.with_columns(pl.when(pl.col('orientation').is_in(['PP', 'MM'])).then(pl.col('weight')).otherwise(-pl.col('weight')).alias('signed_contribution'))
    collapsed = (
        df.group_by(['domain1', 'domain2'])
          .agg([
              pl.sum('signed_contribution').alias('signed_weight'),
              pl.sum('weight').alias('total_weight')
          ])
    )
    domain_to_plasmids = defaultdict(set)
    for row in ordered.iter_rows(named=True):
        domain_to_plasmids[row['target_name']].add(row['plasmid'])
    G= nx.MultiDiGraph()
    for row in adj_df.iter_rows(named=True):
        d1 = row['domain1']
        d2 = row['domain2']
        weight = row['weight']
        orientation = row['orientation']
        G.add_edge(
            d1,
            d2,
            weight=weight,
            orientation=orientation
        )
    for node in G.nodes():
        plasmid_set = domain_to_plasmids.get(node, set())
        G.nodes[node]['label'] = node
        G.nodes[node]['plasmid_count'] = len(plasmid_set)
        G.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
    for global_id, (u, v, k) in enumerate(G.edges(keys=True)):
        G[u][v][k]['id'] = str(global_id)
    nx.write_graphml(G, os.path.join(species_analysis_path, f'{species}_domain_architecture_network_{number_of_loc_plasmids}.graphml'), edge_id_from_attribute='id')
    F = nx.DiGraph()
    for row in collapsed.iter_rows(named=True):
        d1 = row['domain1']
        d2 = row['domain2']
        signed_weight = row['signed_weight']
        total_weight = row['total_weight']
        F.add_edge(
            d1,
            d2,
            signed_weight=signed_weight,
            total_weight=total_weight
        )
    for node in F.nodes():
        plasmid_set = domain_to_plasmids.get(node, set())
        F.nodes[node]['label'] = node
        F.nodes[node]['plasmid_count'] = len(plasmid_set)
        F.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
    nx.write_graphml(F, os.path.join(species_analysis_path, f'{species}_domain_architecture_signed_network_{number_of_loc_plasmids}.graphml'))


sys.stdout.close()
sys.stdout = sys.__stdout__


###################################################################################################################

big_species_analysis_path = Path(os.path.join(species_analysis_path, 'big_species'))
os.makedirs(big_species_analysis_path, exist_ok=True)


big_species = ['Escherichia_coli', 'Klebsiella_pneumoniae', 'Salmonella_enterica', 'Staphylococcus_aureus']

sys.stdout = open(os.devnull, 'w')
for species in big_species:
    big_species_analysis_path_current = Path(os.path.join(big_species_analysis_path, f'{species}'))
    os.makedirs(big_species_analysis_path_current, exist_ok=True)
    all_run_stats_F = []
    all_run_stats_G = []
    plasmids_of_species = [k for k,v in nuc_spc_oi.items() if v == species]
    number_of_loc_plasmids = len(plasmids_of_species)
    number_of_runs = list(range(1,11))
    for run in number_of_runs:
        random.shuffle(plasmids_of_species) 
        max_size = len(plasmids_of_species)
        num_of_batches = 100 
        batch_sizes = np.unique(np.geomspace(1, max_size, num=num_of_batches, dtype=int))
        batch_num_to_plasmids = {}
        for size in batch_sizes:
            batch_num_to_plasmids[size] = plasmids_of_species[:size]
        F_graph_stats = []
        G_graph_stats = []
        for num, plasmids in batch_num_to_plasmids.items():
            df_filt = df_merged.filter(pl.col('plasmid').is_in(plasmids))
            df = df_filt
            df = df.sort(['plasmid', 'start', 'ali_from'])
            ordered = df.select(['plasmid','query_name','target_name','start','ali_from','strand'])
            df = ordered
            adjacency = defaultdict(int)
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
                df_shifted.select(['plasmid','domain1','domain2','strand1','strand2']),
                wrap.select(['plasmid','domain1','domain2','strand1','strand2'])
            ])
            df_edges = df_edges.with_columns(
                pl.when((pl.col('strand1') == 1) & (pl.col('strand2') == 1)).then(pl.lit('PP'))
                 .when((pl.col('strand1') == -1) & (pl.col('strand2') == -1)).then(pl.lit('MM'))
                 .when((pl.col('strand1') == 1) & (pl.col('strand2') == -1)).then(pl.lit('PM'))
                 .otherwise(pl.lit('MP'))
                 .alias('orientation')
            )
            adj_df = (
                df_edges.group_by(['domain1', 'domain2', 'orientation'])
                .agg(pl.len().cast(pl.Int64).alias('weight'))
            )
            df = adj_df
            df = df.with_columns(pl.when(pl.col('orientation').is_in(['PP', 'MM'])).then(pl.col('weight')).otherwise(-pl.col('weight')).alias('signed_contribution'))
            collapsed = (
                df.group_by(['domain1', 'domain2'])
                  .agg([
                      pl.sum('signed_contribution').alias('signed_weight'),
                      pl.sum('weight').alias('total_weight')
                  ])
            )
            df = adj_df
            domain_to_plasmids = defaultdict(set)
            for row in ordered.iter_rows(named=True):
                domain_to_plasmids[row['target_name']].add(row['plasmid'])
            G= nx.MultiDiGraph()
            for row in df.iter_rows(named=True):
                d1 = row['domain1']
                d2 = row['domain2']
                weight = row['weight']
                orientation = row['orientation']
                G.add_edge(
                    d1,
                    d2,
                    weight=weight,
                    orientation=orientation
                )
            for node in G.nodes():
                plasmid_set = domain_to_plasmids.get(node, set())
                G.nodes[node]['label'] = node
                G.nodes[node]['plasmid_count'] = len(plasmid_set)
                G.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
            for global_id, (u, v, k) in enumerate(G.edges(keys=True)):
                G[u][v][k]['id'] = str(global_id)
            nx.write_graphml(G, os.path.join(big_species_analysis_path_current, f'batch_{len(plasmids)}_{species}_domain_architecture_network.graphml'), edge_id_from_attribute='id')
            df = collapsed
            F = nx.DiGraph()
            for row in df.iter_rows(named=True):
                d1 = row['domain1']
                d2 = row['domain2']
                signed_weight = row['signed_weight']
                total_weight = row['total_weight']
                F.add_edge(
                    d1,
                    d2,
                    signed_weight=signed_weight,
                    total_weight=total_weight
                )
            for node in F.nodes():
                plasmid_set = domain_to_plasmids.get(node, set())
                F.nodes[node]['label'] = node
                F.nodes[node]['plasmid_count'] = len(plasmid_set)
                F.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
            nx.write_graphml(F, os.path.join(big_species_analysis_path_current, f'batch_{len(plasmids)}_{species}_domain_architecture_signed_network.graphml'), edge_id_from_attribute='id')
            F_node_number = F.number_of_nodes()
            F_edge_number = F.number_of_edges()
            F_average_degree = (2 * F_edge_number / F_node_number) if F_node_number > 0 else 0.0
            F_graph_stats.append({
                'plasmid_number': num,
                'node_number': F_node_number,
                'edge_number': F_edge_number,
                'average_degree': F_average_degree
            })
            G_node_number = G.number_of_nodes()
            G_edge_number = G.number_of_edges()
            G_average_degree = (2 * G_edge_number / G_node_number) if G_node_number > 0 else 0.0
            G_graph_stats.append({
                'plasmid_number': num,
                'node_number': G_node_number,
                'edge_number': G_edge_number,
                'average_degree': G_average_degree
            })
        all_run_stats_F.append(F_graph_stats)
        all_run_stats_G.append(G_graph_stats)
    Fdf = pl.DataFrame([row for run in all_run_stats_F for row in run])
    avg_F = (Fdf.group_by('plasmid_number').mean().sort('plasmid_number'))
    Gdf = pl.DataFrame([row for run in all_run_stats_G for row in run])
    avg_G = (Gdf.group_by('plasmid_number').mean().sort('plasmid_number'))
    F_stats_df = pl.DataFrame(avg_F)
    F_stats_csv_path = os.path.join(big_species_analysis_path_current, 'F_graph_statistics.csv')
    F_stats_df.write_csv(F_stats_csv_path)
    G_stats_df = pl.DataFrame(avg_G)
    G_stats_csv_path = os.path.join(big_species_analysis_path_current, 'G_graph_statistics.csv')
    G_stats_df.write_csv(G_stats_csv_path)

sys.stdout.close()
sys.stdout = sys.__stdout__



def find_unbranching_paths(G):
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



for species in big_species:
    big_species_analysis_path_current = Path(os.path.join(big_species_analysis_path, f'{species}'))
    f_csv,g_csv = big_species_analysis_path_current / 'F_graph_statistics.csv', big_species_analysis_path_current / 'G_graph_statistics.csv'
    fdf, gdf = pd.read_csv(f_csv), pd.read_csv(g_csv)
    batch_x, f_nodes  = fdf['plasmid_number'].tolist(), fdf['node_number'].tolist()
    f_edges, g_edges = fdf['edge_number'].tolist(), gdf['edge_number'].tolist()
    complete_edges, complete_degree = [n*(n-1) for n in f_nodes], [(n-1) for n in f_nodes]
    f_density, g_density = [o/c if c > 0 else 0 for o, c in zip(f_edges, complete_edges)], [o/c if c > 0 else 0 for o, c in zip(g_edges, complete_edges)]
    f_degree, g_degree = [2*e/n if n > 0 else 0 for e, n in zip(f_edges, f_nodes)], [2*e/n if n > 0 else 0 for e, n in zip(g_edges, f_nodes)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    fig.suptitle('No. Edges / No. plasmids in batch')
    ax1.plot(batch_x, f_edges, '--', label='data', color='blue')
    ax1.plot(batch_x, complete_edges, ':', label='fully connected', color='orange')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('No. plasmids in batch')
    ax1.set_ylabel('No. edges')
    ax1.set_title("Signed architecture network")
    ax1.legend() 
    ax2.plot(batch_x, g_edges, '--', label='data', color='green')
    ax2.plot(batch_x, complete_edges, ':', label='fully connected', color='orange')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('No. plasmids in batch')
    ax2.set_ylabel('No. edges')
    ax2.set_title("Unsigned architecture network")
    ax2.legend() 
    edges_out = os.path.join(big_species_analysis_path_current, f'{species}_edges.png')
    plt.savefig(edges_out)
    plt.close()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    fig.suptitle('Avg. node degree / No. plasmids in batch')
    ax1.plot(batch_x, f_degree, '--', label='data', color='blue')
    ax1.plot(batch_x, complete_degree, ':', label='fully connected', color='orange')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('No. plasmids in batch')
    ax1.set_ylabel('Avg. node degree')
    ax1.set_title("Signed architecture network")
    ax1.legend() 
    ax2.plot(batch_x, g_degree, '--', label='data', color='green')
    ax2.plot(batch_x, complete_degree, ':', label='fully connected', color='orange')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('No. plasmids in batch')
    ax2.set_ylabel('Avg. node degree')
    ax2.set_title("Unsigned architecture network")
    ax2.legend() 
    degrees_out = os.path.join(big_species_analysis_path_current, f'{species}_degrees.png')
    plt.savefig(degrees_out)
    plt.close()
    fig, ax1 = plt.subplots(figsize=(10, 12))
    fig.suptitle('Density / No. plasmids in batch')
    ax1.plot(batch_x, f_density, '--', label='signed data', color='blue')
    ax1.plot(batch_x, g_density, ':', label='unsigned data', color='red')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('No. plasmids in batch')
    ax1.set_ylabel('No. Edges / No. possible edges')
    ax1.legend() 
    yticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([str(y) for y in yticks])
    density_out = os.path.join(big_species_analysis_path_current, f'{species}_density.png')
    plt.savefig(density_out)
    plt.close()


#batch_{len(plasmids)}_{species}_domain_architecture_signed_network.graphml


for species in big_species:
    all_path_stats = []
    big_species_analysis_path_current = Path(os.path.join(big_species_analysis_path, f'{species}'))
    batch_files = sorted(big_species_analysis_path_current.glob(f'batch_*_{species}_domain_architecture_network.graphml'),
                         key=lambda p: int(p.name.split('_')[1]))
    path_csv_rows = []
    for gml_path in batch_files:
        batch_num = int(gml_path.name.split('_')[1])
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
        print(f'  batch={batch_num:6d}: {n_nodes:5d} nodes, '
              f'{len(paths):4d} paths, max_len={max_len}, coverage={coverage:.1%}')
    paths_csv_path = big_species_analysis_path_current / 'unbranching_paths.csv'
    path_df = pd.DataFrame.from_dict(path_csv_rows)
    path_df.to_csv(paths_csv_path, index=False)
    pathstats_csv_path =  big_species_analysis_path_current / 'unbranching_paths_stats.csv'
    pathstats_df = pd.DataFrame.from_dict(all_path_stats)
    pathstats_df['coverage_percentage'] = [x*100 for x in pathstats_df['coverage']]
    pathstats_df.to_csv(pathstats_csv_path, index=False)
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(10, 70))
    fig.suptitle('Unbranching paths statistics')
    batch_n = pathstats_df['batch'].to_numpy()
    n_nodes2 = pathstats_df['n_nodes'].to_numpy()  
    n_paths = pathstats_df['n_paths'].to_numpy()    
    max_pathlen = pathstats_df['max_path_len'].to_numpy()
    mean_pathlen = pathstats_df['mean_path_len'].to_numpy()
    path_coverage = pathstats_df['coverage_percentage'].to_numpy()  
    ax1.plot(batch_n, n_paths, '--', label='data', color='blue')
    ax1.set_xlabel('Plasmid No. in batch')
    ax1.set_ylabel('No. unbranching paths')
    ax1.set_title('No. paths with increasing batch size')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend() 
    ax2.plot(batch_n, n_paths/n_nodes2, '--', label='data', color='blue')
    ax2.set_xlabel('Plasmid No. in batch')
    ax2.set_ylabel('No. unbranching paths relative to node number')
    ax2.set_title('No. paths relative to No. nodes with increasing batch size')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend() 
    ax3.plot(batch_n, max_pathlen, '--', label='data', color='blue')
    ax3.set_xlabel('Plasmid No. in batch')
    ax3.set_ylabel('Max length of unbranching paths')
    ax3.set_title('Max length of unbranching paths with increasing batch size')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend() 
    ax4.plot(batch_n, max_pathlen/n_nodes2, '--', label='data', color='blue')
    ax4.set_xlabel('Plasmid No. in batch')
    ax4.set_ylabel('Max length of unbranching paths relative to No. nodes')
    ax4.set_title('Max path length relative to No. nodes with increasing batch size')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend() 
    ax5.plot(batch_n, mean_pathlen, '--', label='data', color='blue')
    ax5.set_xlabel('Plasmid No. in batch')
    ax5.set_ylabel('Mean length of unbranching paths')
    ax5.set_title('Mean length of unbranching paths with increasing batch size')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.legend() 
    ax6.plot(batch_n, mean_pathlen/n_nodes2, '--', label='data', color='blue')
    ax6.set_xlabel('Plasmid No. in batch')
    ax6.set_ylabel('Mean length of unbranching paths relative to No. nodes')
    ax6.set_title('Mean path length relative to No. nodes with increasing batch size')
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.legend() 
    ax7.plot(batch_n, path_coverage, '--', label='data', color='blue')
    ax7.set_xlabel('Plasmid No. in batch')
    ax7.set_ylabel('Coverage of unbranching paths')
    ax7.set_title('Coverage of unbranching paths with increasing batch size')
    ax7.set_xscale('log')
    ax7.set_yscale('log')
    ax7.legend() 
    paths_out = Path(os.path.join(big_species_analysis_path_current, f'{species}_paths.png'))
    plt.savefig(paths_out)
    plt.close()



###################################################################################################################
######################### 
# for location 
#nuc_cluster

loc_cluster = list(set(list(nuc_cluster.values())))
loc_inc = dict(sorted(dict(Counter(list(nuc_cluster.values()))).items(), key=lambda item: item[1]))

#loc_clusters = list(set(list(nuc_loc.values())))
#loc_inc = dict(sorted(dict(Counter(list(nuc_loc.values()))).items(), key=lambda item: item[1]))
#loc_inc = {k:v for k, v in loc_inc.items() if type(k) == str}
#nuc_loc = {k:v for k, v in nuc_loc.items() if type(v) == str}



nuc_loc_oi = {k:v for k,v in nuc_cluster.items() if loc_inc.get(v) > 1}
nuc_loc_oi = {k:f'cluster_{str(v)}' for k, v in nuc_loc_oi.items()}

loc_oi = list(set(list(nuc_loc_oi.values())))

location_analysis_path = Path(os.path.join(os.getcwd(), 'location_specific_plasmid_analysis'))
os.makedirs(location_analysis_path, exist_ok=True)




###################################################################################################################
#first just look at overall graphs

sys.stdout = open(os.devnull, 'w')
for location in loc_oi:
    plasmids_of_location = [k for k,v in nuc_loc_oi.items() if v == location]
    number_of_loc_plasmids = len(plasmids_of_location)
    df_filt = df_merged.filter(pl.col('plasmid').is_in(plasmids_of_location))
    df = df_filt
    df = df_filt.sort(['plasmid', 'start', 'ali_from']).select(['plasmid','query_name','target_name','start','ali_from','strand'])
    ordered = df.select(['plasmid','query_name','target_name','start','ali_from','strand'])
    adjacency = defaultdict(int)
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
        df_shifted.select(['plasmid','domain1','domain2','strand1','strand2']),
        wrap.select(['plasmid','domain1','domain2','strand1','strand2'])
    ])
    df_edges = df_edges.with_columns(
        pl.when((pl.col('strand1') == 1) & (pl.col('strand2') == 1)).then(pl.lit('PP'))
         .when((pl.col('strand1') == -1) & (pl.col('strand2') == -1)).then(pl.lit('MM'))
         .when((pl.col('strand1') == 1) & (pl.col('strand2') == -1)).then(pl.lit('PM'))
         .otherwise(pl.lit('MP'))
         .alias('orientation')
    )
    adj_df = (
        df_edges.group_by(['domain1', 'domain2', 'orientation'])
        .agg(pl.len().cast(pl.Int64).alias('weight'))
    )
    df = adj_df.with_columns(pl.when(pl.col('orientation').is_in(['PP', 'MM'])).then(pl.col('weight')).otherwise(-pl.col('weight')).alias('signed_contribution'))
    collapsed = (
        df.group_by(['domain1', 'domain2'])
          .agg([
              pl.sum('signed_contribution').alias('signed_weight'),
              pl.sum('weight').alias('total_weight')
          ])
    )
    domain_to_plasmids = defaultdict(set)
    for row in ordered.iter_rows(named=True):
        domain_to_plasmids[row['target_name']].add(row['plasmid'])
    G= nx.MultiDiGraph()
    for row in adj_df.iter_rows(named=True):
        d1 = row['domain1']
        d2 = row['domain2']
        weight = row['weight']
        orientation = row['orientation']
        G.add_edge(
            d1,
            d2,
            weight=weight,
            orientation=orientation
        )
    for node in G.nodes():
        plasmid_set = domain_to_plasmids.get(node, set())
        G.nodes[node]['label'] = node
        G.nodes[node]['plasmid_count'] = len(plasmid_set)
        G.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
    for global_id, (u, v, k) in enumerate(G.edges(keys=True)):
        G[u][v][k]['id'] = str(global_id)
    nx.write_graphml(G, os.path.join(location_analysis_path, f'{location}_domain_architecture_network_{number_of_loc_plasmids}.graphml'), edge_id_from_attribute='id')
    F = nx.DiGraph()
    for row in collapsed.iter_rows(named=True):
        d1 = row['domain1']
        d2 = row['domain2']
        signed_weight = row['signed_weight']
        total_weight = row['total_weight']
        F.add_edge(
            d1,
            d2,
            signed_weight=signed_weight,
            total_weight=total_weight
        )
    for node in F.nodes():
        plasmid_set = domain_to_plasmids.get(node, set())
        F.nodes[node]['label'] = node
        F.nodes[node]['plasmid_count'] = len(plasmid_set)
        F.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
    nx.write_graphml(F, os.path.join(location_analysis_path, f'{location}_domain_architecture_signed_network_{number_of_loc_plasmids}.graphml'))


sys.stdout.close()
sys.stdout = sys.__stdout__


###################################################################################################################

big_location_analysis_path = Path(os.path.join(location_analysis_path, 'big_location'))
os.makedirs(big_location_analysis_path, exist_ok=True)


big_location = [f'cluster_{x}' for x in list(loc_inc.keys())[-5:]]

sys.stdout = open(os.devnull, 'w')
for location in big_location:
    big_location_analysis_path_current = Path(os.path.join(big_location_analysis_path, f'{location}'))
    os.makedirs(big_location_analysis_path_current, exist_ok=True)
    all_run_stats_F = []
    all_run_stats_G = []
    plasmids_of_location = [k for k,v in nuc_loc_oi.items() if v == location]
    number_of_loc_plasmids = len(plasmids_of_location)
    number_of_runs = list(range(1,11))
    for run in number_of_runs:
        random.shuffle(plasmids_of_location) 
        max_size = len(plasmids_of_location)
        num_of_batches = 100 
        batch_sizes = np.unique(np.geomspace(1, max_size, num=num_of_batches, dtype=int))
        batch_num_to_plasmids = {}
        for size in batch_sizes:
            batch_num_to_plasmids[size] = plasmids_of_location[:size]
        F_graph_stats = []
        G_graph_stats = []
        for num, plasmids in batch_num_to_plasmids.items():
            df_filt = df_merged.filter(pl.col('plasmid').is_in(plasmids))
            df = df_filt
            df = df.sort(['plasmid', 'start', 'ali_from'])
            ordered = df.select(['plasmid','query_name','target_name','start','ali_from','strand'])
            df = ordered
            adjacency = defaultdict(int)
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
                df_shifted.select(['plasmid','domain1','domain2','strand1','strand2']),
                wrap.select(['plasmid','domain1','domain2','strand1','strand2'])
            ])
            df_edges = df_edges.with_columns(
                pl.when((pl.col('strand1') == 1) & (pl.col('strand2') == 1)).then(pl.lit('PP'))
                 .when((pl.col('strand1') == -1) & (pl.col('strand2') == -1)).then(pl.lit('MM'))
                 .when((pl.col('strand1') == 1) & (pl.col('strand2') == -1)).then(pl.lit('PM'))
                 .otherwise(pl.lit('MP'))
                 .alias('orientation')
            )
            adj_df = (
                df_edges.group_by(['domain1', 'domain2', 'orientation'])
                .agg(pl.len().cast(pl.Int64).alias('weight'))
            )
            df = adj_df
            df = df.with_columns(pl.when(pl.col('orientation').is_in(['PP', 'MM'])).then(pl.col('weight')).otherwise(-pl.col('weight')).alias('signed_contribution'))
            collapsed = (
                df.group_by(['domain1', 'domain2'])
                  .agg([
                      pl.sum('signed_contribution').alias('signed_weight'),
                      pl.sum('weight').alias('total_weight')
                  ])
            )
            df = adj_df
            domain_to_plasmids = defaultdict(set)
            for row in ordered.iter_rows(named=True):
                domain_to_plasmids[row['target_name']].add(row['plasmid'])
            G= nx.MultiDiGraph()
            for row in df.iter_rows(named=True):
                d1 = row['domain1']
                d2 = row['domain2']
                weight = row['weight']
                orientation = row['orientation']
                G.add_edge(
                    d1,
                    d2,
                    weight=weight,
                    orientation=orientation
                )
            for node in G.nodes():
                plasmid_set = domain_to_plasmids.get(node, set())
                G.nodes[node]['label'] = node
                G.nodes[node]['plasmid_count'] = len(plasmid_set)
                G.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
            for global_id, (u, v, k) in enumerate(G.edges(keys=True)):
                G[u][v][k]['id'] = str(global_id)
            nx.write_graphml(G, os.path.join(big_location_analysis_path_current, f'batch_{len(plasmids)}_{location}_domain_architecture_network.graphml'), edge_id_from_attribute='id')
            df = collapsed
            F = nx.DiGraph()
            for row in df.iter_rows(named=True):
                d1 = row['domain1']
                d2 = row['domain2']
                signed_weight = row['signed_weight']
                total_weight = row['total_weight']
                F.add_edge(
                    d1,
                    d2,
                    signed_weight=signed_weight,
                    total_weight=total_weight
                )
            for node in F.nodes():
                plasmid_set = domain_to_plasmids.get(node, set())
                F.nodes[node]['label'] = node
                F.nodes[node]['plasmid_count'] = len(plasmid_set)
                F.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
            nx.write_graphml(F, os.path.join(big_location_analysis_path_current, f'batch_{len(plasmids)}_{location}_domain_architecture_signed_network.graphml'), edge_id_from_attribute='id')
            F_node_number = F.number_of_nodes()
            F_edge_number = F.number_of_edges()
            F_average_degree = (2 * F_edge_number / F_node_number) if F_node_number > 0 else 0.0
            F_graph_stats.append({
                'plasmid_number': num,
                'node_number': F_node_number,
                'edge_number': F_edge_number,
                'average_degree': F_average_degree
            })
            G_node_number = G.number_of_nodes()
            G_edge_number = G.number_of_edges()
            G_average_degree = (2 * G_edge_number / G_node_number) if G_node_number > 0 else 0.0
            G_graph_stats.append({
                'plasmid_number': num,
                'node_number': G_node_number,
                'edge_number': G_edge_number,
                'average_degree': G_average_degree
            })
        all_run_stats_F.append(F_graph_stats)
        all_run_stats_G.append(G_graph_stats)
    Fdf = pl.DataFrame([row for run in all_run_stats_F for row in run])
    avg_F = (Fdf.group_by('plasmid_number').mean().sort('plasmid_number'))
    Gdf = pl.DataFrame([row for run in all_run_stats_G for row in run])
    avg_G = (Gdf.group_by('plasmid_number').mean().sort('plasmid_number'))
    F_stats_df = pl.DataFrame(avg_F)
    F_stats_csv_path = os.path.join(big_location_analysis_path_current, 'F_graph_statistics.csv')
    F_stats_df.write_csv(F_stats_csv_path)
    G_stats_df = pl.DataFrame(avg_G)
    G_stats_csv_path = os.path.join(big_location_analysis_path_current, 'G_graph_statistics.csv')
    G_stats_df.write_csv(G_stats_csv_path)

sys.stdout.close()
sys.stdout = sys.__stdout__



def find_unbranching_paths(G):
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




for location in big_location:
    big_location_analysis_path_current = Path(os.path.join(big_location_analysis_path, f'{location}'))
    f_csv,g_csv = big_location_analysis_path_current / 'F_graph_statistics.csv', big_location_analysis_path_current / 'G_graph_statistics.csv'
    fdf, gdf = pd.read_csv(f_csv), pd.read_csv(g_csv)
    batch_x, f_nodes  = fdf['plasmid_number'].tolist(), fdf['node_number'].tolist()
    f_edges, g_edges = fdf['edge_number'].tolist(), gdf['edge_number'].tolist()
    complete_edges, complete_degree = [n*(n-1) for n in f_nodes], [(n-1) for n in f_nodes]
    f_density, g_density = [o/c if c > 0 else 0 for o, c in zip(f_edges, complete_edges)], [o/c if c > 0 else 0 for o, c in zip(g_edges, complete_edges)]
    f_degree, g_degree = [2*e/n if n > 0 else 0 for e, n in zip(f_edges, f_nodes)], [2*e/n if n > 0 else 0 for e, n in zip(g_edges, f_nodes)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    fig.suptitle('No. Edges / No. plasmids in batch')
    ax1.plot(batch_x, f_edges, '--', label='data', color='blue')
    ax1.plot(batch_x, complete_edges, ':', label='fully connected', color='orange')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('No. plasmids in batch')
    ax1.set_ylabel('No. edges')
    ax1.set_title("Signed architecture network")
    ax1.legend() 
    ax2.plot(batch_x, g_edges, '--', label='data', color='green')
    ax2.plot(batch_x, complete_edges, ':', label='fully connected', color='orange')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('No. plasmids in batch')
    ax2.set_ylabel('No. edges')
    ax2.set_title("Unsigned architecture network")
    ax2.legend() 
    edges_out = os.path.join(big_location_analysis_path_current, f'{location}_edges.png')
    plt.savefig(edges_out)
    plt.close()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    fig.suptitle('Avg. node degree / No. plasmids in batch')
    ax1.plot(batch_x, f_degree, '--', label='data', color='blue')
    ax1.plot(batch_x, complete_degree, ':', label='fully connected', color='orange')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('No. plasmids in batch')
    ax1.set_ylabel('Avg. node degree')
    ax1.set_title("Signed architecture network")
    ax1.legend() 
    ax2.plot(batch_x, g_degree, '--', label='data', color='green')
    ax2.plot(batch_x, complete_degree, ':', label='fully connected', color='orange')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('No. plasmids in batch')
    ax2.set_ylabel('Avg. node degree')
    ax2.set_title("Unsigned architecture network")
    ax2.legend() 
    degrees_out = os.path.join(big_location_analysis_path_current, f'{location}_degrees.png')
    plt.savefig(degrees_out)
    plt.close()
    fig, ax1 = plt.subplots(figsize=(10, 12))
    fig.suptitle('Density / No. plasmids in batch')
    ax1.plot(batch_x, f_density, '--', label='signed data', color='blue')
    ax1.plot(batch_x, g_density, ':', label='unsigned data', color='red')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('No. plasmids in batch')
    ax1.set_ylabel('No. Edges / No. possible edges')
    ax1.legend() 
    yticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([str(y) for y in yticks])
    density_out = os.path.join(big_location_analysis_path_current, f'{location}_density.png')
    plt.savefig(density_out)
    plt.close()


#batch_{len(plasmids)}_{location}_domain_architecture_signed_network.graphml


for location in big_location:
    all_path_stats = []
    big_location_analysis_path_current = Path(os.path.join(big_location_analysis_path, f'{location}'))
    batch_files = sorted(big_location_analysis_path_current.glob(f'batch_*_{location}_domain_architecture_network.graphml'),
                         key=lambda p: int(p.name.split('_')[1]))
    path_csv_rows = []
    for gml_path in batch_files:
        batch_num = int(gml_path.name.split('_')[1])
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
        print(f'  batch={batch_num:6d}: {n_nodes:5d} nodes, '
              f'{len(paths):4d} paths, max_len={max_len}, coverage={coverage:.1%}')
    paths_csv_path = big_location_analysis_path_current / 'unbranching_paths.csv'
    path_df = pd.DataFrame.from_dict(path_csv_rows)
    path_df.to_csv(paths_csv_path, index=False)
    pathstats_csv_path =  big_location_analysis_path_current / 'unbranching_paths_stats.csv'
    pathstats_df = pd.DataFrame.from_dict(all_path_stats)
    pathstats_df['coverage_percentage'] = [x*100 for x in pathstats_df['coverage']]
    pathstats_df.to_csv(pathstats_csv_path, index=False)
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(10, 70))
    fig.suptitle('Unbranching paths statistics')
    batch_n = pathstats_df['batch'].to_numpy()
    n_nodes2 = pathstats_df['n_nodes'].to_numpy()  
    n_paths = pathstats_df['n_paths'].to_numpy()    
    max_pathlen = pathstats_df['max_path_len'].to_numpy()
    mean_pathlen = pathstats_df['mean_path_len'].to_numpy()
    path_coverage = pathstats_df['coverage_percentage'].to_numpy()  
    ax1.plot(batch_n, n_paths, '--', label='data', color='blue')
    ax1.set_xlabel('Plasmid No. in batch')
    ax1.set_ylabel('No. unbranching paths')
    ax1.set_title('No. paths with increasing batch size')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend() 
    ax2.plot(batch_n, n_paths/n_nodes2, '--', label='data', color='blue')
    ax2.set_xlabel('Plasmid No. in batch')
    ax2.set_ylabel('No. unbranching paths relative to node number')
    ax2.set_title('No. paths relative to No. nodes with increasing batch size')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend() 
    ax3.plot(batch_n, max_pathlen, '--', label='data', color='blue')
    ax3.set_xlabel('Plasmid No. in batch')
    ax3.set_ylabel('Max length of unbranching paths')
    ax3.set_title('Max length of unbranching paths with increasing batch size')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend() 
    ax4.plot(batch_n, max_pathlen/n_nodes2, '--', label='data', color='blue')
    ax4.set_xlabel('Plasmid No. in batch')
    ax4.set_ylabel('Max length of unbranching paths relative to No. nodes')
    ax4.set_title('Max path length relative to No. nodes with increasing batch size')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend() 
    ax5.plot(batch_n, mean_pathlen, '--', label='data', color='blue')
    ax5.set_xlabel('Plasmid No. in batch')
    ax5.set_ylabel('Mean length of unbranching paths')
    ax5.set_title('Mean length of unbranching paths with increasing batch size')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.legend() 
    ax6.plot(batch_n, mean_pathlen/n_nodes2, '--', label='data', color='blue')
    ax6.set_xlabel('Plasmid No. in batch')
    ax6.set_ylabel('Mean length of unbranching paths relative to No. nodes')
    ax6.set_title('Mean path length relative to No. nodes with increasing batch size')
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.legend() 
    ax7.plot(batch_n, path_coverage, '--', label='data', color='blue')
    ax7.set_xlabel('Plasmid No. in batch')
    ax7.set_ylabel('Coverage of unbranching paths')
    ax7.set_title('Coverage of unbranching paths with increasing batch size')
    ax7.set_xscale('log')
    ax7.set_yscale('log')
    ax7.legend() 
    paths_out = Path(os.path.join(big_location_analysis_path_current, f'{location}_paths.png'))
    plt.savefig(paths_out)
    plt.close()

################################################################################################################################
################################################################################################################################
################################################################################################################################



#Kmeans was really not working -probably because the data is binary??
#Jaccard also not working because so many plasmids are extremely disimilar that the outliers ruin the clustering breadth
#used UMAP embedding Jaccard distance + HDBSCAN density clustering based on internet advice

#for plasmids use non-junky proteins from multifast

#data_dir = Path(os.path.join(os.getcwd(),'plasmid_motif_network/intermediate'))
#files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))

#df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()

#df_merged = df_merged.with_columns(
#    pl.col('strand').cast(pl.Int32).alias('strand')
#)

#all_proteins = list(set(list(df_merged['target_name'])))


#just_betas = [x for x in list(set(list(df_merged['target_name']))) if 'lactamase' in x]


import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
import umap
import hdbscan

OUT_DIR = Path('clustering_results')
OUT_DIR.mkdir(exist_ok=True)

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



###################################################################################################################
###################################################################################################################
###################################################################################################################





#will need to assign my genes and my pfam hit sequences (go back to get gene... ugh...) to beta lactamase families to do an 
#MSA


###################################################################################################################
###################################################################################################################
###################################################################################################################

#CARD


import json
import re
import pandas as pd
from pathlib import Path
from Bio.Seq import Seq
import subprocess

beta_fasta_path = os.path.join(os.getcwd(), 'beta_lactam_fastas')
beta_fastas = os.listdir(beta_fasta_path)
#>>> len(list(set(beta_fastas)))
#825
beta_fromfasta_ids = [f'{'_'.join(x.split('_')[:-1])}' for x in beta_fastas]

#data_dir = Path(os.path.join(os.getcwd(),'plasmid_motif_network/intermediate'))
#files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))
#
#df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
#
#df_merged = df_merged.with_columns(
#    pl.col('strand').cast(pl.Int32).alias('strand')
#)


beta_pfam_df = df_merged.filter(pl.col('target_name').str.contains('lactamase'))
beta_frompfam_ids = list(beta_pfam_df['query_name'])

beta_pfam_not_in_plsdbfastas = [x for x in beta_frompfam_ids if x not in beta_fromfasta_ids]


pfam_betas_fasta_path = Path(os.path.join(os.getcwd(), 'pfam_beta_lactamases_fastas'))
os.makedirs(pfam_betas_fasta_path, exist_ok=True)

plasmids_path = Path(os.path.join(os.getcwd(), 'plasmids'))


pfamhits_ids = []
pfamhits_orientation = []

with open('hmminput_allplasmid_proteins_strandorientation.fa', 'r') as g:
    for line in g:
        if line.startswith('>'):
            pfamhits_ids.append(line[1:-1])
        else:
            pfamhits_orientation.append(line[:-1])


pfamhit_id_to_orientation = dict(zip(pfamhits_ids, pfamhits_orientation))

with open('pfam_betalactamase_genesequences.fa', 'w') as g:
    for x in beta_frompfam_ids:
        plasmid_id = '_'.join(x.split('_')[:-2])
        start = int(x.split('_')[-2])
        stop = int(x.split('_')[-1])
        plasmid_fasta_path = os.path.join(plasmids_path, f'{plasmid_id}.fa')
        with open(plasmid_fasta_path, 'r') as f:
            sequence = f.readlines()[1]
            geneseq = sequence[start:stop]
            if pfamhit_id_to_orientation.get(x) == 'minus':
                geneseq = str(Seq(geneseq).reverse_complement())
        g.write(f'>{x}\n{geneseq}\n')


g.close()



#Need to map beta lactamase gene sequences to gene names (and where possible broader families as a separate category), 
#based on the plasmid_start_stop id both from fastas directory (files named) plasmidnuccoreid_start_stop_genename.fa 
#where header is >plasmidnuccoreid_start_stop_genename and from pfam to gene name from file 
# pfam_betalactamase_genesequences.fa where header is >plasmidnuccoreid_start_stop
#needs to be consistent with CARD database (because it needs to be linked to card prevalence data for beta-lactamases later)



#note that around 40% of the hits from pfam match the plsdb fastas in terms of id
#all of which do match are in the beta lactamase fasta directory












































































































#CARD_JSON = Path(os.path.join(os.getcwd(), 'card.json'))

CARD_JSON = Path(os.path.join(os.getcwd(), 'all_card/card.json'))
OUT_DIR = Path('card_gof_reference')
OUT_DIR.mkdir(exist_ok=True)

with open(CARD_JSON) as f:
    card = json.load(f)


bl_entries = []
for aro_id, entry in card.items():
    if not isinstance(entry, dict):
        continue
    categories = entry.get('ARO_category', {})
    is_betalactamase = any(
        'beta-lactamase' in v.get('category_aro_name', '').lower() or
        'beta-lactam' in v.get('category_aro_name', '').lower()
        for v in categories.values()
    )
    if not is_betalactamase:
        continue
    families = [
        v.get('category_aro_name', '')
        for v in categories.values()
        if v.get('category_aro_class_name') == 'AMR Gene Family'
    ]
    drug_classes = [
        v.get('category_aro_name', '')
        for v in categories.values()
        if v.get('category_aro_class_name') == 'Drug Class'
    ]
    ref_acc = ''
    ref_seq = ''
    for seq_data in entry.get('model_sequences', {}).get('sequence', {}).values():
        prot = seq_data.get('protein_sequence', {})
        ref_acc = prot.get('accession', '')
        ref_seq = prot.get('sequence', '')
        break
    bl_entries.append({
        'ARO_accession': entry.get('ARO_accession', ''),
        'ARO_name':      entry.get('ARO_name', ''),
        'model_type':    entry.get('model_type', ''),
        'gene_family':   '; '.join(families),
        'drug_class':    '; '.join(drug_classes),
        'ref_protein_acc': ref_acc,
        'ref_protein_seq': ref_seq,
    })

bl_df = pd.DataFrame(bl_entries)

bl_notgay_fastas = os.listdir(os.path.join(os.getcwd(), 'beta_lactam_fastas'))
bl_notgay_genes = ['.'.join(x.split('.')[:-1]).split('_')[-1] for x in bl_notgay_fastas]

bl_notpremapped = [x for x in bl_notgay_genes if x not in bl_df['ARO_name'].tolist()]
#bl_checker = len(bl_check) / len(bl_notgay_genes)





GOF_KEYWORDS = [
    'extended spectrum',
    'extended-spectrum',
    'gain of function',
    'gain-of-function',
    'increased resistance',
    'broadened spectrum',
    'enhanced activity',
    'carbapenem',
    'esbl',
]


rows = []
for aro_id, entry in card.items():
    if not isinstance(entry, dict):
        continue
    if entry.get('model_type') != 'protein variant model':
        continue
    aro_accession = entry.get('ARO_accession', '')
    aro_name      = entry.get('ARO_name', '')
    aro_desc      = entry.get('ARO_description', '')
    drug_classes  = '; '.join(
        v.get('category_aro_name', '')
        for v in entry.get('ARO_category', {}).values()
        if v.get('category_aro_class_name') == 'Drug Class'
    )
    gene_families = '; '.join(
        v.get('category_aro_name', '')
        for v in entry.get('ARO_category', {}).values()
        if v.get('category_aro_class_name') == 'AMR Gene Family'
    )
    ref_seqs = {
        seq_id: seq_data.get('protein_sequence', {}).get('accession', '')
        for seq_id, seq_data in entry.get('model_sequences', {}).get('sequence', {}).items()
    }
    ref_accession = list(ref_seqs.values())[0] if ref_seqs else ''

    for param in entry.get('model_param', {}).values():
        if param.get('param_type') != 'single resistance variant':
            continue
        all_muts  = param.get('param_value', {})
        curated_r = set(param.get('Curated-R', {}).values())
        clinical  = set(param.get('clinical',  {}).values())

        for mut_id, mutation in all_muts.items():
            if not mutation:
                continue
            m = re.match(r'([A-Za-z\*])(\d+)([A-Za-z\*]+)', mutation)
            if not m:
                continue
            rows.append({
                'ARO_accession':         aro_accession,
                'ARO_name':              aro_name,
                'ARO_description':       aro_desc,
                'drug_class':            drug_classes,
                'gene_family':           gene_families,
                'ref_protein_acc':       ref_accession,
                'mutation':              mutation,
                'ref_aa':                m.group(1).upper(),
                'position':              int(m.group(2)),
                'alt_aa':                m.group(3).upper(),
                'in_curated_r':          mutation in curated_r,
                'has_clinical_evidence': mutation in clinical,
            })

variants_df = pd.DataFrame(rows)

n_models = len([v for v in card.values() if isinstance(v, dict) and v.get('model_type') == 'protein variant model'])
print(f'protein variant models:        {n_models}')
print(f'total mutations extracted:     {len(variants_df)}')
print(f'in Curated-R (predicted):      {variants_df["in_curated_r"].sum()}')
print(f'with clinical evidence:        {variants_df["has_clinical_evidence"].sum()}')
print(f'unique AROs:                   {variants_df["ARO_accession"].nunique()}')

variants_df.to_csv(OUT_DIR / 'card_variant_catalogue.csv', index=False)
print(f'saved to {OUT_DIR}/card_variant_catalogue.csv')






































































































































###################################################################################################################

#
#import subprocess
#import pandas as pd
#from pathlib import Path
#
#OUT_DIR = Path('card_gof_reference')
#OUT_DIR.mkdir(exist_ok=True)
#
#PROTEIN_FA = Path('hmminput_allplasmid_proteins_nojunk.fa')
#NUC_DIR = Path('beta_lactam_fastas')
#THREADS = 8
#
#
#incidence = 0
#noninc = 0
#
#nuc_out = OUT_DIR / 'all_betalactamase_nuc.fa'
#nuc_out_clean = OUT_DIR / 'all_betalactamase_nuc_clean.fa'
#nuc_outprobs = OUT_DIR / 'all_betalactamase_nuc_problems.fa'
#with open(nuc_out, 'w') as outf:
#    with open(nuc_outprobs, 'w') as outfprobs:
#        with open(nuc_out_clean, 'w') as outfclean:
#            all_nucseqs = OUT_DIR / 'all_query_sequences.fa'
#            with open(all_nucseqs, 'r') as ing:
#                for line in ing:
#                    if line.startswith('>'):
#                        outf.write(line)
#                        noninc += 1
#                    else:
#                        realseq = line[:-1]
#                        protseq = str(Seq(realseq).translate())
#                        outf.write(f'{realseq}\n')
#                        if protseq.startswith('M') and '*' not in protseq[:-1] and  protseq[-1] == '*':
#                            incidence += 1
#                            #if not start M, or have stop before end, or not end stop, or % 3 leave it be
#                            outfclean.write(prior)
#                            outfclean.write(f'{realseq}\n')
#                        else:
#                            outfprobs.write(prior)
#                            outfprobs.write(f'{realseq}\n')
#                    prior = line
#
#
##subprocess.run([
##    'rgi', 'main',
##    '--input_sequence', str(PROTEIN_FA),
##    '--output_file',    str(OUT_DIR / 'rgi_proteins'),
##    '--input_type',     'protein',
##    '--local',
##    '--clean',
##    '--num_threads',    str(THREADS),
##], check=True)
##
#subprocess.run([
#    'rgi', 'main',
#    '--input_sequence', str(nuc_out),
#    '--output_file',    str(OUT_DIR / 'rgi_nuc'),
#    '--input_type',     'contig',
#    '--local',
#    '--clean',
#    '--num_threads',    str(THREADS),
#], check=True)
#
#prot_df = pd.read_csv(OUT_DIR / 'rgi_proteins.txt', sep='\t')
#nuc_df  = pd.read_csv(OUT_DIR / 'rgi_nuc.txt',      sep='\t')
#
#keep_cols = ['ORF_ID', 'Best_Hit_ARO', 'ARO', 'Best_Identities',
#             'Model_type', 'SNPs_in_Best_Hit_ARO', 'Best_Identifier']
#
#prot_df = prot_df[[c for c in keep_cols if c in prot_df.columns]].copy()
#nuc_df  = nuc_df [[c for c in keep_cols if c in nuc_df.columns ]].copy()
#
#prot_df['source'] = 'protein'
#nuc_df['source']  = 'nucleotide'
#
#combined = pd.concat([prot_df, nuc_df], ignore_index=True)
#variant_hits = combined[combined['Model_type'].str.contains('variant|mutant', case=False, na=False)]
#
#print(f'total RGI hits:         {len(combined)}')
#print(f'variant model hits:     {len(variant_hits)}')
#print(f'unique AROs hit:        {combined["Best_Hit_ARO"].nunique()}')
#
#combined.to_csv(OUT_DIR / 'rgi_combined.csv', index=False)
#variant_hits.to_csv(OUT_DIR / 'rgi_variant_hits.csv', index=False)
#print(f'saved to {OUT_DIR}/')
#
#



###################################################################################################################


import re
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from Bio.Align import PairwiseAligner

OUT_DIR    = Path('card_gof_reference')
PROTEIN_FA = Path('hmminput_allplasmid_proteins_nojunk.fa')
NUC_DIR    = Path('beta_lactamase_fastas')

CARD_REF_FA = Path('protein_fasta_protein_variant_model.fasta')

gof_df      = pd.read_csv(OUT_DIR / 'card_gof_variants.csv')
rgi_df      = pd.read_csv(OUT_DIR / 'rgi_variant_hits.csv')

print('loading sequences...')
your_seqs  = {r.id: str(r.seq) for r in SeqIO.parse(PROTEIN_FA, 'fasta')}

nuc_seqs = {}
for fa in NUC_DIR.glob('*.fa'):
    for r in SeqIO.parse(fa, 'fasta'):
        nuc_seqs[r.id] = str(r.seq)
your_seqs.update(nuc_seqs)

card_seqs  = {r.id: str(r.seq) for r in SeqIO.parse(CARD_REF_FA, 'fasta')}
print(f'your sequences: {len(your_seqs)}  CARD refs: {len(card_seqs)}')

aligner = PairwiseAligner()
aligner.mode            = 'global'
aligner.match_score     = 2
aligner.mismatch_score  = -1
aligner.open_gap_score  = -10
aligner.extend_gap_score = -0.5

def build_position_map(ref_seq, query_seq):
    alignments = aligner.align(ref_seq, query_seq)
    if not alignments:
        return {}
    aln         = alignments[0]
    ref_aln     = str(aln[0])
    query_aln   = str(aln[1])
    ref_pos     = 0
    query_pos   = 0
    ref_to_query = {}
    for r, q in zip(ref_aln, query_aln):
        if r != '-':
            ref_pos += 1
        if q != '-':
            query_pos += 1
        if r != '-':
            ref_to_query[ref_pos] = (query_pos if q != '-' else None, q if q != '-' else None)
    return ref_to_query

query_to_aro = dict(zip(rgi_df['ORF_ID'], rgi_df['Best_Hit_ARO']))
query_to_ref = dict(zip(rgi_df['ORF_ID'], rgi_df['Best_Identifier']))

gof_by_aro = gof_df.groupby('ARO_name')

results = []
total   = len(rgi_df)

for i, (_, row) in enumerate(rgi_df.iterrows()):
    query_id = row['ORF_ID']
    aro_name = row['Best_Hit_ARO']
    ref_id   = row['Best_Identifier']

    query_seq = your_seqs.get(query_id)
    ref_seq   = card_seqs.get(ref_id)

    if not query_seq or not ref_seq:
        continue

    if aro_name not in gof_by_aro.groups:
        continue

    pos_map = build_position_map(ref_seq, query_seq)
    relevant = gof_by_aro.get_group(aro_name)

    for _, var in relevant.iterrows():
        card_pos   = var['position']
        mapped     = pos_map.get(card_pos)
        query_pos  = mapped[0] if mapped else None
        query_aa   = mapped[1] if mapped else None

        results.append({
            'query_id':              query_id,
            'aro_name':              aro_name,
            'card_ref_id':           ref_id,
            'mutation':              var['mutation'],
            'card_ref_aa':           var['ref_aa'],
            'card_position':         card_pos,
            'resistance_alt_aa':     var['alt_aa'],
            'query_position':        query_pos,
            'query_current_aa':      query_aa,
            'clinical_significance': var['clinical_significance'],
        })

    if i % 100 == 0:
        print(f'  {i}/{total} sequences processed')

results_df = pd.DataFrame(results)

print(f'\ntotal GoF position mappings: {len(results_df)}')
print(f'unique query sequences:      {results_df["query_id"].nunique()}')
print(f'unique AROs covered:         {results_df["aro_name"].nunique()}')

results_df.to_csv(OUT_DIR / 'gof_position_map.csv', index=False)
print(f'saved to {OUT_DIR}/gof_position_map.csv')
print('\ncolumns:')
print(results_df.dtypes.to_string())





###################################################################################################################




































































































































































































#JUNKYARD
#################################################################################################################
#Circos plots?

#Make visualisations for the graphs up to 10 plasmids that have the label for node identity to make circular force-directed graphs with plasmids where there is an addition non-visualised label for the plasmids the node belongs for lookup
#Ideally the nodes would be different colours, the line thickness of the edges (as arrows) would represent the weight. 
#
#from pycirclize import Circos
#import pandas as pd
#
## Create matrix dataframe (3 x 6)
#row_names = ["S1", "S2", "S3"]
#col_names = ["E1", "E2", "E3", "E4", "E5", "E6"]
#matrix_data = [
#    [4, 14, 13, 17, 5, 2],
#    [7, 1, 6, 8, 12, 15],
#    [9, 10, 3, 16, 11, 18],
#]
#matrix_df = pd.DataFrame(matrix_data, index=row_names, columns=col_names)
#
## Initialize Circos instance for chord diagram plot
#circos = Circos.chord_diagram(
#    matrix_df,
#    start=-265,
#    end=95,
#    space=5,
#    r_lim=(93, 100),
#    cmap="tab10",
#    label_kws=dict(r=94, size=12, color="white"),
#    link_kws=dict(ec="black", lw=0.5),
#)
#
#print(matrix_df)
#fig = circos.plotfig()
#
#
#
#
#import re
#from collections import defaultdict
#from pathlib import Path
#import pandas as pd
#import networkx as nx
#from pycirclize import Circos
#import math
#
#GRAPHML = Path(os.path.join(os.getcwd(), 'plasmid_batched_graphs/1000_domain_architecture_network.graphml'))
#
#domain_df = pd.read_csv('Pfam-A.clans.tsv', sep='\t', header=None)
#domain_dict = dict(zip(domain_df[3].tolist(), domain_df[1].tolist()))
#
#OUT_PNG   = Path(os.path.join(out_dir, 'circos_1000plasmids.png'))
#
# ── Classification ────────────────────────────────────────────────────────────
# Two-stage: domain name checked first (catches biological exceptions like TetR
# whose structural clan HTH_3 is too broad), then clan name.
# Edit labels - they become the sector names on the plot.

#DOMAIN_OVERRIDES = [
#    (r"^TetR_",                          "AMR"),
#    (r"^mcr",                            "AMR"),
#    (r"^qnr",                            "AMR"),
#    (r"^sul\d",                          "AMR"),
#    (r"^dfr",                            "AMR"),
#    (r"^VanA|^VanB|^VanH|^VanX|^VanZ",  "AMR"),
#    (r"^MobA|^MobB|^MobC",              "Plasmid biology"),
#    (r"^IstB",                           "Mobile element"),
#    (r"^ParE_toxin",                     "Toxin-antitoxin"),
#    (r"^DUF\d",                          "Unknown (DUF)"),
#]

#CLAN_MAP = [
#    (r"Beta.lactamase|Lactamase_B",             "AMR"),
#    (r"Aminoglycoside|APH|AAC_",                "AMR"),
#    (r"Vancomycin|Sulfonamide|DHFR|Quinolone",  "AMR"),
#    (r"^MFS$",                                  "Transporter"),
#    (r"^RND$",                                  "Transporter"),
#    (r"Retroviral_integrase|Phage_integrase",   "Mobile element"),
#    (r"DDE_transposase",                        "Mobile element"),
#    (r"^Resolvase",                             "Mobile element"),
#    (r"Transposase",                            "Mobile element"),
#    (r"Replication_init",                       "Plasmid biology"),
#    (r"Mob_relaxase|Relaxase",                  "Plasmid biology"),
#    (r"Plasmid_stab|Partition",                 "Plasmid biology"),
#    (r"Conjugal|VirB|Type_IV",                  "Conjugation"),
#    (r"Response_reg",                           "Regulation"),
#    (r"Histidine_kinase|HATPase|HAMP",          "Regulation"),
#    (r"Sigma",                                  "Regulation"),
#    (r"LysR|GntR|AraC|MarR|LacI",              "Regulation"),
#    (r"^HTH",                                   "Regulation"),
#    (r"Antitoxin",                              "Toxin-antitoxin"),
#    (r"Toxin",                                  "Toxin-antitoxin"),
#    (r"P-loop_NTPase",                          "ATPase / motor"),
#    (r"Acetyltransf|GNAT|Methyltrans",          "Enzyme"),
#    (r"Alpha.beta_hydrolase|Adh_short",         "Enzyme"),
#]




#domain_df = pd.read_csv('Pfam-A.clans.tsv', sep='\t', header=None)
#domain_dict = dict(zip(domain_df[3].tolist(), domain_df[2].tolist()))

#def classify(domain_name, domain_dict):
#    for pattern, label in DOMAIN_OVERRIDES:
#        if re.search(pattern, domain_name, re.I):
#            return label
#    clan = domain_dict.get(domain_name)
#    if isinstance(clan, str):
#        for pattern, label in CLAN_MAP:
#            if re.search(pattern, clan, re.I):
#                return label
#    return "Other"





#GG = nx.read_graphml(GRAPHML)
#domain_class = {node: classify(node, domain_dict) for node in GG.nodes()}


#domain_class = {x:domain_dict.get(x) for x in list(GG.nodes)}
#domain_class= {k: v if isinstance(v, str) else 'Unknown' for k, v in domain_class.items()}

#domain_mapped_nodes = [(x, domain_dict.get(x)) for x in list(GG.nodes)]
#mapped_nodes = [x for x in domain_mapped_nodes if type(x[1]) == str]
#unmapped_nodes = [x for x in domain_mapped_nodes if type(x[1]) != str]
#
#test = [x for x in unmapped_nodes if 'DUF' not in x[0]]
#
#phage_bits = [x for x in test if 'phage'  in x[0] or 'Phage' in x[0]]
#tox_bits = [x for x in test if 'tox'  in x[0] or 'Tox' in x[0]]
#
#gra = [x for x in test if x not in phage_bits and x not in tox_bits]



#'Secretion '
#'Toxin-Antitoxin'
#'AMR'
#'Conjugation'
#'DNA replication and maintainance'
#'Transcription and regulation'
#'Metabolic'
#'AMR'
#'Unknown'
#'Plasmid integral'
#'Mobile genetic element'
#'Toxins and effectors'
#'Transport and membrane'
#'Other functional (mobility adhesion secretion etc.)'
#'CRISPR '
#'Unknown'

#weights = defaultdict(lambda: defaultdict(float))
#for u, v, data in GG.edges(data=True):
#    weights[domain_class[u]][domain_class[v]] += float(data.get("weight", 1))
#
#
#classes = sorted(set(domain_class.values()))
#matrix  = pd.DataFrame(0.0, index=classes, columns=classes)
#for r in classes:
#    for c in classes:
#        matrix.loc[r, c] = weights[r][c]
#
#
#active = matrix.index[(matrix.sum(axis=1) + matrix.sum(axis=0)) > 0]
#matrix = matrix.loc[active, active]
#print(f'classes in plot: {list(matrix.index)}')
#
#
#
#OUT_PNG   = Path(os.path.join(out_dir, 'circos_1000plasmidstest.png'))
#
#circos2 = Circos.chord_diagram(
#    matrix,
#    space=10,
#    r_lim=(90, 100),
#    cmap="tab10",
#    label_kws=dict(r=105, size=11, color="black"),
#    link_kws=dict(ec="black", lw=0.5),
#)
#
#fig = circos2.plotfig(figsize=(12, 12))
#fig.savefig(OUT_PNG, dpi=180, bbox_inches='tight')
#print(f'Saved → {OUT_PNG}')


###
#define the batches of plasmids of variable number, random but reproducible. Could be made truly random if desired.


#batch_numbers = list(range(1,10,1)) + list(range(10,100,10)) + list(range(100,1000,100)) + list(range(1000,10000, 1000)) + list(range(10000, 27500, 2500)) + [27114]
#batch_num_to_plasmids = {}
#for batch_num in batch_numbers:
#    selected_plasmids = random.sample(list(set(list(df_merged['plasmid']))), batch_num)
#    batch_num_to_plasmids[batch_num] = selected_plasmids
#max_size = len(all_plasmids)
#num_of_batches = 100 
#batch_sizes = np.unique(np.geomspace(1, max_size, num=num_of_batches, dtype=int))
#
#batch_num_to_plasmids = {}
#for size in batch_sizes:
#    batch_num_to_plasmids[size] = all_plasmids[:size]
#batch_num_to_plasmids[max_size] = all_plasmids



#sanity check
#test = nx.read_graphml('/home/kd541/kdan2/kdan/data/plasmid_batched_graphs/27114_domain_architecture_signed_network.graphml')
#test = list(test)
#len([x for x in test if x in domain_dict.keys()]) / len(test) * 100
#number_of_runs = list(range(1,3))
#for run in number_of_runs:
#    random.shuffle(all_plasmids) 
#    max_size = len(all_plasmids)
#    num_of_batches = 5 
#    batch_sizes = np.unique(np.geomspace(1, max_size, num=num_of_batches, dtype=int))
#    batch_num_to_plasmids = {}
#    for size in batch_sizes:
#        batch_num_to_plasmids[size] = all_plasmids[:size]
#    F_graph_stats = []
#    G_graph_stats = []
#    for num, plasmids in batch_num_to_plasmids.items():
#        df_filt = df_merged.filter(pl.col('plasmid').is_in(plasmids))
#        df = df_filt
#        df = df.sort(['plasmid', 'start', 'ali_from'])
#        #make ordered list of domain hits per gene/protein
#        ordered = df.select(['plasmid','query_name','target_name','start','ali_from','strand'])
#        df = ordered
#        #make adj
#        adjacency = defaultdict(int)
#        #for plasmid, group in tqdm(df.group_by('plasmid'), disable=True):
#        for (plasmid,), group in df.partition_by('plasmid', as_dict=True).items():
#            g = group.sort(['start', 'ali_from'])
#            domains = g['target_name'].to_list()
#            strands = g['strand'].to_list()
#            if len(domains) < 2:
#                continue
#            n = len(domains)
#            for i in range(n):
#                d1 = domains[i]
#                d2 = domains[(i+1) % n]
#                s1 = strands[i]
#                s2 = strands[(i+1) % n]
#                if s1 == 1 and s2 == 1:
#                    ori = 'PP'
#                elif s1 == -1 and s2 == -1:
#                    ori = 'MM'
#                elif s1 == 1 and s2 == -1:
#                    ori = 'PM'
#                else:
#                    ori = 'MP'
#                #adjacency[(d1, d2, ori)] += 1
#                adjacency[(d1, d2, ori)] = adjacency[(d1, d2, ori)] + 1
#        rows = [(k[0], k[1], k[2], v) for k, v in adjacency.items()]
#        if not rows:
#            continue
#        d1s, d2s, oris, weights = zip(*rows)
#        adj_df = pl.DataFrame({'domain1': list(d1s),'domain2': list(d2s),'orientation': list(oris),'weight': list(weights)})
#        #adj_df = pl.DataFrame(rows,schema=['domain1', 'domain2', 'orientation', 'weight'])
#        #signed adj
#        df = adj_df
#        df = df.with_columns(pl.when(pl.col('orientation').is_in(['PP', 'MM'])).then(pl.col('weight')).otherwise(-pl.col('weight')).alias('signed_contribution'))
#        collapsed = (
#            df.group_by(['domain1', 'domain2'])
#              .agg([
#                  pl.sum('signed_contribution').alias('signed_weight'),
#                  pl.sum('weight').alias('total_weight')
#              ])
#        )
#        #make graph for adj
#        df = adj_df
#        domain_to_plasmids = defaultdict(set)
#        for row in ordered.iter_rows(named=True):
#            domain_to_plasmids[row['target_name']].add(row['plasmid'])
#        #G = nx.DiGraph()
#        G= nx.MultiDiGraph()
#        for row in df.iter_rows(named=True):
#            d1 = row['domain1']
#            d2 = row['domain2']
#            weight = row['weight']
#            orientation = row['orientation']
#            G.add_edge(
#                d1,
#                d2,
#                weight=weight,
#                orientation=orientation
#            )
#        for node in G.nodes():
#            plasmid_set = domain_to_plasmids.get(node, set())
#            G.nodes[node]['label'] = node
#            G.nodes[node]['plasmid_count'] = len(plasmid_set)
#            G.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
#        for global_id, (u, v, k) in enumerate(G.edges(keys=True)):
#            G[u][v][k]['id'] = str(global_id)
#        nx.write_graphml(G, os.path.join(output_path, f'{num}_domain_architecture_network.graphml'), edge_id_from_attribute='id')
#        #make graph for signed adj
#        df = collapsed
#        F = nx.DiGraph()
#        for row in df.iter_rows(named=True):
#            d1 = row['domain1']
#            d2 = row['domain2']
#            signed_weight = row['signed_weight']
#            total_weight = row['total_weight']
#            F.add_edge(
#                d1,
#                d2,
#                signed_weight=signed_weight,
#                total_weight=total_weight
#            )
#        for node in F.nodes():
#            plasmid_set = domain_to_plasmids.get(node, set())
#            F.nodes[node]['label'] = node
#            F.nodes[node]['plasmid_count'] = len(plasmid_set)
#            F.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
#        nx.write_graphml(F, os.path.join(output_path, f'{num}_domain_architecture_signed_network.graphml'))
#        F_node_number = F.number_of_nodes()
#        F_edge_number = F.number_of_edges()
#        F_average_degree = (2 * F_edge_number / F_node_number) if F_node_number > 0 else 0.0
#        F_graph_stats.append({
#            'plasmid_number': num,
#            'node_number': F_node_number,
#            'edge_number': F_edge_number,
#            'average_degree': F_average_degree
#        })
#        G_node_number = G.number_of_nodes()
#        G_edge_number = G.number_of_edges()
#        G_average_degree = (2 * G_edge_number / G_node_number) if G_node_number > 0 else 0.0
#        G_graph_stats.append({
#            'plasmid_number': num,
#            'node_number': G_node_number,
#            'edge_number': G_edge_number,
#            'average_degree': G_average_degree
#        })
#    all_run_stats_F.append(F_graph_stats)
#    all_run_stats_G.append(G_graph_stats)
    