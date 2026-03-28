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



data_dir = Path(os.path.join(os.getcwd(),'plasmid_motif_network/intermediate'))
files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))

df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()

df_merged = df_merged.with_columns(
    pl.col('strand').cast(pl.Int32).alias('strand')
)


#Get plasmid info

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
#Generate general graphs for location groups

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
#for larger groups of plasmids by location do per batch size analyses

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


#degree edges and density plots

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



#path plots
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
