import os
import re 
import pandas as pd
import numpy as np
from collections import Counter 
from pathlib import Path
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
import os
import re
import math
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import polars as pl
import networkx as nx
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import os
import re 
import math
import warnings
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import networkx as nx
import leidenalg
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================

CLUSTER_PATH = Path('clustering_results')
graph_dir    = CLUSTER_PATH
out_dir      = CLUSTER_PATH

CLUSTER_PATH.mkdir(exist_ok=True)
out_dir.mkdir(exist_ok=True)

MERGED_FASTA_DIR = Path('merged_nonoverlapping_fastas')
MIN_OBS = 10  # minimum copies for entropy/relative-entropy to be computed

# =============================================================================
# 2. DATA LOADING & PREPROCESSING
# =============================================================================
print("Loading parquet files...")
data_dir = Path('plasmid_motif_network/intermediate')
files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))
df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
df_merged = df_merged.with_columns(
    pl.col('strand').cast(pl.Int32).alias('strand')
)

print("Loading cluster assignments...")
jac_clusters = pd.read_csv(CLUSTER_PATH / 'umap_hdbscan_clusters.csv')
plasmid_jac_cluster = dict(zip(jac_clusters['plasmid'], jac_clusters['cluster']))
clusters = sorted(list(set(plasmid_jac_cluster.values())))
cluster_totals = Counter(plasmid_jac_cluster.values())

# =============================================================================
# 3. DOMAIN ARCHITECTURE NETWORK GENERATION
# =============================================================================
print("Generating domain architecture networks per cluster...")
for cluster in tqdm(clusters, desc="Clusters"):
    plasmids = [k for k,v in plasmid_jac_cluster.items() if v == cluster]
    df_filt = df_merged.filter(pl.col('plasmid').is_in(plasmids))
    df_filt = df_filt.sort(['plasmid', 'start', 'ali_from'])
    
    ordered = df_filt.select(['plasmid','query_name','target_name','start','ali_from','strand'])
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
    
    adj_df = adj_df.with_columns(pl.when(pl.col('orientation').is_in(['PP', 'MM'])).then(pl.col('weight')).otherwise(-pl.col('weight')).alias('signed_contribution'))
    
    collapsed = (
        adj_df.group_by(['domain1', 'domain2'])
          .agg([
              pl.sum('signed_contribution').alias('signed_weight'),
              pl.sum('weight').alias('total_weight')
          ])
    )
    
    domain_to_plasmids = defaultdict(set)
    for row in ordered.iter_rows(named=True):
        domain_to_plasmids[row['target_name']].add(row['plasmid'])
        
    G = nx.MultiDiGraph()
    for row in adj_df.iter_rows(named=True):
        G.add_edge(
            row['domain1'], row['domain2'],
            weight=row['weight'], orientation=row['orientation']
        )
        
    for node in G.nodes():
        plasmid_set = domain_to_plasmids.get(node, set())
        G.nodes[node]['label'] = node
        G.nodes[node]['plasmid_count'] = len(plasmid_set)
        G.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
        
    for global_id, (u, v, k) in enumerate(G.edges(keys=True)):
        G[u][v][k]['id'] = str(global_id)
        
    nx.write_graphml(G, CLUSTER_PATH / f'cluster_{cluster}_domain_architecture_network.graphml', edge_id_from_attribute='id')
    
    F = nx.DiGraph()
    for row in collapsed.iter_rows(named=True):
        F.add_edge(
            row['domain1'], row['domain2'],
            signed_weight=row['signed_weight'], total_weight=row['total_weight']
        )
        
    for node in F.nodes():
        plasmid_set = domain_to_plasmids.get(node, set())
        F.nodes[node]['label'] = node
        F.nodes[node]['plasmid_count'] = len(plasmid_set)
        F.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
        
    nx.write_graphml(F, CLUSTER_PATH / f'cluster_{cluster}_domain_architecture_signed_network.graphml')

# =============================================================================
# 4. DOMAIN POSITIONS & MAPPING
# =============================================================================
ordered_df = df_merged.sort(['plasmid', 'start', 'ali_from']).select(['plasmid', 'target_name'])

print("Building circular domain lists...")
plasmid_to_domains = defaultdict(list)
for row in ordered_df.iter_rows(named=True):
    plasmid_to_domains[row['plasmid']].append(row['target_name'])

domain_positions = defaultdict(list)
for plasmid, doms in plasmid_to_domains.items():
    n = len(doms)
    if n <= 1:
        continue
    for i, dom in enumerate(doms):
        domain_positions[dom].append((plasmid, i))

print(f"Loaded {len(domain_positions)} unique domains across {len(plasmid_to_domains)} plasmids.")

test = pd.read_csv('amrfindermapped_beta_lactamases.csv')
all_gene_names = [x for x in test['gene_name'].unique() if isinstance(x, str)]
gene_to_family = {
    gname: test.loc[test['gene_name'] == gname, 'gene_family'].iloc[0]
    for gname in all_gene_names
}

gene_to_pids = defaultdict(set)
for _, row in test.iterrows():
    if isinstance(row['gene_name'], str):
        gene_to_pids[row['gene_name']].add(row['query_id'])

print(f"Loaded {len(all_gene_names)} BL gene names from AMRFinder mapping.")

# =============================================================================
# 5. ENTROPY LOGIC
# =============================================================================
def context_entropy_score(contexts):
    """
    contexts : list of (left_domain, right_domain) tuples
    Returns  : (penalised_score, raw_H, N, K)
    """
    N = len(contexts)
    if N <= 1:
        return np.nan, np.nan, N, 0
    counts = Counter(contexts)
    K = len(counts)
    if K <= 1:
        return 0.0, 0.0, N, K
    probs = np.array(list(counts.values())) / N
    H = -np.sum(probs * np.log2(probs))
    H_context = H / np.log2(K)
    score = H_context * (K / N)
    return round(score, 6), round(H, 6), N, K

# Build PID → positions lookup
pid_to_positions = defaultdict(list)
plasmid_index_counter = defaultdict(int)

ordered_full = df_merged.sort(['plasmid', 'start', 'ali_from']).select(['query_name', 'plasmid', 'target_name'])
for row in ordered_full.iter_rows(named=True):
    plasmid = row['plasmid']
    idx = plasmid_index_counter[plasmid]
    plasmid_index_counter[plasmid] += 1
    pid_to_positions[row['query_name']].append((plasmid, idx, row['target_name']))

# =============================================================================
# 6. PER-GENE CLUSTER METRICS
# =============================================================================
print("\nCalculating per-gene cluster entropy metrics...")

records = []

for gene_name in tqdm(all_gene_names, desc='Per-gene'):
    pids_this_gene = gene_to_pids[gene_name]
    family = gene_to_family.get(gene_name, '')
    
    cluster_contexts = defaultdict(list)
    cluster_gene_plasmids = defaultdict(set)
    total_copies = 0

    for pid in pids_this_gene:
        for plasmid, idx, dom in pid_to_positions.get(pid, []):
            cluster = plasmid_jac_cluster.get(plasmid)
            if cluster is None:
                continue
                
            entries = plasmid_to_domains[plasmid]
            n = len(entries)
            if n <= 1:
                continue

            left  = entries[(idx - 1) % n]
            right = entries[(idx + 1) % n]

            cluster_contexts[cluster].append((left, right))
            cluster_gene_plasmids[cluster].add(plasmid)
            total_copies += 1

    if total_copies == 0:
        continue # Skip if entirely absent

    cluster_entropies = {}
    cluster_context_ratios = {}
    cluster_weights = {}
    
    for c in clusters:
        contexts = cluster_contexts.get(c, [])
        if not contexts:
            cluster_entropies[c] = np.nan
            cluster_context_ratios[c] = np.nan
            cluster_weights[c] = 0.0
            continue
            
        _, H, N_c, K_c = context_entropy_score(contexts)
        
        gene_plasmids_in_cluster = len(cluster_gene_plasmids[c])
        total_plasmids_in_cluster = cluster_totals[c]
        weight = gene_plasmids_in_cluster / total_plasmids_in_cluster
        
        cluster_entropies[c] = H if not np.isnan(H) else np.nan
        cluster_context_ratios[c] = K_c / N_c if N_c > 0 else np.nan
        cluster_weights[c] = weight

    # Aggregate Valid Entropies
    valid_entropies = [cluster_entropies[c] for c in clusters if not np.isnan(cluster_entropies[c])]
    valid_weights = [cluster_weights[c] for c in clusters if not np.isnan(cluster_entropies[c])]

    if valid_entropies:
        mean_ent = np.mean(valid_entropies)
        median_ent = np.median(valid_entropies)
        max_ent = np.max(valid_entropies)
        weighted_mean_ent = np.average(valid_entropies, weights=valid_weights) if sum(valid_weights) > 0 else np.nan
    else:
        mean_ent, median_ent, max_ent, weighted_mean_ent = np.nan, np.nan, np.nan, np.nan

    # Aggregate Valid Context Ratios
    valid_ratios = [cluster_context_ratios[c] for c in clusters if not np.isnan(cluster_context_ratios[c])]
    valid_weights_ratios = [cluster_weights[c] for c in clusters if not np.isnan(cluster_context_ratios[c])]
    
    if valid_ratios:
        mean_ratio = np.mean(valid_ratios)
        median_ratio = np.median(valid_ratios)
        max_ratio = np.max(valid_ratios)
        weighted_mean_ratio = np.average(valid_ratios, weights=valid_weights_ratios) if sum(valid_weights_ratios) > 0 else np.nan
    else:
        mean_ratio, median_ratio, max_ratio, weighted_mean_ratio = np.nan, np.nan, np.nan, np.nan

    # Global Context Ratio
    global_contexts = [ctx for ctxs in cluster_contexts.values() for ctx in ctxs]
    global_K = len(set(global_contexts))
    global_ratio = global_K / total_copies if total_copies > 0 else 0

    record = {
        'gene_name': gene_name,
        'gene_family': family,
        'n_copies': total_copies,
        'global_unique_contexts': global_K,
        'global_context_ratio': round(global_ratio, 6),
        
        'median_cluster_entropy': round(median_ent, 6) if not np.isnan(median_ent) else np.nan,
        'mean_cluster_entropy': round(mean_ent, 6) if not np.isnan(mean_ent) else np.nan,
        'max_cluster_entropy': round(max_ent, 6) if not np.isnan(max_ent) else np.nan,
        'weighted_mean_cluster_entropy': round(weighted_mean_ent, 6) if not np.isnan(weighted_mean_ent) else np.nan,
        
        'median_cluster_context_ratio': round(median_ratio, 6) if not np.isnan(median_ratio) else np.nan,
        'mean_cluster_context_ratio': round(mean_ratio, 6) if not np.isnan(mean_ratio) else np.nan,
        'max_cluster_context_ratio': round(max_ratio, 6) if not np.isnan(max_ratio) else np.nan,
        'weighted_mean_cluster_context_ratio': round(weighted_mean_ratio, 6) if not np.isnan(weighted_mean_ratio) else np.nan
    }
    
    for c in clusters:
        record[f'entropy_cluster_{c}'] = cluster_entropies[c]
        record[f'context_ratio_cluster_{c}'] = cluster_context_ratios[c]
        record[f'prevalence_cluster_{c}'] = cluster_weights[c]

    records.append(record)

df_results = pd.DataFrame(records).sort_values('weighted_mean_cluster_entropy', ascending=False).reset_index(drop=True)

# =============================================================================
# 7. OUTPUT FOR GENES
# =============================================================================
print(f"\n── Results summary (n={len(df_results)} genes) ──")
print(f"Genes with computed weighted mean entropy: {df_results['weighted_mean_cluster_entropy'].notna().sum()}")

print("\nTop 15 by weighted mean cluster entropy:")
display_cols = ['gene_name', 'gene_family', 'n_copies', 'max_cluster_entropy', 'weighted_mean_cluster_entropy', 'max_cluster_context_ratio', 'weighted_mean_cluster_context_ratio']
print(df_results[display_cols].head(15).to_string(index=False))

out_file = out_dir / 'per_gene_bl_cluster_metrics.csv'
df_results.to_csv(out_file, index=False)
print(f"\nSaved to {out_file}")


# =============================================================================
# 8. ALL DOMAINS: CLUSTER ENTROPY & RELATIVE ENTROPY
# =============================================================================
print("\n" + "="*60)
print(" CALCULATING METRICS FOR ALL DOMAINS ACROSS CLUSTERS")
print("="*60)

print("Computing global raw entropies for background reference...")
all_domain_raw_H = {}

for dom, pos_list in tqdm(domain_positions.items(), desc='Global Raw Entropy'):
    tokens = []
    for plasmid, idx in pos_list:
        entries = plasmid_to_domains[plasmid]
        n = len(entries)
        if n <= 1: continue
        left = entries[(idx - 1) % n]
        right = entries[(idx + 1) % n]
        tokens.append((left, right))
        
    _, H, n_copies, _ = context_entropy_score(tokens)
    if n_copies >= MIN_OBS and not np.isnan(H):
        all_domain_raw_H[dom] = H

print("\nCalculating per-cluster metrics for all domains...")
all_domain_records = []

for dom, pos_list in tqdm(domain_positions.items(), desc='All-domain Cluster Metrics'):
    cluster_contexts = defaultdict(list)
    cluster_plasmids = defaultdict(set)
    total_copies = 0

    for plasmid, idx in pos_list:
        c = plasmid_jac_cluster.get(plasmid)
        if c is None: 
            continue
            
        entries = plasmid_to_domains[plasmid]
        n = len(entries)
        if n <= 1: 
            continue

        left  = entries[(idx - 1) % n]
        right = entries[(idx + 1) % n]

        cluster_contexts[c].append((left, right))
        cluster_plasmids[c].add(plasmid)
        total_copies += 1

    if total_copies < MIN_OBS:
        continue

    cluster_entropies = {}
    cluster_rel_entropies = {}
    cluster_context_ratios = {}
    cluster_weights = {}

    for c in clusters:
        contexts = cluster_contexts.get(c, [])
        domain_plasmids_in_cluster = len(cluster_plasmids.get(c, set()))
        total_plasmids_in_cluster = cluster_totals[c]
        weight = domain_plasmids_in_cluster / total_plasmids_in_cluster
        
        if not contexts:
            cluster_entropies[c] = np.nan
            cluster_rel_entropies[c] = np.nan
            cluster_context_ratios[c] = np.nan
            cluster_weights[c] = 0.0
            continue
            
        _, H, N_c, K_c = context_entropy_score(contexts)
        
        cluster_entropies[c] = H if not np.isnan(H) else np.nan
        cluster_context_ratios[c] = K_c / N_c if N_c > 0 else np.nan
        
        bg_domains = set()
        for plas in cluster_plasmids[c]:
            for d in plasmid_to_domains.get(plas, []):
                if d != dom:
                    bg_domains.add(d)
                    
        bg_raw = [all_domain_raw_H[d] for d in bg_domains if d in all_domain_raw_H]
        if len(bg_raw) >= 3 and not np.isnan(H):
            cluster_rel_entropies[c] = round(H - np.mean(bg_raw), 6)
        else:
            cluster_rel_entropies[c] = np.nan
            
        cluster_weights[c] = weight

    # Aggregate Raw Entropy
    valid_raw = [cluster_entropies[c] for c in clusters if not np.isnan(cluster_entropies[c])]
    valid_w_raw = [cluster_weights[c] for c in clusters if not np.isnan(cluster_entropies[c])]
    if valid_raw:
        mean_raw = np.mean(valid_raw)
        median_raw = np.median(valid_raw)
        max_raw = np.max(valid_raw)
        w_mean_raw = np.average(valid_raw, weights=valid_w_raw) if sum(valid_w_raw) > 0 else np.nan
    else:
        mean_raw = median_raw = max_raw = w_mean_raw = np.nan
        
    # Aggregate Relative Entropy
    valid_rel = [cluster_rel_entropies[c] for c in clusters if not np.isnan(cluster_rel_entropies[c])]
    valid_w_rel = [cluster_weights[c] for c in clusters if not np.isnan(cluster_rel_entropies[c])]
    if valid_rel:
        mean_rel = np.mean(valid_rel)
        median_rel = np.median(valid_rel)
        max_rel = np.max(valid_rel)
        w_mean_rel = np.average(valid_rel, weights=valid_w_rel) if sum(valid_w_rel) > 0 else np.nan
    else:
        mean_rel = median_rel = max_rel = w_mean_rel = np.nan

    # Aggregate Context Ratios
    valid_rat = [cluster_context_ratios[c] for c in clusters if not np.isnan(cluster_context_ratios[c])]
    valid_w_rat = [cluster_weights[c] for c in clusters if not np.isnan(cluster_context_ratios[c])]
    if valid_rat:
        mean_rat = np.mean(valid_rat)
        median_rat = np.median(valid_rat)
        max_rat = np.max(valid_rat)
        w_mean_rat = np.average(valid_rat, weights=valid_w_rat) if sum(valid_w_rat) > 0 else np.nan
    else:
        mean_rat = median_rat = max_rat = w_mean_rat = np.nan

    # Global Context Ratio
    global_contexts = [ctx for ctxs in cluster_contexts.values() for ctx in ctxs]
    global_K = len(set(global_contexts))
    global_ratio = global_K / total_copies if total_copies > 0 else 0

    rec = {
        'domain_name': dom,
        'n_copies': total_copies,
        'global_unique_contexts': global_K,
        'global_context_ratio': round(global_ratio, 6),
        
        'median_cluster_raw_entropy': round(median_raw, 6) if not np.isnan(median_raw) else np.nan,
        'mean_cluster_raw_entropy': round(mean_raw, 6) if not np.isnan(mean_raw) else np.nan,
        'max_cluster_raw_entropy': round(max_raw, 6) if not np.isnan(max_raw) else np.nan,
        'weighted_mean_cluster_raw_entropy': round(w_mean_raw, 6) if not np.isnan(w_mean_raw) else np.nan,
        
        'median_cluster_rel_entropy': round(median_rel, 6) if not np.isnan(median_rel) else np.nan,
        'mean_cluster_rel_entropy': round(mean_rel, 6) if not np.isnan(mean_rel) else np.nan,
        'max_cluster_rel_entropy': round(max_rel, 6) if not np.isnan(max_rel) else np.nan,
        'weighted_mean_cluster_rel_entropy': round(w_mean_rel, 6) if not np.isnan(w_mean_rel) else np.nan,
        
        'median_cluster_context_ratio': round(median_rat, 6) if not np.isnan(median_rat) else np.nan,
        'mean_cluster_context_ratio': round(mean_rat, 6) if not np.isnan(mean_rat) else np.nan,
        'max_cluster_context_ratio': round(max_rat, 6) if not np.isnan(max_rat) else np.nan,
        'weighted_mean_cluster_context_ratio': round(w_mean_rat, 6) if not np.isnan(w_mean_rat) else np.nan
    }
    
    for c in clusters:
        rec[f'raw_entropy_cluster_{c}'] = cluster_entropies[c]
        rec[f'rel_entropy_cluster_{c}'] = cluster_rel_entropies[c]
        rec[f'context_ratio_cluster_{c}'] = cluster_context_ratios[c]
        rec[f'prevalence_cluster_{c}'] = cluster_weights[c]
        
    all_domain_records.append(rec)

# =============================================================================
# 9. OUTPUT GENERATION
# =============================================================================
df_all_domains = pd.DataFrame(all_domain_records)
df_all_domains = df_all_domains.sort_values('weighted_mean_cluster_raw_entropy', ascending=False).reset_index(drop=True)

print(f"\n── All-Domains Results summary (n={len(df_all_domains)} domains) ──")
print(f"Domains with computed weighted raw entropy: {df_all_domains['weighted_mean_cluster_raw_entropy'].notna().sum()}")

print("\nTop 15 Domains by Weighted Mean Cluster Raw Entropy:")
display_cols = ['domain_name', 'n_copies', 'max_cluster_raw_entropy', 'weighted_mean_cluster_raw_entropy', 'max_cluster_rel_entropy', 'weighted_mean_cluster_rel_entropy']
print(df_all_domains[display_cols].head(15).to_string(index=False))

out_all_domains_file = out_dir / 'all_domains_cluster_metrics.csv'
df_all_domains.to_csv(out_all_domains_file, index=False)
print(f"\nSaved full domain dataframe to {out_all_domains_file}")








df = pd.read_csv('clustering_results/all_domains_cluster_metrics.csv')

# 1. Define the metrics to normalize
cols_to_normalize = [
    'median_cluster_raw_entropy', 'mean_cluster_raw_entropy', 'max_cluster_raw_entropy', 
    'weighted_mean_cluster_raw_entropy', 'median_cluster_rel_entropy', 'mean_cluster_rel_entropy', 
    'max_cluster_rel_entropy', 'weighted_mean_cluster_rel_entropy', 'median_cluster_context_ratio', 
    'mean_cluster_context_ratio', 'max_cluster_context_ratio', 'weighted_mean_cluster_context_ratio'
]

# 2. Calculate all new columns at once in a dictionary
norm_data = {
    f'{col}_norm_by_contexts': df[col] / df['global_unique_contexts'] 
    for col in cols_to_normalize
}

# 3. Concatenate to the main dataframe in one shot to avoid fragmentation
df = pd.concat([df, pd.DataFrame(norm_data)], axis=1)

# 4. (Optional) De-fragment the existing frame if needed
df = df.copy()

# 5. Print results for the new columns
for norm_col in norm_data.keys():
    top_30 = df.sort_values(by=[norm_col], ascending=False)['domain_name'].tolist()[:30]
    print(f'\nFor {norm_col}: {", ".join(top_30)}\n')





metrics = [
    'median_cluster_raw_entropy', 'mean_cluster_raw_entropy', 
    'max_cluster_raw_entropy', 'weighted_mean_cluster_raw_entropy', 
    'median_cluster_rel_entropy', 'mean_cluster_rel_entropy', 
    'max_cluster_rel_entropy', 'weighted_mean_cluster_rel_entropy', 
    'median_cluster_context_ratio', 'mean_cluster_context_ratio', 
    'max_cluster_context_ratio', 'weighted_mean_cluster_context_ratio'
]

# Loop through each metric and print the top 30 domains
for metric in metrics:
    # Ensure the column exists in your DataFrame
    if metric in df.columns:
        # Sort ascending (lowest values first) and grab the top 30 domain names
        top_30 = df.sort_values(by=metric, ascending=False)['domain_name'].tolist()[:30]
        print(f'For {metric}:\n{", ".join(top_30)}\n')
    else:
        print(f"Column '{metric}' not found in DataFrame.")








#
#
#CLUSTER_PATH = Path('clustering_results')
#
#
#data_dir = Path('plasmid_motif_network/intermediate')
#files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))
#df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
#df_merged = df_merged.with_columns(
#    pl.col('strand').cast(pl.Int32).alias('strand')
#)
#
#
#jac_clusters = pd.read_csv(CLUSTER_PATH / 'umap_hdbcsan_clusters.csv')
#plasmid_jac_cluster = dict(zip(jac_clusters['plasmid'], jac_clusters['cluster']))
#clusters = list(set(list(plasmid_jac_cluster.values())))
#
#for cluster in clusters:
#    plasmids = [k for k,v in plasmid_jac_cluster.items() if v == cluster]
#    df_filt = df_merged.filter(pl.col('plasmid').is_in(plasmids))
#    df_filt = df_filt.sort(['plasmid', 'start', 'ali_from'])
#    ordered = df_filt.select(['plasmid','query_name','target_name','start','ali_from','strand'])
#    adjacency = defaultdict(int)
#    df_shifted = ordered.with_columns([
#        pl.col('target_name').shift(-1).over('plasmid').alias('domain2'),
#        pl.col('strand').shift(-1).over('plasmid').alias('strand2'),
#    ]).rename({'target_name': 'domain1', 'strand': 'strand1'})
#    df_shifted = df_shifted.filter(pl.col('domain2').is_not_null())
#    wrap = ordered.group_by('plasmid').agg([
#        pl.col('target_name').last().alias('domain1'),
#        pl.col('target_name').first().alias('domain2'),
#        pl.col('strand').last().alias('strand1'),
#        pl.col('strand').first().alias('strand2'),
#    ])
#    df_edges = pl.concat([
#        df_shifted.select(['plasmid','domain1','domain2','strand1','strand2']),
#        wrap.select(['plasmid','domain1','domain2','strand1','strand2'])
#    ])
#    df_edges = df_edges.with_columns(
#        pl.when((pl.col('strand1') == 1) & (pl.col('strand2') == 1)).then(pl.lit('PP'))
#         .when((pl.col('strand1') == -1) & (pl.col('strand2') == -1)).then(pl.lit('MM'))
#         .when((pl.col('strand1') == 1) & (pl.col('strand2') == -1)).then(pl.lit('PM'))
#         .otherwise(pl.lit('MP'))
#         .alias('orientation')
#    )
#    adj_df = (
#        df_edges.group_by(['domain1', 'domain2', 'orientation'])
#        .agg(pl.len().cast(pl.Int64).alias('weight'))
#    )
#    adj_df = adj_df.with_columns(pl.when(pl.col('orientation').is_in(['PP', 'MM'])).then(pl.col('weight')).otherwise(-pl.col('weight')).alias('signed_contribution'))
#    collapsed = (
#        adj_df.group_by(['domain1', 'domain2'])
#          .agg([
#              pl.sum('signed_contribution').alias('signed_weight'),
#              pl.sum('weight').alias('total_weight')
#          ])
#    )
#    domain_to_plasmids = defaultdict(set)
#    for row in ordered.iter_rows(named=True):
#        domain_to_plasmids[row['target_name']].add(row['plasmid'])
#    G= nx.MultiDiGraph()
#    for row in adj_df.iter_rows(named=True):
#        d1 = row['domain1']
#        d2 = row['domain2']
#        weight = row['weight']
#        orientation = row['orientation']
#        G.add_edge(
#            d1,
#            d2,
#            weight=weight,
#            orientation=orientation
#        )
#    for node in G.nodes():
#        plasmid_set = domain_to_plasmids.get(node, set())
#        G.nodes[node]['label'] = node
#        G.nodes[node]['plasmid_count'] = len(plasmid_set)
#        G.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
#    for global_id, (u, v, k) in enumerate(G.edges(keys=True)):
#        G[u][v][k]['id'] = str(global_id)
#    nx.write_graphml(G, CLUSTER_PATH / f'cluster_{cluster}_domain_architecture_network.graphml', edge_id_from_attribute='id')
#    F = nx.DiGraph()
#    for row in collapsed.iter_rows(named=True):
#        d1 = row['domain1']
#        d2 = row['domain2']
#        signed_weight = row['signed_weight']
#        total_weight = row['total_weight']
#        F.add_edge(
#            d1,
#            d2,
#            signed_weight=signed_weight,
#            total_weight=total_weight
#        )
#    for node in F.nodes():
#        plasmid_set = domain_to_plasmids.get(node, set())
#        F.nodes[node]['label'] = node
#        F.nodes[node]['plasmid_count'] = len(plasmid_set)
#        F.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
#    nx.write_graphml(F, CLUSTER_PATH / f'cluster_{cluster}_domain_architecture_signed_network.graphml')
#    
#
#
#
#import os
#import re 
#import math
#import warnings
#import pickle
#import random
#import sys
#from collections import Counter, defaultdict
#from pathlib import Path
#
#import numpy as np
#import pandas as pd
#import polars as pl
#import networkx as nx
#import leidenalg
#from datasketch import MinHash, MinHashLSH
#from tqdm import tqdm
#
#warnings.filterwarnings('ignore')
#
## =============================================================================
## 1. SETUP & CONFIGURATION
## =============================================================================
#
#CLUSTER_PATH = Path('clustering_results')
#graph_dir    = CLUSTER_PATH
#out_dir      = CLUSTER_PATH
#
#CLUSTER_PATH.mkdir(exist_ok=True)
#out_dir.mkdir(exist_ok=True)
#
#MERGED_FASTA_DIR = Path('merged_nonoverlapping_fastas')
#MIN_OBS = 10  # minimum copies for entropy/relative-entropy to be computed
#
#
#
#data_dir = Path('plasmid_motif_network/intermediate')
#files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))
#df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
#df_merged = df_merged.with_columns(
#    pl.col('strand').cast(pl.Int32).alias('strand')
#)
#
#
#
#jac_clusters = pd.read_csv(CLUSTER_PATH / 'umap_hdbscan_clusters.csv')
#plasmid_jac_cluster = dict(zip(jac_clusters['plasmid'], jac_clusters['cluster']))
#clusters = sorted(list(set(plasmid_jac_cluster.values())))
#cluster_totals = Counter(plasmid_jac_cluster.values())
#
#
#
#print("Generating domain architecture networks per cluster...")
#for cluster in tqdm(clusters, desc="Clusters"):
#    plasmids = [k for k,v in plasmid_jac_cluster.items() if v == cluster]
#    df_filt = df_merged.filter(pl.col('plasmid').is_in(plasmids))
#    df_filt = df_filt.sort(['plasmid', 'start', 'ali_from'])
#    ordered = df_filt.select(['plasmid','query_name','target_name','start','ali_from','strand'])
#    adjacency = defaultdict(int)
#    df_shifted = ordered.with_columns([
#        pl.col('target_name').shift(-1).over('plasmid').alias('domain2'),
#        pl.col('strand').shift(-1).over('plasmid').alias('strand2'),
#    ]).rename({'target_name': 'domain1', 'strand': 'strand1'})
#    df_shifted = df_shifted.filter(pl.col('domain2').is_not_null())
#    wrap = ordered.group_by('plasmid').agg([
#        pl.col('target_name').last().alias('domain1'),
#        pl.col('target_name').first().alias('domain2'),
#        pl.col('strand').last().alias('strand1'),
#        pl.col('strand').first().alias('strand2'),
#    ])
#    df_edges = pl.concat([
#        df_shifted.select(['plasmid','domain1','domain2','strand1','strand2']),
#        wrap.select(['plasmid','domain1','domain2','strand1','strand2'])
#    ])
#    df_edges = df_edges.with_columns(
#        pl.when((pl.col('strand1') == 1) & (pl.col('strand2') == 1)).then(pl.lit('PP'))
#         .when((pl.col('strand1') == -1) & (pl.col('strand2') == -1)).then(pl.lit('MM'))
#         .when((pl.col('strand1') == 1) & (pl.col('strand2') == -1)).then(pl.lit('PM'))
#         .otherwise(pl.lit('MP'))
#         .alias('orientation')
#    )
#    adj_df = (
#        df_edges.group_by(['domain1', 'domain2', 'orientation'])
#        .agg(pl.len().cast(pl.Int64).alias('weight'))
#    )
#    adj_df = adj_df.with_columns(pl.when(pl.col('orientation').is_in(['PP', 'MM'])).then(pl.col('weight')).otherwise(-pl.col('weight')).alias('signed_contribution'))
#    collapsed = (
#        adj_df.group_by(['domain1', 'domain2'])
#          .agg([
#              pl.sum('signed_contribution').alias('signed_weight'),
#              pl.sum('weight').alias('total_weight')
#          ])
#    )
#    domain_to_plasmids = defaultdict(set)
#    for row in ordered.iter_rows(named=True):
#        domain_to_plasmids[row['target_name']].add(row['plasmid'])
#    G = nx.MultiDiGraph()
#    for row in adj_df.iter_rows(named=True):
#        G.add_edge(
#            row['domain1'], row['domain2'],
#            weight=row['weight'], orientation=row['orientation']
#        )
#    for node in G.nodes():
#        plasmid_set = domain_to_plasmids.get(node, set())
#        G.nodes[node]['label'] = node
#        G.nodes[node]['plasmid_count'] = len(plasmid_set)
#        G.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
#    for global_id, (u, v, k) in enumerate(G.edges(keys=True)):
#        G[u][v][k]['id'] = str(global_id)
#    nx.write_graphml(G, CLUSTER_PATH / f'cluster_{cluster}_domain_architecture_network.graphml', edge_id_from_attribute='id')
#    F = nx.DiGraph()
#    for row in collapsed.iter_rows(named=True):
#        F.add_edge(
#            row['domain1'], row['domain2'],
#            signed_weight=row['signed_weight'], total_weight=row['total_weight']
#        )
#    for node in F.nodes():
#        plasmid_set = domain_to_plasmids.get(node, set())
#        F.nodes[node]['label'] = node
#        F.nodes[node]['plasmid_count'] = len(plasmid_set)
#        F.nodes[node]['plasmids'] = ';'.join(sorted(plasmid_set))
#    nx.write_graphml(F, CLUSTER_PATH / f'cluster_{cluster}_domain_architecture_signed_network.graphml')
#
#
#
#
#
#
#ordered_df = df_merged.sort(['plasmid', 'start', 'ali_from']).select(['plasmid', 'target_name'])
#
#print("Building circular domain lists...")
#plasmid_to_domains = defaultdict(list)
#for row in ordered_df.iter_rows(named=True):
#    plasmid_to_domains[row['plasmid']].append(row['target_name'])
#
#domain_positions = defaultdict(list)
#for plasmid, doms in plasmid_to_domains.items():
#    n = len(doms)
#    if n <= 1:
#        continue
#    for i, dom in enumerate(doms):
#        domain_positions[dom].append((plasmid, i))
#
#print(f"Loaded {len(domain_positions)} unique domains across {len(plasmid_to_domains)} plasmids.")
#
#test = pd.read_csv('amrfindermapped_beta_lactamases.csv')
#all_gene_names = [x for x in test['gene_name'].unique() if isinstance(x, str)]
#gene_to_family = {
#    gname: test.loc[test['gene_name'] == gname, 'gene_family'].iloc[0]
#    for gname in all_gene_names
#}
#
#gene_to_pids = defaultdict(set)
#for _, row in test.iterrows():
#    if isinstance(row['gene_name'], str):
#        gene_to_pids[row['gene_name']].add(row['query_id'])
#
#print(f"Loaded {len(all_gene_names)} BL gene names from AMRFinder mapping.")
#
## =============================================================================
## 5. ENTROPY LOGIC
## =============================================================================
#
#def context_entropy_score(contexts):
#    """
#    contexts : list of (left_domain, right_domain) tuples
#    Returns  : (penalised_score, raw_H, N, K)
#    """
#    N = len(contexts)
#    if N <= 1:
#        return np.nan, np.nan, N, 0
#    counts = Counter(contexts)
#    K = len(counts)
#    if K <= 1:
#        return 0.0, 0.0, N, K
#    probs = np.array(list(counts.values())) / N
#    H = -np.sum(probs * np.log2(probs))
#    H_context = H / np.log2(K)
#    score = H_context * (K / N)
#    return round(score, 6), round(H, 6), N, K
#
## Build PID → positions lookup
#pid_to_positions = defaultdict(list)
#plasmid_index_counter = defaultdict(int)
#
#ordered_full = df_merged.sort(['plasmid', 'start', 'ali_from']).select(['query_name', 'plasmid', 'target_name'])
#for row in ordered_full.iter_rows(named=True):
#    plasmid = row['plasmid']
#    idx = plasmid_index_counter[plasmid]
#    plasmid_index_counter[plasmid] += 1
#    pid_to_positions[row['query_name']].append((plasmid, idx, row['target_name']))
#
## =============================================================================
## 6. PER-GENE CLUSTER METRICS
## =============================================================================
#print("\nCalculating per-gene cluster entropy metrics...")
#
#records = []
#
#for gene_name in tqdm(all_gene_names, desc='Per-gene'):
#    pids_this_gene = gene_to_pids[gene_name]
#    family = gene_to_family.get(gene_name, '')
#    
#    # Track contexts and plasmids per cluster
#    cluster_contexts = defaultdict(list)
#    cluster_gene_plasmids = defaultdict(set)
#    total_copies = 0
#
#    for pid in pids_this_gene:
#        for plasmid, idx, dom in pid_to_positions.get(pid, []):
#            cluster = plasmid_jac_cluster.get(plasmid)
#            if cluster is None:
#                continue
#                
#            entries = plasmid_to_domains[plasmid]
#            n = len(entries)
#            if n <= 1:
#                continue
#
#            left  = entries[(idx - 1) % n]
#            right = entries[(idx + 1) % n]
#
#            cluster_contexts[cluster].append((left, right))
#            cluster_gene_plasmids[cluster].add(plasmid)
#            total_copies += 1
#
#    if total_copies == 0:
#        continue # Skip if entirely absent
#
#    # Calculate cluster-specific metrics
#    cluster_entropies = {}
#    cluster_weights = {}
#    
#    for c in clusters:
#        contexts = cluster_contexts.get(c, [])
#        if not contexts:
#            cluster_entropies[c] = np.nan
#            cluster_weights[c] = 0.0
#            continue
#            
#        _, H, _, _ = context_entropy_score(contexts)
#        
#        # Calculate prevalence weight: (plasmids with gene in cluster) / (total plasmids in cluster)
#        gene_plasmids_in_cluster = len(cluster_gene_plasmids[c])
#        total_plasmids_in_cluster = cluster_totals[c]
#        
#        weight = gene_plasmids_in_cluster / total_plasmids_in_cluster
#        
#        cluster_entropies[c] = H if not np.isnan(H) else np.nan
#        cluster_weights[c] = weight
#
#    # Aggregate valid entropies
#    valid_entropies = [cluster_entropies[c] for c in clusters if not np.isnan(cluster_entropies[c])]
#    valid_weights = [cluster_weights[c] for c in clusters if not np.isnan(cluster_entropies[c])]
#
#    if valid_entropies:
#        mean_ent = np.mean(valid_entropies)
#        median_ent = np.median(valid_entropies)
#        # Weighted mean calculation
#        if sum(valid_weights) > 0:
#            weighted_mean_ent = np.average(valid_entropies, weights=valid_weights)
#        else:
#            weighted_mean_ent = np.nan
#    else:
#        mean_ent, median_ent, weighted_mean_ent = np.nan, np.nan, np.nan
#
#    # Build the record
#    record = {
#        'gene_name': gene_name,
#        'gene_family': family,
#        'n_copies': total_copies,
#        'median_cluster_entropy': median_ent,
#        'mean_cluster_entropy': mean_ent,
#        'weighted_mean_cluster_entropy': weighted_mean_ent
#    }
#    
#    # Append per-cluster values
#    for c in clusters:
#        record[f'entropy_cluster_{c}'] = cluster_entropies[c]
#        record[f'prevalence_cluster_{c}'] = cluster_weights[c]
#
#    records.append(record)
#
#df_results = pd.DataFrame(records).sort_values('weighted_mean_cluster_entropy', ascending=False).reset_index(drop=True)
#
## =============================================================================
## 7. OUTPUT
## =============================================================================
#
#print(f"\n── Results summary (n={len(df_results)} genes) ──")
#print(f"Genes with computed weighted mean entropy: {df_results['weighted_mean_cluster_entropy'].notna().sum()}")
#
## Print top 15 by weighted mean
#print("\nTop 15 by weighted mean cluster entropy:")
#display_cols = ['gene_name', 'gene_family', 'n_copies', 'median_cluster_entropy', 'mean_cluster_entropy', 'weighted_mean_cluster_entropy']
#print(df_results[display_cols].head(15).to_string(index=False))
#
#out_file = out_dir / 'per_gene_bl_cluster_metrics.csv'
#df_results.to_csv(out_file, index=False)
#print(f"\nSaved to {out_file}")
#
#
## =============================================================================
## 8. ALL DOMAINS: CLUSTER ENTROPY & RELATIVE ENTROPY
## =============================================================================
#print("\n" + "="*60)
#print(" CALCULATING METRICS FOR ALL DOMAINS ACROSS CLUSTERS")
#print("="*60)
#
## 8a. Calculate global background raw entropies (used for relative entropy)
#print("Computing global raw entropies for background reference...")
#all_domain_raw_H = {}
#
#for dom, pos_list in tqdm(domain_positions.items(), desc='Global Raw Entropy'):
#    tokens = []
#    for plasmid, idx in pos_list:
#        entries = plasmid_to_domains[plasmid]
#        n = len(entries)
#        if n <= 1: continue
#        left = entries[(idx - 1) % n]
#        right = entries[(idx + 1) % n]
#        tokens.append((left, right))
#        
#    _, H, n_copies, _ = context_entropy_score(tokens)
#    if n_copies >= MIN_OBS and not np.isnan(H):
#        all_domain_raw_H[dom] = H
#
## 8b. Calculate per-cluster metrics for every domain
#print("\nCalculating per-cluster raw and relative entropies for all domains...")
#all_domain_records = []
#
#for dom, pos_list in tqdm(domain_positions.items(), desc='All-domain Cluster Metrics'):
#    cluster_contexts = defaultdict(list)
#    cluster_plasmids = defaultdict(set)
#    total_copies = 0
#
#    # Distribute domain occurrences into their respective clusters
#    for plasmid, idx in pos_list:
#        c = plasmid_jac_cluster.get(plasmid)
#        if c is None: 
#            continue
#            
#        entries = plasmid_to_domains[plasmid]
#        n = len(entries)
#        if n <= 1: 
#            continue
#
#        left  = entries[(idx - 1) % n]
#        right = entries[(idx + 1) % n]
#
#        cluster_contexts[c].append((left, right))
#        cluster_plasmids[c].add(plasmid)
#        total_copies += 1
#
#    # Skip domains with too few total observations
#    if total_copies < MIN_OBS:
#        continue
#
#    cluster_entropies = {}
#    cluster_rel_entropies = {}
#    cluster_weights = {}
#
#    for c in clusters:
#        contexts = cluster_contexts.get(c, [])
#        domain_plasmids_in_cluster = len(cluster_plasmids.get(c, set()))
#        total_plasmids_in_cluster = cluster_totals[c]
#        
#        # Prevalence of this domain within the specific cluster
#        weight = domain_plasmids_in_cluster / total_plasmids_in_cluster
#        
#        if not contexts:
#            cluster_entropies[c] = np.nan
#            cluster_rel_entropies[c] = np.nan
#            cluster_weights[c] = 0.0
#            continue
#            
#        # Raw Entropy for this cluster
#        _, H, _, _ = context_entropy_score(contexts)
#        cluster_entropies[c] = H if not np.isnan(H) else np.nan
#        
#        # Relative Entropy for this cluster (vs. background of co-occurring domains)
#        bg_domains = set()
#        for plas in cluster_plasmids[c]:
#            for d in plasmid_to_domains.get(plas, []):
#                if d != dom:
#                    bg_domains.add(d)
#                    
#        bg_raw = [all_domain_raw_H[d] for d in bg_domains if d in all_domain_raw_H]
#        if len(bg_raw) >= 3 and not np.isnan(H):
#            cluster_rel_entropies[c] = round(H - np.mean(bg_raw), 6)
#        else:
#            cluster_rel_entropies[c] = np.nan
#            
#        cluster_weights[c] = weight
#
#    # Aggregate Raw Entropy Metrics
#    valid_raw = [cluster_entropies[c] for c in clusters if not np.isnan(cluster_entropies[c])]
#    valid_weights_raw = [cluster_weights[c] for c in clusters if not np.isnan(cluster_entropies[c])]
#    
#    if valid_raw:
#        mean_raw = np.mean(valid_raw)
#        median_raw = np.median(valid_raw)
#        w_mean_raw = np.average(valid_raw, weights=valid_weights_raw) if sum(valid_weights_raw) > 0 else np.nan
#    else:
#        mean_raw = median_raw = w_mean_raw = np.nan
#        
#    # Aggregate Relative Entropy Metrics
#    valid_rel = [cluster_rel_entropies[c] for c in clusters if not np.isnan(cluster_rel_entropies[c])]
#    valid_weights_rel = [cluster_weights[c] for c in clusters if not np.isnan(cluster_rel_entropies[c])]
#    
#    if valid_rel:
#        mean_rel = np.mean(valid_rel)
#        median_rel = np.median(valid_rel)
#        w_mean_rel = np.average(valid_rel, weights=valid_weights_rel) if sum(valid_weights_rel) > 0 else np.nan
#    else:
#        mean_rel = median_rel = w_mean_rel = np.nan
#
#    # Store all statistics for this domain
#    rec = {
#        'domain_name': dom,
#        'n_copies': total_copies,
#        'median_cluster_raw_entropy': round(median_raw, 6) if not np.isnan(median_raw) else np.nan,
#        'mean_cluster_raw_entropy': round(mean_raw, 6) if not np.isnan(mean_raw) else np.nan,
#        'weighted_mean_cluster_raw_entropy': round(w_mean_raw, 6) if not np.isnan(w_mean_raw) else np.nan,
#        'median_cluster_rel_entropy': round(median_rel, 6) if not np.isnan(median_rel) else np.nan,
#        'mean_cluster_rel_entropy': round(mean_rel, 6) if not np.isnan(mean_rel) else np.nan,
#        'weighted_mean_cluster_rel_entropy': round(w_mean_rel, 6) if not np.isnan(w_mean_rel) else np.nan
#    }
#    
#    # Append the per-cluster breakdowns
#    for c in clusters:
#        rec[f'raw_entropy_cluster_{c}'] = cluster_entropies[c]
#        rec[f'rel_entropy_cluster_{c}'] = cluster_rel_entropies[c]
#        rec[f'prevalence_cluster_{c}'] = cluster_weights[c]
#        
#    all_domain_records.append(rec)
#
## =============================================================================
## 9. OUTPUT GENERATION
## =============================================================================
#
#df_all_domains = pd.DataFrame(all_domain_records)
#df_all_domains = df_all_domains.sort_values('weighted_mean_cluster_raw_entropy', ascending=False).reset_index(drop=True)
#
#print(f"\n── All-Domains Results summary (n={len(df_all_domains)} domains) ──")
#print(f"Domains with computed weighted raw entropy: {df_all_domains['weighted_mean_cluster_raw_entropy'].notna().sum()}")
#print(f"Domains with computed weighted rel entropy: {df_all_domains['weighted_mean_cluster_rel_entropy'].notna().sum()}")
#
## Print top 15 domains by weighted mean raw entropy
#print("\nTop 15 Domains by Weighted Mean Cluster Raw Entropy:")
#display_cols = ['domain_name', 'n_copies', 'weighted_mean_cluster_raw_entropy', 'weighted_mean_cluster_rel_entropy', 'mean_cluster_raw_entropy', 'median_cluster_raw_entropy']
#print(df_all_domains[display_cols].head(15).to_string(index=False))
#
## Save the final wide dataframe
#out_all_domains_file = out_dir / 'all_domains_cluster_metrics.csv'
#df_all_domains.to_csv(out_all_domains_file, index=False)
#print(f"\nSaved full domain dataframe to {out_all_domains_file}")








































































