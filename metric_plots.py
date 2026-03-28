#In the script, a gene is selected for highlighting (marked in bold red) if it meets the following condition:
#
#    It ranks in the Top 20% of scores for a given metric. (For metrics like Prevalence and Mobility, "top" means the highest numbers. For Entropy, "top" means the lowest numbers, indicating strict conservation).
#
#    It achieves this "Top 20%" status across at least 3 of the 5 core metrics displayed in your main heatmap (Raw Entropy, Mobility, In-Data Prevalence,


"""
writeup_plots_25.py
===================
Simplified plotting script for beta-lactamase gene prioritisation.
Omits hub metrics due to Pfam-domain amalgamation logical flaw.
Generates heatmaps, scatter grids, and barplots with clean English labels and tiered highlighting.
Calculates overall and cluster-level raw/relative entropies dynamically on a per-sequence (PID) basis.
"""

import os
import math
import itertools
import warnings
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# 0. PATHS, PARAMETERS & ENGLISH LABELS
# =============================================================================
OUT = Path('writeup_plots_25')
OUT.mkdir(exist_ok=True)
MIN_OBS = 10

# Ranking direction: False = Descending (Highest value is 'best'/rank 1)
RANK_ASCENDING = {
    'overall_raw_entropy': False,
    'overall_relative_entropy': False,
    'mean_cluster_raw_entropy': False,
    'mean_cluster_rel_entropy': False,
    'dup_rate': False,          
    'pool_rate': False,         
    'pct_mobile': False,        
    'in_data_prevalence': False,
    'NCBI_Plasmid': False,
    'NCBI_WGS': False,
    'NCBI_Chromosome': False,
    'simpson_diversity': False
}

# English translations for all metrics
METRIC_NAMES = {
    'overall_raw_entropy': 'Overall Raw Entropy',
    'overall_relative_entropy': 'Overall Relative Entropy',
    'mean_cluster_raw_entropy': 'Mean Cluster Raw Entropy',
    'mean_cluster_rel_entropy': 'Mean Cluster Relative Entropy',
    'dup_rate': 'Duplication Rate',
    'pool_rate': 'MGE-Association Rate',
    'pct_mobile': 'Mobile Plasmid Association (%)',
    'in_data_prevalence': 'In-Data Prevalence',
    'NCBI_Plasmid': 'NCBI Plasmid Prevalence',
    'NCBI_WGS': 'NCBI WGS Prevalence',
    'NCBI_Chromosome': 'NCBI Chromosome Prevalence',
    'simpson_diversity': 'Species Spread (Simpson Diversity)'
}

def context_entropy(contexts):
    N = len(contexts)
    if N <= 1: return np.nan, N
    counts = Counter(contexts)
    probs = np.array(list(counts.values())) / N
    H = -np.sum(probs * np.log2(probs))
    return round(H, 6), N

# =============================================================================
# 1. LOAD DATA & RECALCULATE ENTROPY/DUPLICATION ON A SEQUENCE BASIS
# =============================================================================
print("Loading cluster assignments...")
jac_clusters_file = Path('clustering_results/umap_hdbscan_clusters.csv')
plasmid_jac_cluster = {}
if jac_clusters_file.exists():
    jac_clusters = pd.read_csv(jac_clusters_file)
    if 'plasmid' in jac_clusters.columns and 'cluster' in jac_clusters.columns:
        plasmid_jac_cluster = dict(zip(jac_clusters['plasmid'], jac_clusters['cluster']))
else:
    print("Warning: clustering_results/umap_hdbscan_clusters.csv not found. Cluster metrics will be NaN.")

print("Recalculating clean overall/cluster entropy and duplication rates...")
data_dir = Path('plasmid_motif_network/intermediate')
files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))
df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()

ordered_df = df_merged.sort(['plasmid', 'start', 'ali_from']).select(['plasmid', 'target_name', 'query_name'])
plasmid_to_domains = defaultdict(list)
pid_to_positions = defaultdict(list)
plasmid_index_counter = defaultdict(int)

for row in ordered_df.iter_rows(named=True):
    plasmid = row['plasmid']
    dom = row['target_name']
    pid = row['query_name']
    
    idx = plasmid_index_counter[plasmid]
    plasmid_index_counter[plasmid] += 1
    
    plasmid_to_domains[plasmid].append(dom)
    if pid:
        pid_to_positions[pid].append((plasmid, idx, dom))

total_plasmids = len(plasmid_to_domains)

all_domain_raw_H = {}
for plas, doms in tqdm(plasmid_to_domains.items(), desc='Background Entropy'):
    n = len(doms)
    if n <= 1: continue
    for i, dom in enumerate(doms):
        if dom not in all_domain_raw_H:
            all_domain_raw_H[dom] = []
        all_domain_raw_H[dom].append((doms[(i - 1) % n], doms[(i + 1) % n]))

for dom, ctxs in all_domain_raw_H.items():
    H, N = context_entropy(ctxs)
    all_domain_raw_H[dom] = H if N >= MIN_OBS else np.nan

test = pd.read_csv('amrfindermapped_beta_lactamases.csv')
gene_to_pids = defaultdict(set)
for _, row in test.iterrows():
    if isinstance(row['gene_name'], str):
        gene_to_pids[row['gene_name']].add(row['query_id'])

overall_records = []
for gene_name in tqdm(list(gene_to_pids.keys()), desc='Overall Gene Metrics'):
    pids = gene_to_pids[gene_name]
    contexts, plasmids_seen, pfam_domains_this_gene = [], set(), set()
    dup_events, total_tokens = 0, 0
    
    cluster_contexts = defaultdict(list)
    cluster_plasmids = defaultdict(set)
    
    # Sequence-basis operation: Iterate over each PID associated with the gene
    for pid in pids:
        for plasmid, idx, dom in pid_to_positions.get(pid, []):
            pfam_domains_this_gene.add(dom)
            entries = plasmid_to_domains[plasmid]
            n = len(entries)
            if n <= 1: continue
            
            # Extract flanking domains for this specific sequence
            left, right = entries[(idx - 1) % n], entries[(idx + 1) % n]
            contexts.append((left, right))
            plasmids_seen.add(plasmid)
            
            if left in pfam_domains_this_gene or 'lactamase' in left.lower(): dup_events += 1
            if right in pfam_domains_this_gene or 'lactamase' in right.lower(): dup_events += 1
            total_tokens += 1
            
            c = plasmid_jac_cluster.get(plasmid)
            if c is not None:
                cluster_contexts[c].append((left, right))
                cluster_plasmids[c].add(plasmid)
            
    dup_rate = dup_events / total_tokens if total_tokens > 0 else 0.0
    
    # --- Overall Entropy ---
    H, N = context_entropy(contexts)
    bg_domains = {d for plas in plasmids_seen for d in plasmid_to_domains[plas] if d not in pfam_domains_this_gene}
    bg_raw = [all_domain_raw_H.get(d, np.nan) for d in bg_domains]
    bg_raw = [x for x in bg_raw if not np.isnan(x)]
    
    bg_mean = np.mean(bg_raw) if len(bg_raw) >= 3 else np.nan
    rel_H = round(H / bg_mean, 6) if not np.isnan(H) and not np.isnan(bg_mean) and bg_mean > 0 else np.nan
    
    # --- Cluster Entropy ---
    cluster_raw_H_list = []
    cluster_rel_H_list = []
    
    for c, ctxs in cluster_contexts.items():
        H_c, N_c = context_entropy(ctxs)
        if not np.isnan(H_c):
            cluster_raw_H_list.append(H_c)
            
            bg_doms_c = set()
            for plas in cluster_plasmids[c]:
                for d in plasmid_to_domains[plas]:
                    if d not in pfam_domains_this_gene:
                        bg_doms_c.add(d)
                        
            bg_raw_c = [all_domain_raw_H.get(d, np.nan) for d in bg_doms_c]
            bg_raw_c = [x for x in bg_raw_c if not np.isnan(x)]
            
            bg_mean_c = np.mean(bg_raw_c) if len(bg_raw_c) >= 3 else np.nan
            if not np.isnan(bg_mean_c) and bg_mean_c > 0:
                cluster_rel_H_list.append(H_c / bg_mean_c)

    mean_cluster_raw_entropy = round(np.mean(cluster_raw_H_list), 6) if cluster_raw_H_list else np.nan
    mean_cluster_rel_entropy = round(np.mean(cluster_rel_H_list), 6) if cluster_rel_H_list else np.nan
    
    overall_records.append({
        'gene_name': gene_name,
        'overall_raw_entropy': H if N >= MIN_OBS else np.nan,
        'overall_relative_entropy': rel_H,
        'mean_cluster_raw_entropy': mean_cluster_raw_entropy,
        'mean_cluster_rel_entropy': mean_cluster_rel_entropy,
        'dup_rate': dup_rate,
        'in_data_prevalence': len(plasmids_seen) / total_plasmids if total_plasmids else 0
    })

df_master = pd.DataFrame(overall_records)

# =============================================================================
# 2. LOAD EXTERNAL METRICS & DEDUPLICATE
# =============================================================================
print("Merging metadata...")

if Path('mge_association_results/per_gene_mge_association.csv').exists():
    df_mge = pd.read_csv('mge_association_results/per_gene_mge_association.csv')
    df_mge = df_mge.loc[:, ~df_mge.columns.duplicated()]
    if 'gene_name' not in df_mge.columns and 'label' in df_mge.columns: 
        df_mge.rename(columns={'label': 'gene_name'}, inplace=True)
    if 'gene_name' in df_mge.columns and 'pool_rate' in df_mge.columns:
        df_mge = df_mge.drop_duplicates(subset=['gene_name'])
        df_master = df_master.merge(df_mge[['gene_name', 'pool_rate']], on='gene_name', how='left')

if Path('beta_lactamase_mobility_stats_final.csv').exists():
    df_mob = pd.read_csv('beta_lactamase_mobility_stats_final.csv')
    df_mob = df_mob.loc[:, ~df_mob.columns.duplicated()]
    if 'gene_name' not in df_mob.columns and 'gene' in df_mob.columns: 
        df_mob.rename(columns={'gene': 'gene_name'}, inplace=True)
    df_mob['pct_mobile'] = df_mob.get('pct_conjugative', 0) + df_mob.get('pct_mobilizable', 0)
    if 'gene_name' in df_mob.columns and 'pct_mobile' in df_mob.columns:
        df_mob = df_mob.drop_duplicates(subset=['gene_name'])
        df_master = df_master.merge(df_mob[['gene_name', 'pct_mobile']], on='gene_name', how='left')

if Path('beta_lactamase_species_breadth_final.csv').exists():
    df_spc = pd.read_csv('beta_lactamase_species_breadth_final.csv')
    df_spc = df_spc.loc[:, ~df_spc.columns.duplicated()]
    if 'gene_name' not in df_spc.columns:
        if 'label' in df_spc.columns: df_spc.rename(columns={'label': 'gene_name'}, inplace=True)
        elif 'gene' in df_spc.columns: df_spc.rename(columns={'gene': 'gene_name'}, inplace=True)
    if 'gene_name' in df_spc.columns and 'simpson_diversity' in df_spc.columns:
        df_spc = df_spc.drop_duplicates(subset=['gene_name'])
        df_master = df_master.merge(df_spc[['gene_name', 'simpson_diversity']], on='gene_name', how='left')

if Path('card_prevalence.txt').exists():
    df_card = pd.read_csv('card_prevalence.txt', sep='\t')
    df_card = df_card.loc[:, ~df_card.columns.duplicated()]
    if 'gene_name' not in df_card.columns and 'Name' in df_card.columns:
        df_card.rename(columns={'Name': 'gene_name'}, inplace=True)
    df_card.rename(columns={'NCBI Plasmid': 'NCBI_Plasmid', 'NCBI WGS': 'NCBI_WGS', 'NCBI Chromosome': 'NCBI_Chromosome'}, inplace=True)
    card_cols = ['gene_name', 'NCBI_Plasmid', 'NCBI_WGS', 'NCBI_Chromosome']
    avail_card_cols = [c for c in card_cols if c in df_card.columns]
    if 'gene_name' in avail_card_cols:
        df_card = df_card.drop_duplicates(subset=['gene_name'])
        df_master = df_master.merge(df_card[avail_card_cols], on='gene_name', how='left')

df_master = df_master.drop_duplicates(subset=['gene_name']).reset_index(drop=True)
df_master.to_csv(OUT / 'master_metrics_simplified.csv', index=False)


# =============================================================================
# 3. PREPARE HIGHLIGHT GENES
# =============================================================================
heatmap_metrics = [
    'overall_raw_entropy', 
    'mean_cluster_raw_entropy', 
    'mean_cluster_rel_entropy', 
    'pct_mobile', 
    'in_data_prevalence', 
    'NCBI_WGS', 
    'simpson_diversity'
]
avail_heatmap_metrics = [m for m in heatmap_metrics if m in df_master.columns]

highlight_counts = Counter()
for m in avail_heatmap_metrics:
    asc = RANK_ASCENDING.get(m, False)
    sub = df_master[['gene_name', m]].fillna(0)
    if not sub.empty:
        n_top = max(1, len(sub) // 5) # Top 20%
        for g in sub.sort_values(m, ascending=asc).head(n_top)['gene_name']:
            highlight_counts[g] += 1

# Tier 1: Top 20% in at least 3 metrics
highlight_genes = {g for g, c in highlight_counts.items() if c >= 3}

# Tier 2: The absolute top 3 genes overall (calculated via mean_rank on the selected metrics)
rank_df = df_master[['gene_name'] + avail_heatmap_metrics].dropna(how='all', subset=avail_heatmap_metrics).copy()
for m in avail_heatmap_metrics:
    asc = RANK_ASCENDING.get(m, False)
    rank_df[m + '_rank'] = rank_df[m].rank(ascending=asc, na_option='bottom')
rank_df['mean_rank'] = rank_df[[m + '_rank' for m in avail_heatmap_metrics]].mean(axis=1)
top_3_genes = set(rank_df.sort_values('mean_rank').head(3)['gene_name'])


# =============================================================================
# 4. PLOTTING
# =============================================================================
print("Generating Plots...")

# --- 1. Main Heatmap ---
sub_hm = df_master[['gene_name'] + avail_heatmap_metrics].copy().dropna(how='all', subset=avail_heatmap_metrics)
if not sub_hm.empty:
    for m in avail_heatmap_metrics:
        asc = RANK_ASCENDING.get(m, False)
        sub_hm[m + '_rank'] = sub_hm[m].rank(ascending=asc, na_option='bottom')
    
    sub_hm['mean_rank'] = sub_hm[[m + '_rank' for m in avail_heatmap_metrics]].mean(axis=1)
    top_hm = sub_hm.sort_values('mean_rank').head(50).set_index('gene_name')[avail_heatmap_metrics]
    
    top_hm = top_hm.rename(columns=METRIC_NAMES)
    plot_data = (top_hm - top_hm.mean()) / top_hm.std() 
    
    plt.figure(figsize=(11, 12))
    ax = sns.heatmap(plot_data, cmap='viridis', cbar_kws={'label': 'Z-score (Ranked)'})
    plt.title('Top 50 β-Lactamase Genes\n(★ = Overall Top 3 Genes)', fontsize=14, pad=15)
    
    # Update labels with stars and red color
    labels = [t.get_text() for t in ax.get_yticklabels()]
    new_labels = [f"{g} ★" if g in top_3_genes else g for g in labels]
    ax.set_yticklabels(new_labels)
    
    for tick_label, orig_gene in zip(ax.get_yticklabels(), labels):
        if orig_gene in highlight_genes:
            tick_label.set_color('#c0392b')
            tick_label.set_fontweight('bold')
            
    plt.tight_layout()
    plt.savefig(OUT / 'heatmap_1_selected_metrics.png', dpi=150)
    plt.close()

# --- 2. Spearman Heatmap ---
avail_sp = [m for m in ['overall_raw_entropy', 'overall_relative_entropy', 'mean_cluster_raw_entropy', 'mean_cluster_rel_entropy', 'dup_rate', 'pool_rate'] if m in df_master.columns]
if avail_sp:
    corr = df_master[avail_sp].corr(method='spearman')
    corr.columns = [METRIC_NAMES.get(c, c) for c in corr.columns]
    corr.index = [METRIC_NAMES.get(c, c) for c in corr.index]
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt='.2f', mask=mask, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Spearman Correlation (Entropy, Duplication & MGE)', pad=20)
    plt.tight_layout()
    plt.savefig(OUT / 'spearman_A_overall.png', dpi=150)
    plt.close()

# --- 3. Scatter Correlation Plots ---
all_scatter_metrics = [
    'overall_raw_entropy', 'overall_relative_entropy', 
    'mean_cluster_raw_entropy', 'mean_cluster_rel_entropy',
    'dup_rate', 'pool_rate', 'pct_mobile', 'in_data_prevalence', 
    'NCBI_Plasmid', 'NCBI_WGS', 'NCBI_Chromosome', 'simpson_diversity'
]
avail_sc = [m for m in all_scatter_metrics if m in df_master.columns]

if len(avail_sc) > 1:
    pairs = list(itertools.combinations(avail_sc, 2))
    p_vals, valid_pairs = [], []
    
    for m1, m2 in pairs:
        mask = df_master[m1].notna() & df_master[m2].notna()
        if mask.sum() > 3 and df_master.loc[mask, m1].nunique() > 1 and df_master.loc[mask, m2].nunique() > 1:
            r, p = spearmanr(df_master.loc[mask, m1], df_master.loc[mask, m2])
            p_vals.append(p)
            valid_pairs.append((m1, m2, r))
        else:
            p_vals.append(np.nan)
            valid_pairs.append((m1, m2, np.nan))
            
    p_vals_clean = [p for p in p_vals if not np.isnan(p)]
    if p_vals_clean:
        _, p_adj_clean, _, _ = multipletests(p_vals_clean, method='fdr_bh')
    
    p_adj = []
    idx = 0
    for p in p_vals:
        if np.isnan(p): p_adj.append(np.nan)
        else:
            p_adj.append(p_adj_clean[idx])
            idx += 1

    rows = math.ceil(len(pairs) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(16, 6 * rows))
    axes = axes.flatten()
    
    for i, (m1, m2) in enumerate(pairs):
        ax = axes[i]
        mask = df_master[m1].notna() & df_master[m2].notna()
        if mask.sum() > 3:
            sns.scatterplot(x=df_master.loc[mask, m1], y=df_master.loc[mask, m2], ax=ax, s=70, alpha=0.7)
            
            if df_master.loc[mask, m1].nunique() > 1:
                m_fit, b_fit = np.polyfit(df_master.loc[mask, m1], df_master.loc[mask, m2], 1)
                xs = np.linspace(df_master.loc[mask, m1].min(), df_master.loc[mask, m1].max(), 100)
                ax.plot(xs, m_fit*xs+b_fit, color='red', lw=2)
            
            r = valid_pairs[i][2]
            p = p_adj[i]
            star = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
            ax.set_title(f"R={r:.2f}{star} | p_adj={p:.2e}", fontsize=14, fontweight='bold')
        else:
            ax.set_title("Insufficient Data", fontsize=14)
            
        ax.set_xlabel(METRIC_NAMES.get(m1, m1), fontsize=12)
        ax.set_ylabel(METRIC_NAMES.get(m2, m2), fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

    for j in range(len(pairs), len(axes)): axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUT / 'scatter_correlation_grid.png', dpi=150)
    plt.close()

# --- 4. Highlighted Barplots ---
cols = 2
rows = math.ceil(len(avail_sc) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 8))
axes = axes.flatten()

for i, m in enumerate(avail_sc):
    ax = axes[i]
    sub = df_master[['gene_name', m]].dropna().sort_values(m, ascending=False).head(50)
    
    n_items = len(sub)
    if n_items == 0:
        ax.set_title(f"{METRIC_NAMES.get(m, m)} (No Data)", fontweight='bold')
        continue
        
    y_pos = np.arange(n_items, 0, -1)
    
    colors = ['#c0392b' if g in highlight_genes else '#7f8c8d' for g in sub['gene_name']]
    ax.barh(y_pos, sub[m], color=colors, height=0.8)
    
    # Append star for Top 3
    ax.set_yticks(y_pos)
    labels = sub['gene_name'].tolist()
    new_labels = [f"{g} ★" if g in top_3_genes else g for g in labels]
    ax.set_yticklabels(new_labels, fontsize=9)
    
    # Red bolding
    for tick_label, orig_gene in zip(ax.get_yticklabels(), labels):
        if orig_gene in highlight_genes:
            tick_label.set_color('#c0392b')
            tick_label.set_fontweight('bold')
    
    ax.set_ylim(0, 51)
    ax.set_title(METRIC_NAMES.get(m, m), fontweight='bold', fontsize=14)
    ax.tick_params(axis='x', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
for j in range(len(avail_sc), len(axes)): axes[j].axis('off')

# Custom Legend
patch_hl = mpatches.Patch(color='#c0392b', label='Highlighted (Top 20% in ≥3 target metrics)')
patch_star = mpatches.Patch(color='#c0392b', label='★ = Overall Top 3 Genes')
patch_no = mpatches.Patch(color='#7f8c8d', label='Other')
fig.legend(handles=[patch_hl, patch_star, patch_no], loc='lower right', fontsize=14)

plt.tight_layout()
plt.savefig(OUT / 'barplot_top50_per_metric.png', dpi=150)
plt.close()

print(f"Finished. All plots saved to '{OUT}'")

#
#"""
#writeup_plots_25.py
#===================
#Simplified plotting script for beta-lactamase gene prioritisation.
#Omits hub metrics due to Pfam-domain amalgamation logical flaw.
#Generates heatmaps, scatter grids, and barplots with clean English labels and tiered highlighting.
#"""
#
#import os
#import math
#import itertools
#import warnings
#from pathlib import Path
#from collections import defaultdict, Counter
#import numpy as np
#import pandas as pd
#import polars as pl
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#import seaborn as sns
#from scipy.stats import spearmanr
#from statsmodels.stats.multitest import multipletests
#from tqdm import tqdm
#
#warnings.filterwarnings('ignore')
#
## =============================================================================
## 0. PATHS, PARAMETERS & ENGLISH LABELS
## =============================================================================
#OUT = Path('writeup_plots_25')
#OUT.mkdir(exist_ok=True)
#MIN_OBS = 10
#
## Ranking direction: False = Descending (Highest value is 'best'/rank 1)
#RANK_ASCENDING = {
#    'overall_raw_entropy': False,
#    'overall_relative_entropy': False,
#    'mean_cluster_raw_entropy': False,
#    'median_cluster_raw_entropy': False,
#    'max_cluster_raw_entropy': False,
#    'mean_cluster_rel_entropy': False,
#    'median_cluster_rel_entropy': False,
#    'max_cluster_rel_entropy': False,
#    'dup_rate': False,          
#    'pool_rate': False,         
#    'pct_mobile': False,        
#    'in_data_prevalence': False,
#    'NCBI_Plasmid': False,
#    'NCBI_WGS': False,
#    'NCBI_Chromosome': False,
#    'simpson_diversity': False
#}
#
## English translations for all metrics
#METRIC_NAMES = {
#    'overall_raw_entropy': 'Overall Raw Entropy',
#    'overall_relative_entropy': 'Overall Relative Entropy',
#    'mean_cluster_raw_entropy': 'Mean Cluster Raw Entropy',
#    'median_cluster_raw_entropy': 'Median Cluster Raw Entropy',
#    'max_cluster_raw_entropy': 'Max Cluster Raw Entropy',
#    'mean_cluster_rel_entropy': 'Mean Cluster Relative Entropy',
#    'median_cluster_rel_entropy': 'Median Cluster Relative Entropy',
#    'max_cluster_rel_entropy': 'Max Cluster Relative Entropy',
#    'dup_rate': 'Duplication Rate',
#    'pool_rate': 'MGE-Association Rate',
#    'pct_mobile': 'Mobile Plasmid Association (%)',
#    'in_data_prevalence': 'In-Data Prevalence',
#    'NCBI_Plasmid': 'NCBI Plasmid Prevalence',
#    'NCBI_WGS': 'NCBI WGS Prevalence',
#    'NCBI_Chromosome': 'NCBI Chromosome Prevalence',
#    'simpson_diversity': 'Species Spread (Simpson Diversity)'
#}
#
#def context_entropy(contexts):
#    N = len(contexts)
#    if N <= 1: return np.nan, N
#    counts = Counter(contexts)
#    probs = np.array(list(counts.values())) / N
#    H = -np.sum(probs * np.log2(probs))
#    return round(H, 6), N
#
## =============================================================================
## 1. RECALCULATE OVERALL ENTROPY & DUPLICATION 
## =============================================================================
#print("Recalculating clean overall entropy and duplication rates...")
#data_dir = Path('plasmid_motif_network/intermediate')
#files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))
#df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
#
#ordered_df = df_merged.sort(['plasmid', 'start', 'ali_from']).select(['plasmid', 'target_name', 'query_name'])
#plasmid_to_domains = defaultdict(list)
#pid_to_positions = defaultdict(list)
#plasmid_index_counter = defaultdict(int)
#
#for row in ordered_df.iter_rows(named=True):
#    plasmid = row['plasmid']
#    dom = row['target_name']
#    pid = row['query_name']
#    
#    idx = plasmid_index_counter[plasmid]
#    plasmid_index_counter[plasmid] += 1
#    
#    plasmid_to_domains[plasmid].append(dom)
#    if pid:
#        pid_to_positions[pid].append((plasmid, idx, dom))
#
#total_plasmids = len(plasmid_to_domains)
#
#all_domain_raw_H = {}
#for plas, doms in tqdm(plasmid_to_domains.items(), desc='Background Entropy'):
#    n = len(doms)
#    if n <= 1: continue
#    for i, dom in enumerate(doms):
#        if dom not in all_domain_raw_H:
#            all_domain_raw_H[dom] = []
#        all_domain_raw_H[dom].append((doms[(i - 1) % n], doms[(i + 1) % n]))
#
#for dom, ctxs in all_domain_raw_H.items():
#    H, N = context_entropy(ctxs)
#    all_domain_raw_H[dom] = H if N >= MIN_OBS else np.nan
#
#test = pd.read_csv('amrfindermapped_beta_lactamases.csv')
#gene_to_pids = defaultdict(set)
#for _, row in test.iterrows():
#    if isinstance(row['gene_name'], str):
#        gene_to_pids[row['gene_name']].add(row['query_id'])
#
#overall_records = []
#for gene_name in tqdm(list(gene_to_pids.keys()), desc='Overall Gene Metrics'):
#    pids = gene_to_pids[gene_name]
#    contexts, plasmids_seen, pfam_domains_this_gene = [], set(), set()
#    dup_events, total_tokens = 0, 0
#    
#    for pid in pids:
#        for plasmid, idx, dom in pid_to_positions.get(pid, []):
#            pfam_domains_this_gene.add(dom)
#            entries = plasmid_to_domains[plasmid]
#            n = len(entries)
#            if n <= 1: continue
#            left, right = entries[(idx - 1) % n], entries[(idx + 1) % n]
#            contexts.append((left, right))
#            plasmids_seen.add(plasmid)
#            
#            if left in pfam_domains_this_gene or 'lactamase' in left.lower(): dup_events += 1
#            if right in pfam_domains_this_gene or 'lactamase' in right.lower(): dup_events += 1
#            total_tokens += 1
#            
#    dup_rate = dup_events / total_tokens if total_tokens > 0 else 0.0
#    H, N = context_entropy(contexts)
#    
#    bg_domains = {d for plas in plasmids_seen for d in plasmid_to_domains[plas] if d not in pfam_domains_this_gene}
#    bg_raw = [x for x in (all_domain_raw_H.get(d, np.nan) for d in bg_domains) if not np.isnan(x)]
#    
#    bg_mean = np.mean(bg_raw) if len(bg_raw) >= 3 else np.nan
#    rel_H = round(H / bg_mean, 6) if not np.isnan(H) and not np.isnan(bg_mean) and bg_mean > 0 else np.nan
#    
#    overall_records.append({
#        'gene_name': gene_name,
#        'overall_raw_entropy': H if N >= MIN_OBS else np.nan,
#        'overall_relative_entropy': rel_H,
#        'dup_rate': dup_rate,
#        'in_data_prevalence': len(plasmids_seen) / total_plasmids if total_plasmids else 0
#    })
#
#df_master = pd.DataFrame(overall_records)
#
## =============================================================================
## 2. LOAD EXTERNAL METRICS & DEDUPLICATE
## =============================================================================
#print("Merging metadata and cluster metrics...")
#
#if Path('clustering_results/per_gene_bl_cluster_metrics.csv').exists():
#    df_jac = pd.read_csv('clustering_results/per_gene_bl_cluster_metrics.csv')
#    df_jac = df_jac.loc[:, ~df_jac.columns.duplicated()]
#    if 'gene_name' in df_jac.columns:
#        df_jac = df_jac.drop_duplicates(subset=['gene_name'])
#        df_master = df_master.merge(df_jac, on='gene_name', how='left')
#
#if Path('mge_association_results/per_gene_mge_association.csv').exists():
#    df_mge = pd.read_csv('mge_association_results/per_gene_mge_association.csv')
#    df_mge = df_mge.loc[:, ~df_mge.columns.duplicated()]
#    if 'gene_name' not in df_mge.columns and 'label' in df_mge.columns: 
#        df_mge.rename(columns={'label': 'gene_name'}, inplace=True)
#    if 'gene_name' in df_mge.columns and 'pool_rate' in df_mge.columns:
#        df_mge = df_mge.drop_duplicates(subset=['gene_name'])
#        df_master = df_master.merge(df_mge[['gene_name', 'pool_rate']], on='gene_name', how='left')
#
#if Path('beta_lactamase_mobility_stats_final.csv').exists():
#    df_mob = pd.read_csv('beta_lactamase_mobility_stats_final.csv')
#    df_mob = df_mob.loc[:, ~df_mob.columns.duplicated()]
#    if 'gene_name' not in df_mob.columns and 'gene' in df_mob.columns: 
#        df_mob.rename(columns={'gene': 'gene_name'}, inplace=True)
#    df_mob['pct_mobile'] = df_mob.get('pct_conjugative', 0) + df_mob.get('pct_mobilizable', 0)
#    if 'gene_name' in df_mob.columns and 'pct_mobile' in df_mob.columns:
#        df_mob = df_mob.drop_duplicates(subset=['gene_name'])
#        df_master = df_master.merge(df_mob[['gene_name', 'pct_mobile']], on='gene_name', how='left')
#
#if Path('beta_lactamase_species_breadth_final.csv').exists():
#    df_spc = pd.read_csv('beta_lactamase_species_breadth_final.csv')
#    df_spc = df_spc.loc[:, ~df_spc.columns.duplicated()]
#    if 'gene_name' not in df_spc.columns:
#        if 'label' in df_spc.columns: df_spc.rename(columns={'label': 'gene_name'}, inplace=True)
#        elif 'gene' in df_spc.columns: df_spc.rename(columns={'gene': 'gene_name'}, inplace=True)
#    if 'gene_name' in df_spc.columns and 'simpson_diversity' in df_spc.columns:
#        df_spc = df_spc.drop_duplicates(subset=['gene_name'])
#        df_master = df_master.merge(df_spc[['gene_name', 'simpson_diversity']], on='gene_name', how='left')
#
#if Path('card_prevalence.txt').exists():
#    df_card = pd.read_csv('card_prevalence.txt', sep='\t')
#    df_card = df_card.loc[:, ~df_card.columns.duplicated()]
#    if 'gene_name' not in df_card.columns and 'Name' in df_card.columns:
#        df_card.rename(columns={'Name': 'gene_name'}, inplace=True)
#    df_card.rename(columns={'NCBI Plasmid': 'NCBI_Plasmid', 'NCBI WGS': 'NCBI_WGS', 'NCBI Chromosome': 'NCBI_Chromosome'}, inplace=True)
#    card_cols = ['gene_name', 'NCBI_Plasmid', 'NCBI_WGS', 'NCBI_Chromosome']
#    avail_card_cols = [c for c in card_cols if c in df_card.columns]
#    if 'gene_name' in avail_card_cols:
#        df_card = df_card.drop_duplicates(subset=['gene_name'])
#        df_master = df_master.merge(df_card[avail_card_cols], on='gene_name', how='left')
#
#df_master = df_master.drop_duplicates(subset=['gene_name']).reset_index(drop=True)
#df_master.to_csv(OUT / 'master_metrics_simplified.csv', index=False)
#
#
## =============================================================================
## 3. PREPARE HIGHLIGHT GENES
## =============================================================================
#heatmap_metrics = [
#    'overall_raw_entropy', 
#    'pct_mobile', 
#    'in_data_prevalence', 
#    'NCBI_WGS', 
#    'simpson_diversity'
#]
#avail_heatmap_metrics = [m for m in heatmap_metrics if m in df_master.columns]
#
#highlight_counts = Counter()
#for m in avail_heatmap_metrics:
#    asc = RANK_ASCENDING.get(m, False)
#    sub = df_master[['gene_name', m]].fillna(0)
#    if not sub.empty:
#        n_top = max(1, len(sub) // 5) # Top 20%
#        for g in sub.sort_values(m, ascending=asc).head(n_top)['gene_name']:
#            highlight_counts[g] += 1
#
## Tier 1: Top 20% in at least 3 metrics
#highlight_genes = {g for g, c in highlight_counts.items() if c >= 3}
#
## Tier 2: The absolute top 3 genes overall (calculated via mean_rank on the selected metrics)
#rank_df = df_master[['gene_name'] + avail_heatmap_metrics].dropna(how='all', subset=avail_heatmap_metrics).copy()
#for m in avail_heatmap_metrics:
#    asc = RANK_ASCENDING.get(m, False)
#    rank_df[m + '_rank'] = rank_df[m].rank(ascending=asc, na_option='bottom')
#rank_df['mean_rank'] = rank_df[[m + '_rank' for m in avail_heatmap_metrics]].mean(axis=1)
#top_3_genes = set(rank_df.sort_values('mean_rank').head(3)['gene_name'])
#
#
## =============================================================================
## 4. PLOTTING
## =============================================================================
#print("Generating Plots...")
#
## --- 1. Main Heatmap ---
#sub_hm = df_master[['gene_name'] + avail_heatmap_metrics].copy().dropna(how='all', subset=avail_heatmap_metrics)
#if not sub_hm.empty:
#    for m in avail_heatmap_metrics:
#        asc = RANK_ASCENDING.get(m, False)
#        sub_hm[m + '_rank'] = sub_hm[m].rank(ascending=asc, na_option='bottom')
#    
#    sub_hm['mean_rank'] = sub_hm[[m + '_rank' for m in avail_heatmap_metrics]].mean(axis=1)
#    top_hm = sub_hm.sort_values('mean_rank').head(50).set_index('gene_name')[avail_heatmap_metrics]
#    
#    top_hm = top_hm.rename(columns=METRIC_NAMES)
#    plot_data = (top_hm - top_hm.mean()) / top_hm.std() 
#    
#    plt.figure(figsize=(10, 12))
#    ax = sns.heatmap(plot_data, cmap='viridis', cbar_kws={'label': 'Z-score (Ranked)'})
#    plt.title('Top 50 β-Lactamase Genes\n(★ = Overall Top 3 Genes)', fontsize=14, pad=15)
#    
#    # Update labels with stars and red color
#    labels = [t.get_text() for t in ax.get_yticklabels()]
#    new_labels = [f"{g} ★" if g in top_3_genes else g for g in labels]
#    ax.set_yticklabels(new_labels)
#    
#    for tick_label, orig_gene in zip(ax.get_yticklabels(), labels):
#        if orig_gene in highlight_genes:
#            tick_label.set_color('#c0392b')
#            tick_label.set_fontweight('bold')
#            
#    plt.tight_layout()
#    plt.savefig(OUT / 'heatmap_1_selected_metrics.png', dpi=150)
#    plt.close()
#
## --- 2. Spearman Heatmap (Overall only) ---
#avail_sp = [m for m in ['overall_raw_entropy', 'overall_relative_entropy', 'dup_rate', 'pool_rate'] if m in df_master.columns]
#if avail_sp:
#    corr = df_master[avail_sp].corr(method='spearman')
#    corr.columns = [METRIC_NAMES.get(c, c) for c in corr.columns]
#    corr.index = [METRIC_NAMES.get(c, c) for c in corr.index]
#    
#    mask = np.triu(np.ones_like(corr, dtype=bool))
#    plt.figure(figsize=(10, 8))
#    sns.heatmap(corr, annot=True, fmt='.2f', mask=mask, cmap='coolwarm', center=0, vmin=-1, vmax=1)
#    plt.title('Spearman Correlation (Overall Data)', pad=20)
#    plt.tight_layout()
#    plt.savefig(OUT / 'spearman_A_overall.png', dpi=150)
#    plt.close()
#
## --- 3. Scatter Correlation Plots ---
#all_scatter_metrics = [
#    'overall_raw_entropy', 'overall_relative_entropy', 
#    'mean_cluster_raw_entropy', 'mean_cluster_rel_entropy',
#    'dup_rate', 'pool_rate', 'pct_mobile', 'in_data_prevalence', 
#    'NCBI_Plasmid', 'NCBI_WGS', 'NCBI_Chromosome', 'simpson_diversity'
#]
#avail_sc = [m for m in all_scatter_metrics if m in df_master.columns]
#
#if len(avail_sc) > 1:
#    pairs = list(itertools.combinations(avail_sc, 2))
#    p_vals, valid_pairs = [], []
#    
#    for m1, m2 in pairs:
#        mask = df_master[m1].notna() & df_master[m2].notna()
#        if mask.sum() > 3 and df_master.loc[mask, m1].nunique() > 1 and df_master.loc[mask, m2].nunique() > 1:
#            r, p = spearmanr(df_master.loc[mask, m1], df_master.loc[mask, m2])
#            p_vals.append(p)
#            valid_pairs.append((m1, m2, r))
#        else:
#            p_vals.append(np.nan)
#            valid_pairs.append((m1, m2, np.nan))
#            
#    p_vals_clean = [p for p in p_vals if not np.isnan(p)]
#    if p_vals_clean:
#        _, p_adj_clean, _, _ = multipletests(p_vals_clean, method='fdr_bh')
#    
#    p_adj = []
#    idx = 0
#    for p in p_vals:
#        if np.isnan(p): p_adj.append(np.nan)
#        else:
#            p_adj.append(p_adj_clean[idx])
#            idx += 1
#
#    rows = math.ceil(len(pairs) / 2)
#    fig, axes = plt.subplots(rows, 2, figsize=(16, 6 * rows))
#    axes = axes.flatten()
#    
#    for i, (m1, m2) in enumerate(pairs):
#        ax = axes[i]
#        mask = df_master[m1].notna() & df_master[m2].notna()
#        if mask.sum() > 3:
#            sns.scatterplot(x=df_master.loc[mask, m1], y=df_master.loc[mask, m2], ax=ax, s=70, alpha=0.7)
#            
#            if df_master.loc[mask, m1].nunique() > 1:
#                m_fit, b_fit = np.polyfit(df_master.loc[mask, m1], df_master.loc[mask, m2], 1)
#                xs = np.linspace(df_master.loc[mask, m1].min(), df_master.loc[mask, m1].max(), 100)
#                ax.plot(xs, m_fit*xs+b_fit, color='red', lw=2)
#            
#            r = valid_pairs[i][2]
#            p = p_adj[i]
#            star = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
#            ax.set_title(f"R={r:.2f}{star} | p_adj={p:.2e}", fontsize=14, fontweight='bold')
#        else:
#            ax.set_title("Insufficient Data", fontsize=14)
#            
#        ax.set_xlabel(METRIC_NAMES.get(m1, m1), fontsize=12)
#        ax.set_ylabel(METRIC_NAMES.get(m2, m2), fontsize=12)
#        ax.grid(True, linestyle='--', alpha=0.5)
#
#    for j in range(len(pairs), len(axes)): axes[j].axis('off')
#    
#    plt.tight_layout()
#    plt.savefig(OUT / 'scatter_correlation_grid.png', dpi=150)
#    plt.close()
#
## --- 4. Highlighted Barplots ---
#cols = 2
#rows = math.ceil(len(avail_sc) / cols)
#fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 8))
#axes = axes.flatten()
#
#for i, m in enumerate(avail_sc):
#    ax = axes[i]
#    sub = df_master[['gene_name', m]].dropna().sort_values(m, ascending=False).head(50)
#    
#    n_items = len(sub)
#    if n_items == 0:
#        ax.set_title(f"{METRIC_NAMES.get(m, m)} (No Data)", fontweight='bold')
#        continue
#        
#    y_pos = np.arange(n_items, 0, -1)
#    
#    colors = ['#c0392b' if g in highlight_genes else '#7f8c8d' for g in sub['gene_name']]
#    ax.barh(y_pos, sub[m], color=colors, height=0.8)
#    
#    # Append star for Top 3
#    ax.set_yticks(y_pos)
#    labels = sub['gene_name'].tolist()
#    new_labels = [f"{g} ★" if g in top_3_genes else g for g in labels]
#    ax.set_yticklabels(new_labels, fontsize=9)
#    
#    # Red bolding
#    for tick_label, orig_gene in zip(ax.get_yticklabels(), labels):
#        if orig_gene in highlight_genes:
#            tick_label.set_color('#c0392b')
#            tick_label.set_fontweight('bold')
#    
#    ax.set_ylim(0, 51)
#    ax.set_title(METRIC_NAMES.get(m, m), fontweight='bold', fontsize=14)
#    ax.tick_params(axis='x', labelsize=10)
#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    
#for j in range(len(avail_sc), len(axes)): axes[j].axis('off')
#
## Custom Legend
#patch_hl = mpatches.Patch(color='#c0392b', label='Highlighted (Top 20% in ≥3 target metrics)')
#patch_star = mpatches.Patch(color='#c0392b', label='★ = Overall Top 3 Genes')
#patch_no = mpatches.Patch(color='#7f8c8d', label='Other')
#fig.legend(handles=[patch_hl, patch_star, patch_no], loc='lower right', fontsize=14)
#
#plt.tight_layout()
#plt.savefig(OUT / 'barplot_top50_per_metric.png', dpi=150)
#plt.close()
#
#print(f"Finished. All plots saved to '{OUT}'")
