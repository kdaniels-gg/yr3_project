
# %% [1] LOAD AND BUILD CIRCULAR PLASMIDS
import polars as pl
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

MIN_OBS = 10

print("Loading parquet files...")
data_dir = Path('plasmid_motif_network/intermediate')
files = sorted(data_dir.glob('parsed_selected_nonoverlap_*.parquet'))
df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()

ordered_df = df_merged.sort(['plasmid', 'start', 'ali_from']).select(['plasmid', 'target_name'])

print("Building circular domain lists...")
plasmid_to_domains = defaultdict(list)
for row in ordered_df.iter_rows(named=True):
    plasmid_to_domains[row['plasmid']].append(row['target_name'])

domain_positions = defaultdict(list)
for plasmid, doms in plasmid_to_domains.items():
    n = len(doms)
    if n <= 1: continue 
    for i, dom in enumerate(doms):
        domain_positions[dom].append((plasmid, i))

print(f"Loaded {len(domain_positions)} unique Pfam domains across {len(plasmid_to_domains)} plasmids.")

def context_entropy_score(contexts):
    """
    Returns:
    1. Penalized/Normalized score (old metric)
    2. Raw Shannon entropy (H)
    3. N (total copies)
    4. K (unique contexts)
    """
    N = len(contexts)
    if N <= 1:
        return np.nan, np.nan, N, 0
    counts = Counter(contexts)
    K = len(counts)
    if K <= 1:
        return 0.0, 0.0, N, K
    probs = np.array(list(counts.values())) / N
    
    # Raw shannon entropy
    H = -np.sum(probs * np.log2(probs))
    
    # Old normalized score
    H_context = H / np.log2(K)
    score = H_context * (K / N)
    
    return round(score, 6), round(H, 6), N, K

def get_contexts_and_dup_rate(plasmid_idx_list, dom):
    neighbour_tokens = []
    plasmids_seen = set()
    duplication_events = 0
    for plasmid, idx in plasmid_idx_list:
        entries = plasmid_to_domains[plasmid]
        n = len(entries)
        if n <= 1:
            continue
        left = entries[(idx - 1) % n]
        right = entries[(idx + 1) % n]
        neighbour_tokens.append((left, right))
        plasmids_seen.add(plasmid)
        #detect tandem duplication
        if left == dom:
            duplication_events += 1
        if right == dom:
            duplication_events += 1
    duplication_rate = duplication_events / len(neighbour_tokens) if neighbour_tokens else 0.0
    return neighbour_tokens, plasmids_seen, duplication_events, duplication_rate

print("Calculating entropy metrics...")

domain_data = {}
domain_scores = {}
domain_raw_scores = {} # New dictionary for background tracking of raw entropy

for dom, pos_list in domain_positions.items():
    neigh_tokens, plas_seen, dup_events, dup_rate = \
        get_contexts_and_dup_rate(pos_list, dom)
    neigh_score, H_neigh, n_copies, n_unique = \
        context_entropy_score(neigh_tokens)
    
    domain_data[dom] = {
        "plas_seen": plas_seen,
        "context_entropy": neigh_score,
        "raw_entropy": H_neigh, # Tracking raw H explicitly
        "n_copies": n_copies,
        "n_unique_contexts": n_unique,
        "dup_events": dup_events,
        "duplication_rate": dup_rate
    }
    
    if n_copies >= MIN_OBS:
        if not np.isnan(neigh_score):
            domain_scores[dom] = neigh_score
        if not np.isnan(H_neigh):
            domain_raw_scores[dom] = H_neigh

# %% [5] COMPUTE RELATIVE ENTROPY METRICS
print("Computing relative scores...")
records = []
for dom, data in domain_data.items():
    if data["n_copies"] < MIN_OBS:
        continue
    bg_domains = set()
    for plas in data["plas_seen"]:
        for d in plasmid_to_domains.get(plas, []):
            if d != dom:
                bg_domains.add(d)
                
    # Background for old normalized metric
    bg_scores = [domain_scores[d] for d in bg_domains if d in domain_scores]
    rel_entropy = (
        round(data["context_entropy"] - np.mean(bg_scores), 6)
        if len(bg_scores) >= 3 else np.nan
    )
    
    # Background for new raw metric
    bg_raw_scores = [domain_raw_scores[d] for d in bg_domains if d in domain_raw_scores]
    rel_raw_entropy = (
        round(data["raw_entropy"] - np.mean(bg_raw_scores), 6)
        if len(bg_raw_scores) >= 3 else np.nan
    )

    records.append({
        "target_name": dom,
        "n_copies": data["n_copies"],
        "n_unique_contexts": data["n_unique_contexts"],
        "raw_entropy": data["raw_entropy"],
        "relative_raw_entropy": rel_raw_entropy,
        "context_entropy": data["context_entropy"],
        "relative_context_entropy": rel_entropy,
        "duplication_events": data["dup_events"],
        "duplication_rate": data["duplication_rate"]
    })

results_df = pd.DataFrame(records)
# Compute combined scores
results_df['combined_score'] = results_df['relative_context_entropy'] + results_df['duplication_rate']
results_df['combined_raw_score'] = results_df['relative_raw_entropy'] + results_df['duplication_rate']

out_dir = Path('pfam_entropy_distribution')
os.makedirs(out_dir, exist_ok=True)

results_df.to_csv(out_dir / 'pfam_entropy_scatter_data.csv', index=False)
print("Calculation complete! Saved to pfam_entropy_scatter_data.csv")

# ==========================================
# PLOTTING
# ==========================================
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})
sns.set_style("ticks")

# --- PLOT 1: HISTOGRAMS (1x5 Grid) ---
fig, axes = plt.subplots(1, 5, figsize=(24, 4))

sns.histplot(data=results_df, x='context_entropy', bins=50, ax=axes[0], color='#4C72B0', edgecolor='black', alpha=0.8)
axes[0].set_xlabel('Context Entropy (Old)')
axes[0].set_ylabel('Frequency')

sns.histplot(data=results_df, x='relative_context_entropy', bins=50, ax=axes[1], color='#55A868', edgecolor='black', alpha=0.8)
axes[1].set_xlabel('Rel. Context Entropy')
axes[1].set_ylabel('')

sns.histplot(data=results_df, x='raw_entropy', bins=50, ax=axes[2], color='#8172B3', edgecolor='black', alpha=0.8)
axes[2].set_xlabel('Raw Entropy')
axes[2].set_ylabel('')

sns.histplot(data=results_df, x='relative_raw_entropy', bins=50, ax=axes[3], color='#937860', edgecolor='black', alpha=0.8)
axes[3].set_xlabel('Rel. Raw Entropy')
axes[3].set_ylabel('')

sns.histplot(data=results_df, x='combined_raw_score', bins=50, ax=axes[4], color='#C44E52', edgecolor='black', alpha=0.8)
axes[4].set_xlabel('Rel. Raw + Dup. Rate')
axes[4].set_ylabel('')

sns.despine()
plt.tight_layout()
plt.savefig(out_dir / 'entropy_metric_histograms.pdf', bbox_inches='tight')
plt.close()

# --- PLOT 2: TOP 50 BARPLOTS (1x5 Grid, Independent Sorts) ---
top50_ce = results_df.sort_values('context_entropy', ascending=False).head(50)
top50_rce = results_df.sort_values('relative_context_entropy', ascending=False).head(50)
top50_raw = results_df.sort_values('raw_entropy', ascending=False).head(50)
top50_rre = results_df.sort_values('relative_raw_entropy', ascending=False).head(50)
top50_comb_raw = results_df.sort_values('combined_raw_score', ascending=False).head(50)

fig, axes = plt.subplots(1, 5, figsize=(28, 12))

sns.barplot(data=top50_ce, y='target_name', x='context_entropy', ax=axes[0], color='#4C72B0')
axes[0].set_xlabel('Context Entropy (Old)')
axes[0].set_ylabel('')
axes[0].set_title('Top 50: Context Entropy')

sns.barplot(data=top50_rce, y='target_name', x='relative_context_entropy', ax=axes[1], color='#55A868')
axes[1].set_xlabel('Relative Context Entropy')
axes[1].set_ylabel('')
axes[1].set_title('Top 50: Rel. Context Entropy')

sns.barplot(data=top50_raw, y='target_name', x='raw_entropy', ax=axes[2], color='#8172B3')
axes[2].set_xlabel('Raw Entropy')
axes[2].set_ylabel('')
axes[2].set_title('Top 50: Raw Entropy')

sns.barplot(data=top50_rre, y='target_name', x='relative_raw_entropy', ax=axes[3], color='#937860')
axes[3].set_xlabel('Rel. Raw Entropy')
axes[3].set_ylabel('')
axes[3].set_title('Top 50: Rel. Raw Entropy')

sns.barplot(data=top50_comb_raw, y='target_name', x='combined_raw_score', ax=axes[4], color='#C44E52')
axes[4].set_xlabel('Rel. Raw + Dup Rate')
axes[4].set_ylabel('')
axes[4].set_title('Top 50: Combined Raw Score')

for ax in axes:
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

sns.despine(left=True)
plt.tight_layout()
plt.savefig(out_dir / 'top50_metrics_independent_barplot.pdf', bbox_inches='tight')
plt.close()




## %% [1] LOAD AND BUILD CIRCULAR PLASMIDS
#import polars as pl
#import pandas as pd
#import numpy as np
#from collections import defaultdict, Counter
#import matplotlib.pyplot as plt
#import seaborn as sns
#from pathlib import Path
#import os
#import matplotlib.pyplot as plt
#import seaborn as sns
#
#MIN_OBS = 10
#
#print("Loading parquet files...")
#data_dir = Path('plasmid_motif_network/intermediate')
#files = sorted(data_dir.glob('parsed_selected_nonoverlap_*.parquet'))
#df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
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
#    if n <= 1: continue 
#    for i, dom in enumerate(doms):
#        domain_positions[dom].append((plasmid, i))
#
#print(f"Loaded {len(domain_positions)} unique Pfam domains across {len(plasmid_to_domains)} plasmids.")
#
#
#
#
#def context_entropy_score(contexts):
#    """
#    Normalisation -
#    1) divide by log2(K) to remove context count ceiling
#    2) weight by by K/N to account for realised contexts vs opportunities as per copy number
#    """
#    N = len(contexts)
#    if N <= 1:
#        return np.nan, np.nan, N, 0
#    counts = Counter(contexts)
#    K = len(counts)
#    if K <= 1:
#        return 0.0, 0.0, N, K
#    probs = np.array(list(counts.values())) / N
#    #shannon entropy
#    H = -np.sum(probs * np.log2(probs))
#    H_context = H / np.log2(K)
#    score = H_context * (K / N)
#    return round(score, 6), round(H, 6), N, K
#
#
#def get_contexts_and_dup_rate(plasmid_idx_list, dom):
#    neighbour_tokens = []
#    plasmids_seen = set()
#    duplication_events = 0
#    for plasmid, idx in plasmid_idx_list:
#        entries = plasmid_to_domains[plasmid]
#        n = len(entries)
#        if n <= 1:
#            continue
#        left = entries[(idx - 1) % n]
#        right = entries[(idx + 1) % n]
#        neighbour_tokens.append((left, right))
#        plasmids_seen.add(plasmid)
#        #detect tandem duplication
#        if left == dom:
#            duplication_events += 1
#        if right == dom:
#            duplication_events += 1
#    duplication_rate = duplication_events / len(neighbour_tokens)
#    return neighbour_tokens, plasmids_seen, duplication_events, duplication_rate
#
#
#print("Calculating entropy metrics...")
#
#domain_data = {}
#domain_scores = {}
#
#for dom, pos_list in domain_positions.items():
#    neigh_tokens, plas_seen, dup_events, dup_rate = \
#        get_contexts_and_dup_rate(pos_list, dom)
#    neigh_score, H_neigh, n_copies, n_unique = \
#        context_entropy_score(neigh_tokens)
#    domain_data[dom] = {
#        "plas_seen": plas_seen,
#        "context_entropy": neigh_score,
#        "H_neigh": H_neigh,
#        "n_copies": n_copies,
#        "n_unique_contexts": n_unique,
#        "dup_events": dup_events,
#        "duplication_rate": dup_rate
#    }
#    if n_copies >= MIN_OBS and not np.isnan(neigh_score):
#        domain_scores[dom] = neigh_score
#
#
## %% [5] COMPUTE RELATIVE CONTEXT ENTROPY
#
#print("Computing relative scores...")
#records = []
#for dom, data in domain_data.items():
#    if data["n_copies"] < MIN_OBS:
#        continue
#    bg_domains = set()
#    for plas in data["plas_seen"]:
#        for d in plasmid_to_domains.get(plas, []):
#            if d != dom:
#                bg_domains.add(d)
#    bg_scores = [
#        domain_scores[d]
#        for d in bg_domains
#        if d in domain_scores
#    ]
#    rel_entropy = (
#        round(data["context_entropy"] - np.mean(bg_scores), 6)
#        if len(bg_scores) >= 3 else np.nan
#    )
#    records.append({
#        "target_name": dom,
#        "n_copies": data["n_copies"],
#        "n_unique_contexts": data["n_unique_contexts"],
#        "H_neigh": data["H_neigh"],
#        "context_entropy": data["context_entropy"],
#        "relative_context_entropy": rel_entropy,
#        "duplication_events": data["dup_events"],
#        "duplication_rate": data["duplication_rate"]
#    })
#
#
#
#
#results_df = pd.DataFrame(records)
#results_df['combined_score'] = results_df['relative_context_entropy'] + results_df['duplication_rate']
#out_dir = Path('pfam_entropy_distribution')
#os.makedirs(out_dir, exist_ok=True)
#
#results_df.to_csv(out_dir / 'pfam_entropy_scatter_data.csv', index=False)
#
#print("Calculation complete!")
#print("Saved to pfam_entropy_scatter_data.csv")
#
#
#
#
#
#plt.rcParams.update({
#    'font.size': 12,
#    'font.family': 'sans-serif',
#    'axes.titlesize': 14,
#    'axes.labelsize': 12,
#    'xtick.labelsize': 10,
#    'ytick.labelsize': 10,
#})
#sns.set_style("ticks")
#
#
#fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#
#sns.histplot(data=results_df, x='context_entropy', bins=50, ax=axes[0], 
#             color='#4C72B0', edgecolor='black', alpha=0.8)
#axes[0].set_xlabel('Context Entropy')
#axes[0].set_ylabel('Frequency')
#
#
#sns.histplot(data=results_df, x='relative_context_entropy', bins=50, ax=axes[1], 
#             color='#55A868', edgecolor='black', alpha=0.8)
#axes[1].set_xlabel('Relative Context Entropy')
#axes[1].set_ylabel('')
#
#
#sns.histplot(data=results_df, x='combined_score', bins=50, ax=axes[2], 
#             color='#C44E52', edgecolor='black', alpha=0.8)
#axes[2].set_xlabel('Rel. Context Entropy + Dup. Rate')
#axes[2].set_ylabel('')
#
#
#sns.despine()
#plt.tight_layout()
#plt.savefig(out_dir / 'entropy_metric_histograms.pdf', bbox_inches='tight')
#plt.close()
#
#
#
#
#
#
#
#
#top_50_df = results_df.sort_values('combined_score', ascending=False).head(50)
#fig, axes = plt.subplots(1, 3, figsize=(12, 12), sharey=True)
#
#sns.barplot(data=top_50_df, y='target_name', x='context_entropy', ax=axes[0], color='#4C72B0')
#axes[0].set_xlabel('Context Entropy')
#axes[0].set_ylabel('')
#
#sns.barplot(data=top_50_df, y='target_name', x='relative_context_entropy', ax=axes[1], color='#55A868')
#axes[1].set_xlabel('Relative Context Entropy')
#axes[1].set_ylabel('')
#
#sns.barplot(data=top_50_df, y='target_name', x='combined_score', ax=axes[2], color='#C44E52')
#axes[2].set_xlabel('Rel. Context Entropy + Dup. Rate')
#axes[2].set_ylabel('')
#
#for ax in axes:
#    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
#    ax.set_axisbelow(True)
#
#
#sns.despine(left=True)
#plt.tight_layout()
#plt.savefig("top50_metrics_barplot.pdf", bbox_inches='tight')
#plt.show()
#









































# %% [4] PLOT RESULTS: THE SCATTER BREAKAWAY
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

fig, ax = plt.subplots(figsize=(12, 8))

# 1. Plot the main cloud of domains
sns.scatterplot(
    data=results_df, x='n_copies', y='H_raw', 
    alpha=0.5, color='#2166ac', edgecolor=None, ax=ax
)

# 2. Identify and highlight the top 20 domains by absolute Raw Entropy
top_raw = results_df.sort_values('H_raw', ascending=False).head(20)
sns.scatterplot(
    data=top_raw, x='n_copies', y='H_raw', 
    color='#d6604d', s=80, edgecolor='black', zorder=5, ax=ax
)

# 3. Add text labels to the highlighted top domains
for _, row in top_raw.iterrows():
    ax.text(
        row['n_copies'] * 1.05,  # Offset slightly to the right
        row['H_raw'], 
        row['target_name'], 
        fontsize=9, color='black', weight='bold', va='center'
    )

# 4. Draw the Theoretical Maximum Line (H = log2(N))
x_vals = np.linspace(results_df['n_copies'].min(), results_df['n_copies'].max(), 500)
y_max = np.log2(x_vals)
ax.plot(
    x_vals, y_max, 
    color='gray', linestyle='--', linewidth=2, 
    label='Theoretical Max Entropy (Every context is unique)'
)

# Axis formatting
ax.set_xscale('log')
ax.set_title('Pfam Domain Mobility: Raw Entropy vs. Copy Number', fontsize=16, weight='bold')
ax.set_xlabel('Total Observations (Copy Number) [Log Scale]', fontsize=13)
ax.set_ylabel('Raw Contextual Entropy ($H_{raw}$)', fontsize=13)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('pfam_raw_entropy_scatter.png', dpi=300, bbox_inches='tight')
plt.show()