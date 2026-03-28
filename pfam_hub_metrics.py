
# HUB ANALYSIS: Identifying hub domains in plasmid domain-architecture networks
#    DEGREE CENTRALITY
#    total degree (in + out) normalised by (n-1) (total possible edges)
#    high degree nodes are hubs by definition for scale free
#    expect to see mainly MGEs,
#    has no awareness of wheteher it bridges communities or is within a single cluster etc.
#    BETWEENNESS CENTRALITY
#    fraction of all-pairs shortest paths in the graph that pass through
#    a given node. 
#    Identifies bottlenecks that would fragment graph if removed 
#    ie nodes that sit on path between otherwise distant parts of the graph
#    likewise expect to see MGEs, but also beta-lactamases if within these MGEs
#    betweenness is computed on the undirected LCC
#    O(VE) is a lot for big graphs - use exact Brandes if too big
#    EIGENVECTOR CENTRALITY
#    node score proportional to sum of its neighbours scores, 
#    compute iteratively until it converges (on adj matrix) - means being connected
#    to well connected nodes boosts score 
#    Expect to see TEs and genes in MGEs
#    computed on the undirected LCC (directed eigenvector centrality is
#    unreliable when the graph has many weakly-connected components)
#    CLOSENESS CENTRALITY
#    inv of mean shortest path distance from a node to all reachable
#    nodes in the LCC. 
#    ie high closeness means you can reach the rest of the graph quickly from that node
#    if a node is usually close to other domain types
#    but has lower betweenness that means its possibly on intermediate architecture
#    posiitions and thus 'bridges' stuff without being a bottleneck 
#    #expect to see scaffold domains and stuff
#    Could identify core scaffold domains of plasmid backbones.
#    normalised Wasserman-faust correction variant used for
#    subgraph size so scores comparable across nodes
#    HITS — Hubs & Authoritiess
#    computed on directed graph, nodes put into orthogonal role of hub or autohority scores
#    hub score = high if node points to many high-auth nodes
#    auth score = high if node pointed to by many high hub nodes
#    currently only acts as preceding so hubs precede important domains, auth domains are preceeded by important domains
#    would expect auth to be cargo and hub to be MGE, but assumes unidirectionality which is eh
#
#
# GRAPH CHOICE
# degree and HITS use full directed graph i.e., with strand order
# betweeness, eigenvec and closeness use undirected LCC cos;
# directed shortest paths are undefined for nodes with sparse connections so ends up 0ing a lot of values for nodes
# undirected provides full actual topology, direected used for HITS because otherwise you are just computing degree
# LCC coverage fraction  reported because it proxies how representative the subgraph is
#
#
# HUB DEFINITION
# define hub if node in top 5% of any of the metrics.
# HITS auth and hub reported separately
# A node is a HUB if it is in the top HUB_PERCENTILE currently 5% on any metric
#

import os
import math
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr


graph_dir        = Path(os.path.join(os.getcwd(), 'plasmid_batched_graphs'))
species_base_dir = Path(os.path.join(os.getcwd(), 'species_specific_plasmid_analysis', 'big_species'))
hospital_dir     = Path(os.path.join(os.getcwd(), 'hospital_analysis', 'graphml'))
out_dir          = Path(os.path.join(os.getcwd(), 'hub_analysis_results'))
os.makedirs(out_dir, exist_ok=True)

ECOLI_SPECIES_LABEL = 'Escherichia_coli'
HOSPITAL_LABEL      = 'hospital'
PFAM_CLANS_TSV      = 'Pfam-A.clans.tsv'

# Hub = top N% on degree, betweenness, eigenvector, OR closeness (union)
HUB_PERCENTILE = 95    # 95 → top 5%; change to 98 for top 2%
TOP_N_PRINT    = 30    # rows printed to console per dataset
TOP_N_PLOT     = 40    # bars in ranked hub barplot

print('Paths set. Output:', out_dir)


# Columns: 0=accession, 1=clan_acc, 2=clan_name, 3=domain_name, 4=description
# domain_dict maps domain_name → clan_name

domain_df = pd.read_csv(PFAM_CLANS_TSV, sep='\t', header=None)
domain_dict = dict(zip(domain_df[3].tolist(), domain_df[2].tolist()))
domain_dict_clean = {
    k: (None if isinstance(v, float) and math.isnan(v) else v)
    for k, v in domain_dict.items()
}

beta_lac_domains = {k for k, v in domain_dict_clean.items() if v == 'Beta-lactamase'}
print(f'Beta-lactamase Pfam domains ({len(beta_lac_domains)}): {sorted(beta_lac_domains)}')


TE_KEYWORDS = [
    'transpos', 'tnp', 'integrase', 'integron', 'resolvase',
    'recombinase', 'insertion', 'is200', 'is110', 'is3', 'is4', 'is630',
    'merr', 'hin', 'xerc', 'xerd', 'rci', 'tniq', 'tnib', 'tnic',
    'mobilisa', 'mobiliz', 'mob', 'relaxase', 'nickase',
]
REP_KEYWORDS = [
    'rep_', 'repa', 'repb', 'replication', 'replic', 'replicon',
    'plasmid_stab', 'par_', 'sop', 'segs', 'ccd', 'toxin', 'antitoxin',
    'vapb', 'vapa', 'mazf', 'maze', 'hig', 'phd',
]
EFFLUX_KEYWORDS = [
    'efflux', 'resistance', 'tetracycl', 'aminoglyc', 'chloramphenicol',
    'trimethoprim', 'sulfonamide', 'qnr', 'aac', 'aph', 'ant(',
    'mcr', 'dfr', 'cat_', 'tetm', 'tetr', 'teta',
]
CONJUGATION_KEYWORDS = [
    'tra_', 'trbp', 'trbc', 'trbe', 'trbf', 'trbg', 'trbh', 'trbi',
    'trbj', 'trbk', 'trbl', 'virb', 'vird', 'type_iv', 't4ss', 'conjugat',
    'mating', 'pilin', 'pilus',
]


def classify_domain(domain_name, beta_lac_set=beta_lac_domains):
    if domain_name in beta_lac_set:
        return 'Beta-lactamase'
    dn = domain_name.lower()
    if any(k in dn for k in TE_KEYWORDS):
        return 'Transposon/MGE'
    if any(k in dn for k in CONJUGATION_KEYWORDS):
        return 'Conjugation/T4SS'
    if any(k in dn for k in REP_KEYWORDS):
        return 'Replication/Stability'
    if any(k in dn for k in EFFLUX_KEYWORDS):
        return 'Resistance (non-BL)'
    return 'Other/Unknown'


CATEGORY_COLOURS = {
    'Beta-lactamase':       '#d62728',
    'Transposon/MGE':       '#ff7f0e',
    'Conjugation/T4SS':     '#1f77b4',
    'Replication/Stability':'#2ca02c',
    'Resistance (non-BL)':  '#9467bd',
    'Other/Unknown':        '#aec7e8',
}
CATEGORY_ORDER = list(CATEGORY_COLOURS.keys())
print('Domain classifier ready.')


# =============================================================================
# SECTION 4 — GRAPH LOADERS
# =============================================================================

def load_max_batch_overall(graph_dir):
    batch_files = sorted(
        graph_dir.glob('*_domain_architecture_signed_network.graphml'),
        key=lambda p: int(p.name.split('_')[0])
    )
    if not batch_files:
        raise FileNotFoundError(f'No overall graphml files in {graph_dir}')
    path = batch_files[-1]
    print(f'Overall: loading {path.name}')
    return nx.read_graphml(str(path)), path


def load_max_batch_species(species_base_dir, species_label):
    sp_dir  = species_base_dir / species_label
    pattern = f'batch_*_{species_label}_domain_architecture_signed_network.graphml'
    batch_files = sorted(
        sp_dir.glob(pattern), key=lambda p: int(p.name.split('_')[1])
    )
    if not batch_files:
        raise FileNotFoundError(f'No species graphml files in {sp_dir}')
    path = batch_files[-1]
    print(f'E. coli: loading {path.name}')
    return nx.read_graphml(str(path)), path


def load_max_batch_hospital(hospital_dir, hospital_label):
    pattern = f'batch_*_{hospital_label}_domain_architecture_signed_network.graphml'
    batch_files = sorted(
        hospital_dir.glob(pattern), key=lambda p: int(p.name.split('_')[1])
    )
    if not batch_files:
        raise FileNotFoundError(f'No hospital graphml files in {hospital_dir}')
    path = batch_files[-1]
    print(f'Hospital: loading {path.name}')
    return nx.read_graphml(str(path)), path


G_overall,  _ = load_max_batch_overall(graph_dir)
G_ecoli,    _ = load_max_batch_species(species_base_dir, ECOLI_SPECIES_LABEL)
G_hospital, _ = load_max_batch_hospital(hospital_dir, HOSPITAL_LABEL)

graphs = {
    'Overall':  G_overall,
    'E. coli':  G_ecoli,
    'Hospital': G_hospital,
}

for lbl, G in graphs.items():
    print(f'{lbl:10s}  n={G.number_of_nodes():>6,}  m={G.number_of_edges():>7,}')



def compute_centralities(G, label='', hub_pct=HUB_PERCENTILE):
    print(f'\n[{label}] {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges')

    #Undirected LCC
    U         = G.to_undirected()
    lcc_nodes = max(nx.connected_components(U), key=len)
    U_lcc     = U.subgraph(lcc_nodes).copy()
    lcc_frac  = len(lcc_nodes) / G.number_of_nodes()
    print(f'  Undirected LCC: {U_lcc.number_of_nodes():,} nodes ({lcc_frac:.1%} of total)')

    #Degree (directed, full graph)
    print('  [1/5] Degree ...')
    in_deg    = dict(G.in_degree())
    out_deg   = dict(G.out_degree())
    total_deg = {n: in_deg[n] + out_deg[n] for n in G.nodes()}
    deg_cent  = nx.degree_centrality(G)   # normalised by (n-1)

    #Betweenness (undirected LCC)
    print('  [2/5] Betweenness (Brandes on undirected LCC) ...')
    betw = nx.betweenness_centrality(U_lcc, normalized=True)

    #Eigenvector (undirected LCC, power iteration)
    print('  [3/5] Eigenvector ...')
    try:
        eig = nx.eigenvector_centrality(U_lcc, max_iter=1000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        print('    WARNING: eigenvector did not converge — using degree-normalised proxy')
        raw = {n: U_lcc.degree(n) for n in U_lcc.nodes()}
        mx  = max(raw.values()) or 1
        eig = {n: v / mx for n, v in raw.items()}

    #Closeness (undirected LCC, Wasserman-Faust normalised)
    print('  [4/5] Closeness ...')
    close = nx.closeness_centrality(U_lcc, wf_improved=True)

    #HITS (directed graph — preserves strand-order adjacency)
    print('  [5/5] HITS (directed graph) ...')
    try:
        hits_hub, hits_auth = nx.hits(G, max_iter=1000, tol=1e-6, normalized=True)
    except nx.PowerIterationFailedConvergence:
        print('    WARNING: HITS did not converge — scores set to 0')
        hits_hub  = {n: 0.0 for n in G.nodes()}
        hits_auth = {n: 0.0 for n in G.nodes()}
    rows = []
    for node in G.nodes():
        cat     = classify_domain(node)
        is_beta = (cat == 'Beta-lactamase')
        in_lcc  = node in lcc_nodes
        rows.append({
            'domain':               node,
            'category':             cat,
            'is_beta_lactamase':    is_beta,
            'in_lcc':               in_lcc,
            # Degree (directed)
            'total_degree':         total_deg.get(node, 0),
            'in_degree':            in_deg.get(node, 0),
            'out_degree':           out_deg.get(node, 0),
            'degree_centrality':    deg_cent.get(node, 0.0),
            # Betweenness (0 for nodes outside LCC)
            'betweenness':          betw.get(node, 0.0),
            # Eigenvector (0 for nodes outside LCC)
            'eigenvector':          eig.get(node, 0.0),
            # Closeness (0 for nodes outside LCC)
            'closeness':            close.get(node, 0.0),
            # HITS (directed, all nodes)
            'hits_hub_score':       hits_hub.get(node, 0.0),
            'hits_authority_score': hits_auth.get(node, 0.0),
        })
    df = pd.DataFrame(rows)
    metrics_for_hub = ['total_degree', 'betweenness', 'eigenvector', 'closeness']
    for m in metrics_for_hub:
        thresh = np.percentile(df[m], hub_pct)
        df[f'hub_by_{m}'] = df[m] >= thresh
    df['is_hub'] = df[[f'hub_by_{m}' for m in metrics_for_hub]].any(axis=1)
    #HITS directional roles reported separately
    hits_hub_thresh  = np.percentile(df['hits_hub_score'],       hub_pct)
    hits_auth_thresh = np.percentile(df['hits_authority_score'], hub_pct)
    df['is_hits_hub']       = df['hits_hub_score']       >= hits_hub_thresh
    df['is_hits_authority'] = df['hits_authority_score'] >= hits_auth_thresh
    n_hubs      = df['is_hub'].sum()
    n_beta_hubs = df[df['is_hub'] & df['is_beta_lactamase']].shape[0]
    print(f'  Hubs (top {100-hub_pct}%, any metric): {n_hubs}  |  beta-lac hubs: {n_beta_hubs}')
    print(f'  HITS hubs: {df["is_hits_hub"].sum()}  |  HITS authorities: {df["is_hits_authority"].sum()}')
    return df.sort_values('total_degree', ascending=False).reset_index(drop=True)


centrality_results = {}
for lbl, G in graphs.items():
    centrality_results[lbl] = compute_centralities(G, label=lbl)


for lbl, df in centrality_results.items():
    fname = out_dir / f'centrality_{lbl.replace(" ", "_").replace(".", "")}.csv'
    df.to_csv(fname, index=False)
    print(f'Saved {fname.name}')



PRINT_COLS = ['domain', 'category', 'is_beta_lactamase',
              'total_degree', 'betweenness', 'eigenvector',
              'closeness', 'hits_hub_score', 'hits_authority_score']

print('\n' + '='*80)
for lbl, df in centrality_results.items():
    hubs = df[df['is_hub']].head(TOP_N_PRINT)
    print(f'\n--- {lbl}: top {min(TOP_N_PRINT, len(hubs))} hubs ---')
    print(hubs[PRINT_COLS].to_string(index=False))
print('='*80)





print('\n' + '='*80)
for lbl, df in centrality_results.items():
    hubs = df[df['is_hub']].head(100)
    print(f'\n--- {lbl}: top {min(100, len(hubs))} hubs ---')
    print(hubs[PRINT_COLS].to_string(index=False))
print('='*80)



for lbl, df in centrality_results.items():
    top = df.head(TOP_N_PLOT).copy()
    top = top.sort_values('total_degree', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(7, TOP_N_PLOT * 0.3)),
                             gridspec_kw={'width_ratios': [3, 2]})
    fig.suptitle(f'{lbl} — top {TOP_N_PLOT} hub domains by total degree', fontsize=13)

    ax = axes[0]
    colours_bar = [CATEGORY_COLOURS[c] for c in top['category']]
    bars = ax.barh(range(len(top)), top['total_degree'],
                   color=colours_bar, edgecolor='none', alpha=0.88, height=0.72)

    for i, (_, row) in enumerate(top.iterrows()):
        if row['is_beta_lactamase']:
            bars[i].set_edgecolor('#d62728')
            bars[i].set_linewidth(2.5)
        if row['is_hub']:
            ax.text(top['total_degree'].max() * 0.01, i,
                    '★', va='center', fontsize=7, color='black', alpha=0.5)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['domain'], fontsize=7.5)
    ax.set_xlabel('Total degree (in + out)', fontsize=10)
    ax.set_title('Degree  (★ = hub by any metric; red outline = beta-lactamase)', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cat_patches = [mpatches.Patch(color=CATEGORY_COLOURS[c], label=c)
                   for c in CATEGORY_ORDER if c in top['category'].values]
    ax.legend(handles=cat_patches, fontsize=7.5, frameon=False, loc='lower right')

    # ── RIGHT: heatmap of all other metrics ─────────────────────────────────
    ax2 = axes[1]
    metrics_hm = ['betweenness', 'eigenvector', 'closeness',
                  'hits_hub_score', 'hits_authority_score']
    metric_labels_hm = ['Betweenness\n(bottleneck)',
                        'Eigenvector\n(rich neigh.)',
                        'Closeness\n(geometric)',
                        'HITS hub\n(loader)',
                        'HITS auth.\n(cargo)']

    hm_data = top[metrics_hm].copy()
    for col in metrics_hm:
        col_range = hm_data[col].max() - hm_data[col].min()
        hm_data[col] = (hm_data[col] - hm_data[col].min()) / col_range if col_range > 0 else 0.0

    im = ax2.imshow(hm_data.values, aspect='auto', cmap='YlOrRd',
                    vmin=0, vmax=1, origin='lower')
    ax2.set_xticks(range(len(metrics_hm)))
    ax2.set_xticklabels(metric_labels_hm, fontsize=8)
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels(top['domain'], fontsize=7.5)
    ax2.set_title('Other metrics (row-normalised 0–1)', fontsize=9)

    for i, (_, row) in enumerate(top.iterrows()):
        if row['is_beta_lactamase']:
            ax2.add_patch(plt.Rectangle(
                (-0.5, i - 0.5), len(metrics_hm), 1,
                fill=False, edgecolor='#d62728', linewidth=2, clip_on=False
            ))

    plt.colorbar(im, ax=ax2, shrink=0.4, label='Normalised score')
    plt.tight_layout()

    fname = out_dir / f'top_hubs_ranked_{lbl.replace(" ", "_").replace(".", "")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname.name}')



fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle(
    f'Hub domain counts by functional category\n'
    f'(top {100-HUB_PERCENTILE}% on degree, betweenness, eigenvector, or closeness)',
    fontsize=13, y=1.02
)

for ax, (lbl, df) in zip(axes, centrality_results.items()):
    hubs       = df[df['is_hub']]
    cat_counts = hubs['category'].value_counts().reindex(CATEGORY_ORDER, fill_value=0)

    bars = ax.bar(range(len(CATEGORY_ORDER)), cat_counts.values,
                  color=[CATEGORY_COLOURS[c] for c in CATEGORY_ORDER],
                  edgecolor='none', alpha=0.88, width=0.65)

    for bar, val in zip(bars, cat_counts.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(cat_counts.values) * 0.01,
                    str(int(val)), ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels([c.replace('/', '/\n') for c in CATEGORY_ORDER],
                       fontsize=8.5, rotation=30, ha='right')
    ax.set_ylabel('Number of hub domains', fontsize=10)
    ax.set_title(f'{lbl}\n(n={len(hubs)} hubs / {len(df)} domains)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

patches = [mpatches.Patch(color=CATEGORY_COLOURS[c], label=c) for c in CATEGORY_ORDER]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.08))
plt.tight_layout()
plt.savefig(out_dir / 'hub_counts_by_category.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved hub_counts_by_category.png')


#Separate from generic hub barplot because HITS captures upstream/downstream
#architectural role — not overall importance.
#Hub (loader) = domain precedes important domains; Authority (cargo) = domain
#is preceded by important domains.

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle(
    'HITS directional roles by functional category\n'
    '(Loader = consistently upstream; Cargo = consistently downstream)',
    fontsize=13, y=1.02
)

for ax, (lbl, df) in zip(axes, centrality_results.items()):
    hits_hubs  = df[df['is_hits_hub']]
    hits_auths = df[df['is_hits_authority']]
    hub_counts  = hits_hubs['category'].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    auth_counts = hits_auths['category'].value_counts().reindex(CATEGORY_ORDER, fill_value=0)

    x = np.arange(len(CATEGORY_ORDER))
    w = 0.38
    ax.bar(x - w/2, hub_counts.values, width=w,
           color=[CATEGORY_COLOURS[c] for c in CATEGORY_ORDER],
           edgecolor='none', alpha=0.9, label='HITS hub (loader)')
    ax.bar(x + w/2, auth_counts.values, width=w,
           color=[CATEGORY_COLOURS[c] for c in CATEGORY_ORDER],
           edgecolor='black', linewidth=0.8, alpha=0.45, label='HITS authority (cargo)')

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('/', '/\n') for c in CATEGORY_ORDER],
                       fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(lbl, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].legend(fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig(out_dir / 'hits_roles_by_category.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved hits_roles_by_category.png')



fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Beta-lactamase domains: degree × betweenness\n'
             '(dashed lines = top-5% thresholds)',
             fontsize=13, y=1.02)

for ax, (lbl, df) in zip(axes, centrality_results.items()):
    non_beta      = df[~df['is_beta_lactamase']]
    non_beta_hubs = non_beta[non_beta['is_hub']]
    beta          = df[df['is_beta_lactamase']]

    ax.scatter(non_beta['total_degree'],      non_beta['betweenness'],
               c='#aec7e8', s=7, alpha=0.35, linewidths=0, label='Other domains')
    ax.scatter(non_beta_hubs['total_degree'], non_beta_hubs['betweenness'],
               c='#ff7f0e', s=16, alpha=0.7,  linewidths=0, label='Hubs (non-BL)')

    if len(beta) > 0:
        ax.scatter(beta['total_degree'], beta['betweenness'],
                   c='#d62728', s=70, alpha=0.95, linewidths=0.6,
                   edgecolors='black', zorder=5, label='Beta-lactamase')
        for _, row in beta.iterrows():
            ax.annotate(row['domain'],
                        xy=(row['total_degree'], row['betweenness']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, color='#d62728',
                        arrowprops=dict(arrowstyle='-', color='#d62728', lw=0.5))

    ax.axvline(np.percentile(df['total_degree'], HUB_PERCENTILE),
               color='grey', lw=0.8, ls='--', alpha=0.6)
    ax.axhline(np.percentile(df['betweenness'],  HUB_PERCENTILE),
               color='grey', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('Total degree', fontsize=10)
    ax.set_ylabel('Betweenness centrality', fontsize=10)
    ax.set_title(lbl, fontsize=11)
    ax.set_xscale('log')
    ax.legend(fontsize=8, frameon=False, markerscale=1.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(out_dir / 'bl_scatter_degree_betweenness.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved bl_scatter_degree_betweenness.png')


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Beta-lactamase domains: eigenvector × closeness\n'
             '(rich neighbourhood × geometric centrality)',
             fontsize=13, y=1.02)

for ax, (lbl, df) in zip(axes, centrality_results.items()):
    non_beta      = df[~df['is_beta_lactamase'] & df['in_lcc']]
    non_beta_hubs = non_beta[non_beta['is_hub']]
    beta          = df[df['is_beta_lactamase'] & df['in_lcc']]

    ax.scatter(non_beta['eigenvector'],      non_beta['closeness'],
               c='#aec7e8', s=7, alpha=0.35, linewidths=0, label='Other (LCC)')
    ax.scatter(non_beta_hubs['eigenvector'], non_beta_hubs['closeness'],
               c='#ff7f0e', s=16, alpha=0.7,  linewidths=0, label='Hubs (non-BL)')

    if len(beta) > 0:
        ax.scatter(beta['eigenvector'], beta['closeness'],
                   c='#d62728', s=70, alpha=0.95, linewidths=0.6,
                   edgecolors='black', zorder=5, label='Beta-lactamase')
        for _, row in beta.iterrows():
            ax.annotate(row['domain'],
                        xy=(row['eigenvector'], row['closeness']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, color='#d62728',
                        arrowprops=dict(arrowstyle='-', color='#d62728', lw=0.5))

    lcc_df = df[df['in_lcc']]
    ax.axvline(np.percentile(lcc_df['eigenvector'], HUB_PERCENTILE),
               color='grey', lw=0.8, ls='--', alpha=0.6)
    ax.axhline(np.percentile(lcc_df['closeness'],   HUB_PERCENTILE),
               color='grey', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('Eigenvector centrality', fontsize=10)
    ax.set_ylabel('Closeness centrality', fontsize=10)
    ax.set_title(lbl, fontsize=11)
    ax.legend(fontsize=8, frameon=False, markerscale=1.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(out_dir / 'bl_scatter_eigenvector_closeness.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved bl_scatter_eigenvector_closeness.png')


CORR_METRICS = ['total_degree', 'betweenness', 'eigenvector',
                'closeness', 'hits_hub_score', 'hits_authority_score']
CORR_LABELS  = ['Degree', 'Betweenness', 'Eigenvector',
                'Closeness', 'HITS hub', 'HITS auth.']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Spearman correlation between centrality metrics (all nodes)',
             fontsize=13, y=1.02)

for ax, (lbl, df) in zip(axes, centrality_results.items()):
    n = len(CORR_METRICS)
    mat = np.zeros((n, n))
    for i, m1 in enumerate(CORR_METRICS):
        for j, m2 in enumerate(CORR_METRICS):
            mat[i, j], _ = spearmanr(df[m1], df[m2])

    im = ax.imshow(mat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CORR_LABELS, rotation=40, ha='right', fontsize=9)
    ax.set_yticklabels(CORR_LABELS, fontsize=9)
    ax.set_title(lbl, fontsize=11)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(mat[i,j]) > 0.65 else 'black')

plt.colorbar(im, ax=axes[-1], shrink=0.7, label='Spearman r')
plt.tight_layout()
plt.savefig(out_dir / 'centrality_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved centrality_correlation_heatmap.png')



all_beta_domains = sorted(
    set().union(*[set(df[df['is_beta_lactamase']]['domain'])
                  for df in centrality_results.values()])
)

ALL_METRICS = ['total_degree', 'betweenness', 'eigenvector',
               'closeness', 'hits_hub_score', 'hits_authority_score']

rank_rows = []
for dom in all_beta_domains:
    row = {'domain': dom}
    for lbl, df in centrality_results.items():
        match = df[df['domain'] == dom]
        if len(match) == 0:
            for m in ALL_METRICS:
                row[f'{lbl}_{m}']      = None
                row[f'{lbl}_{m}_rank'] = None
            row[f'{lbl}_is_hub']            = None
            row[f'{lbl}_is_hits_hub']       = None
            row[f'{lbl}_is_hits_authority'] = None
            continue
        for m in ALL_METRICS:
            row[f'{lbl}_{m}']      = float(match[m].values[0])
            row[f'{lbl}_{m}_rank'] = int(
                df[m].rank(ascending=False, method='min')[match.index[0]]
            )
        row[f'{lbl}_is_hub']            = bool(match['is_hub'].values[0])
        row[f'{lbl}_is_hits_hub']       = bool(match['is_hits_hub'].values[0])
        row[f'{lbl}_is_hits_authority'] = bool(match['is_hits_authority'].values[0])
    rank_rows.append(row)

rank_df = pd.DataFrame(rank_rows)
rank_df.to_csv(out_dir / 'beta_lactamase_all_centrality_ranks.csv', index=False)

# Print compact rank view
rank_cols = (['domain'] +
             [f'{lbl}_{m}_rank'
              for lbl in centrality_results
              for m in ['total_degree', 'betweenness', 'eigenvector', 'closeness']])
print('\nBeta-lactamase centrality ranks across datasets (1 = most central):')
print(rank_df[rank_cols].to_string(index=False))
print('Saved beta_lactamase_all_centrality_ranks.csv')


if len(all_beta_domains) > 0:
    dataset_labels = list(centrality_results.keys())
    x = np.arange(len(dataset_labels))
    bl_colours = plt.cm.tab10(np.linspace(0, 0.85, max(len(all_beta_domains), 1)))

    metric_titles = {
        'total_degree':        'Total degree\n(local connectivity)',
        'betweenness':         'Betweenness\n(bottleneck / bridge)',
        'eigenvector':         'Eigenvector\n(rich neighbourhood)',
        'closeness':           'Closeness\n(geometric centrality)',
        'hits_hub_score':      'HITS hub score\n(loader / upstream)',
        'hits_authority_score':'HITS authority\n(cargo / downstream)',
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Beta-lactamase domain centrality — all metrics — across datasets',
                 fontsize=14, y=1.01)

    for ax, (metric, title) in zip(axes.flat, metric_titles.items()):
        for dom, col in zip(all_beta_domains, bl_colours):
            vals = []
            for lbl in dataset_labels:
                v = rank_df.loc[rank_df['domain'] == dom, f'{lbl}_{metric}'].values
                vals.append(float(v[0]) if len(v) > 0 and v[0] is not None else np.nan)
            ax.plot(x, vals, marker='o', linewidth=2, markersize=8,
                    color=col, label=dom, alpha=0.9)
            if not np.isnan(vals[-1]):
                ax.annotate(dom, xy=(x[-1], vals[-1]),
                            xytext=(4, 0), textcoords='offset points',
                            fontsize=7, color=col, va='center')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, fontsize=10)
        ax.set_ylabel('Score', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0, 0].legend(fontsize=8, frameon=False, loc='best')
    plt.tight_layout()
    plt.savefig(out_dir / 'beta_lactamase_all_metrics_cross_dataset.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved beta_lactamase_all_metrics_cross_dataset.png')
else:
    print('No beta-lactamase domains found in any dataset — skipping cross-dataset figure.')
    print('  beta_lac_domains:', beta_lac_domains)
    print('  Verify Pfam-A.clans.tsv is in the working directory.')



print('\nAll outputs written to:', out_dir)
for f in sorted(out_dir.iterdir()):
    print(f'  {f.name}')