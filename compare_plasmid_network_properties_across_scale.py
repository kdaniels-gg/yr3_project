# =============================================================================
# GRAPH TOPOLOGY ANALYSIS: Community structure, small-world, cross-dataset comparison
# =============================================================================
# Run interactively, block by block. Each section is self-contained.
# Designed to mirror the style of chad_standard.py / chud_species.py / hospital.py
#
# METHODS RATIONALE
# -----------------
# Community detection:
#   - Leiden algorithm (leidenalg) on the undirected projection of each graph.
#     Preferred over LPA because Leiden guarantees no badly-connected communities
#     (fixes the connectivity issue in Louvain), is reproducible with a fixed seed,
#     and returns a modularity score directly.  LPA is fast but stochastic and can
#     fragment scale-free graphs.  Leiden is the current gold standard.
#   - Modularity Q is reported; values > 0.3 are generally considered meaningful.
#   - Additionally compute: number of communities, size distribution, and the
#     fraction of nodes in the giant community.
#
# Small-world test:
#   - Watts-Strogatz small-world coefficient sigma = (C/C_rand) / (L/L_rand)
#     where C = mean clustering coefficient, L = mean shortest path length.
#     sigma >> 1 indicates small-world organisation.
#   - omega (Telesford et al. 2011) = (L_rand/L) - (C/C_latt) is also computed;
#     omega near 0 is small-world, near -1 is lattice-like, near +1 is random.
#   - Null is the configuration model (preserves degree sequence) — the right
#     choice for scale-free graphs because ER nulls have a Poisson degree
#     distribution that is categorically different from the data, making any
#     comparison meaningless.  The configuration model is the standard null
#     for small-world tests on scale-free networks (Humphries & Gurney 2008).
#   - All metrics computed on the undirected largest connected component (LCC)
#     because path length and clustering are defined for undirected connected graphs.
#     The LCC is noted as a fraction of total nodes for transparency.
#
# Comparison:
#   - All three datasets (overall / E. coli / hospital) are run through identical
#     pipelines and results are tabulated side-by-side with a summary figure.
# =============================================================================

import os
import sys
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import igraph as ig
import leidenalg

# Fix random seeds globally for reproducibility
np.random.seed(42)
RNG = np.random.default_rng(42)

# =============================================================================
# SECTION 1 — PATHS: set these to match your directory layout
# =============================================================================

graph_dir         = Path(os.path.join(os.getcwd(), 'plasmid_batched_graphs'))
species_base_dir  = Path(os.path.join(os.getcwd(), 'species_specific_plasmid_analysis', 'big_species'))
# hospital.py saves graphml files to  <cwd>/hospital_analysis/graphml/
# with the label string 'hospital' (see line 331: label = 'hospital')
hospital_dir      = Path(os.path.join(os.getcwd(), 'hospital_analysis', 'graphml'))
out_dir           = Path(os.path.join(os.getcwd(), 'graph_topology_results'))
os.makedirs(out_dir, exist_ok=True)

ECOLI_SPECIES_LABEL = 'Escherichia_coli'
HOSPITAL_LABEL      = 'hospital'   # matches label = 'hospital' in hospital.py

# Number of null-model replicates for small-world test — 100 is solid, use 50 if slow
N_NULL_SW = 100

print('Paths set. Output directory:', out_dir)


# =============================================================================
# SECTION 2 — HELPERS: graph loading and LCC extraction
# =============================================================================

def load_max_batch_overall(graph_dir):
    """Load the largest-batch overall graphml (same logic as chad_standard.py)."""
    batch_files = sorted(
        graph_dir.glob('*_domain_architecture_signed_network.graphml'),
        key=lambda p: int(p.name.split('_')[0])
    )
    if not batch_files:
        raise FileNotFoundError(f'No overall graphml files found in {graph_dir}')
    path = batch_files[-1]
    print(f'Overall: loading {path.name}')
    return nx.read_graphml(str(path)), path


def load_max_batch_species(species_base_dir, species_label):
    """Load the largest-batch species graphml (mirrors chud_species.py)."""
    sp_dir = species_base_dir / species_label
    pattern = f'batch_*_{species_label}_domain_architecture_signed_network.graphml'
    batch_files = sorted(
        sp_dir.glob(pattern),
        key=lambda p: int(p.name.split('_')[1])
    )
    if not batch_files:
        raise FileNotFoundError(f'No species graphml files found in {sp_dir}')
    path = batch_files[-1]
    print(f'E. coli: loading {path.name}')
    return nx.read_graphml(str(path)), path


def load_max_batch_hospital(hospital_dir, hospital_label):
    """Load the largest-batch hospital graphml (mirrors hospital.py)."""
    pattern = f'batch_*_{hospital_label}_domain_architecture_signed_network.graphml'
    batch_files = sorted(
        hospital_dir.glob(pattern),
        key=lambda p: int(p.name.split('_')[1])
    )
    if not batch_files:
        raise FileNotFoundError(f'No hospital graphml files in {hospital_dir}')
    path = batch_files[-1]
    print(f'Hospital: loading {path.name}')
    return nx.read_graphml(str(path)), path


def to_undirected_lcc(G):
    """
    Convert a directed graph to undirected and extract the largest connected component.
    Returns (LCC_graph, lcc_fraction) where lcc_fraction = |LCC| / |G|.
    """
    U = G.to_undirected()
    components = sorted(nx.connected_components(U), key=len, reverse=True)
    lcc_nodes  = components[0]
    lcc_frac   = len(lcc_nodes) / U.number_of_nodes() if U.number_of_nodes() > 0 else 0
    return U.subgraph(lcc_nodes).copy(), lcc_frac


print('Helper functions defined.')


# =============================================================================
# SECTION 3 — LOAD GRAPHS
# =============================================================================

G_overall,  path_overall  = load_max_batch_overall(graph_dir)
G_ecoli,    path_ecoli    = load_max_batch_species(species_base_dir, ECOLI_SPECIES_LABEL)
G_hospital, path_hospital = load_max_batch_hospital(hospital_dir, HOSPITAL_LABEL)

graphs = {
    'Overall':  G_overall,
    'E. coli':  G_ecoli,
    'Hospital': G_hospital,
}

for label, G in graphs.items():
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f'{label:10s}  nodes={n:>7,}  edges={m:>8,}  directed={nx.is_directed(G)}')


# =============================================================================
# SECTION 4 — COMMUNITY DETECTION (LEIDEN)
# =============================================================================
# Method: Leiden modularity optimisation on the undirected graph.
# We use igraph + leidenalg (same libs already imported in your scripts).
# Modularity Q:  > 0.3 = meaningful community structure
#                > 0.5 = strong community structure
# =============================================================================

def nx_to_igraph_undirected(G_nx):
    """Convert a networkx graph to an undirected igraph.Graph."""
    U  = G_nx.to_undirected()
    nodes = list(U.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    edges_ig = [(node_idx[u], node_idx[v]) for u, v in U.edges()]
    g_ig = ig.Graph(n=len(nodes), edges=edges_ig, directed=False)
    g_ig.vs['name'] = nodes
    return g_ig, nodes


def run_leiden(G_nx, n_iterations=10, seed=42):
    """
    Run Leiden community detection.
    Returns dict with: n_communities, modularity, sizes, giant_fraction, partition
    """
    g_ig, nodes = nx_to_igraph_undirected(G_nx)
    partition = leidenalg.find_partition(
        g_ig,
        leidenalg.ModularityVertexPartition,
        n_iterations=n_iterations,
        seed=seed
    )
    modularity    = partition.modularity
    sizes         = sorted([len(c) for c in partition], reverse=True)
    n_communities = len(sizes)
    n_nodes       = G_nx.number_of_nodes()
    giant_frac    = sizes[0] / n_nodes if n_nodes > 0 else 0

    return {
        'n_communities':   n_communities,
        'modularity_Q':    round(modularity, 4),
        'community_sizes': sizes,
        'giant_community_fraction': round(giant_frac, 4),
        'median_community_size':    int(np.median(sizes)),
        'partition':       partition,
        'igraph':          g_ig,
    }


community_results = {}
for label, G in graphs.items():
    print(f'\nRunning Leiden on {label} ...')
    res = run_leiden(G)
    community_results[label] = res
    print(f'  Communities : {res["n_communities"]}')
    print(f'  Modularity Q: {res["modularity_Q"]}')
    print(f'  Largest community fraction: {res["giant_community_fraction"]:.3f}')
    print(f'  Median community size: {res["median_community_size"]}')


# =============================================================================
# SECTION 5 — COMMUNITY SIZE DISTRIBUTIONS (plot)
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Community size distributions (Leiden)', fontsize=14)
colours = ['steelblue', 'darkgreen', 'firebrick']

for ax, (label, res), colour in zip(axes, community_results.items(), colours):
    sizes = np.array(res['community_sizes'])
    # log-binned histogram for scale-free-style display
    bins = np.logspace(np.log10(max(1, sizes.min())), np.log10(sizes.max()), 30)
    ax.hist(sizes, bins=bins, color=colour, alpha=0.8, edgecolor='none')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Community size (nodes)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'{label}\nQ={res["modularity_Q"]:.3f}, '
                 f'n={res["n_communities"]} communities', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(out_dir / 'community_size_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved community_size_distributions.png')


# =============================================================================
# SECTION 6 — SMALL-WORLD TEST
# =============================================================================
# We compute sigma and omega on the undirected LCC.
# Null: configuration model (preserves degree sequence).
#
# For large graphs (n > 5000), exact average shortest path length is expensive.
# We use a sample of k=500 source nodes (standard practice; Humphries & Gurney 2008).
#
# sigma = (C_obs/C_rand) / (L_obs/L_rand)   — Watts & Strogatz 1998
# omega = (L_rand/L_obs) - (C_obs/C_latt)   — Telesford et al. 2011
# Both use the configuration-model null.
# Lattice reference for omega: ring lattice with same n, m.
# =============================================================================

SAMPLE_K = 500   # source nodes for APL estimation (set lower if memory limited)

def estimate_apl(G_undirected, k=SAMPLE_K, rng=None):
    """
    Estimate average path length by sampling k source nodes.
    Returns float (mean shortest path over reachable pairs).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    nodes = list(G_undirected.nodes())
    if len(nodes) <= k:
        sources = nodes
    else:
        sources = rng.choice(nodes, size=k, replace=False).tolist()
    lengths = []
    for src in sources:
        sp = nx.single_source_shortest_path_length(G_undirected, src)
        lengths.extend(v for v in sp.values() if v > 0)
    return float(np.mean(lengths)) if lengths else float('inf')


def mean_clustering(G_undirected):
    """Average clustering coefficient (undirected)."""
    return nx.average_clustering(G_undirected)


def config_null_lcc(G_directed, rng=None, seed=None):
    """
    Build one configuration-model null (preserving in/out degrees),
    convert to undirected, return its LCC.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    in_seq  = [d for _, d in G_directed.in_degree()]
    out_seq = [d for _, d in G_directed.out_degree()]
    H = nx.directed_configuration_model(
        in_seq, out_seq,
        create_using=nx.DiGraph(),
        seed=int(rng.integers(1e9))
    )
    H.remove_edges_from(nx.selfloop_edges(H))
    lcc_H, _ = to_undirected_lcc(H)
    return lcc_H


def ring_lattice_lcc(n, m):
    """
    Ring lattice with n nodes and approximately m edges (nearest-neighbour),
    used as the lattice reference for omega.
    """
    k = max(2, int(round(m / n)))   # each node connects to k/2 neighbours on each side
    k = k if k % 2 == 0 else k + 1
    H = nx.watts_strogatz_graph(n, k, 0)  # p=0 → pure lattice
    return H


def small_world_stats(G_directed, label='', n_null=N_NULL_SW, sample_k=SAMPLE_K):
    """
    Compute sigma and omega for G_directed against configuration-model nulls.

    Returns dict with all intermediate values plus sigma, omega, p-values.
    """
    print(f'\n  [{label}] Extracting LCC ...')
    lcc, lcc_frac = to_undirected_lcc(G_directed)
    n_lcc = lcc.number_of_nodes()
    m_lcc = lcc.number_of_edges()
    print(f'  [{label}] LCC: {n_lcc} nodes ({lcc_frac:.2%} of total), {m_lcc} edges')

    print(f'  [{label}] Computing observed C and L ...')
    C_obs = mean_clustering(lcc)
    L_obs = estimate_apl(lcc, k=sample_k, rng=RNG)
    print(f'  [{label}] C_obs={C_obs:.4f}, L_obs={L_obs:.4f}')

    print(f'  [{label}] Building {n_null} configuration-model nulls ...')
    C_null_vals, L_null_vals = [], []
    rng_null = np.random.default_rng(42)
    for i in range(n_null):
        H_null = config_null_lcc(G_directed, rng=rng_null)
        if H_null.number_of_nodes() < 3:
            continue
        C_null_vals.append(mean_clustering(H_null))
        L_null_vals.append(estimate_apl(H_null, k=min(sample_k, H_null.number_of_nodes()), rng=rng_null))
        if (i + 1) % 10 == 0:
            print(f'    null {i+1}/{n_null}')

    C_rand = float(np.mean(C_null_vals))
    L_rand = float(np.mean(L_null_vals))
    C_rand_std = float(np.std(C_null_vals))
    L_rand_std = float(np.std(L_null_vals))
    print(f'  [{label}] C_rand={C_rand:.4f} ± {C_rand_std:.4f}')
    print(f'  [{label}] L_rand={L_rand:.4f} ± {L_rand_std:.4f}')

    # sigma (Watts-Strogatz small-world coefficient)
    sigma = (C_obs / C_rand) / (L_obs / L_rand) if (C_rand > 0 and L_rand > 0) else np.nan

    # omega (Telesford; needs lattice reference for C_latt)
    print(f'  [{label}] Building ring-lattice reference for omega ...')
    latt = ring_lattice_lcc(n_lcc, m_lcc)
    C_latt = mean_clustering(latt)
    omega  = (L_rand / L_obs) - (C_obs / C_latt) if (L_obs > 0 and C_latt > 0) else np.nan
    print(f'  [{label}] C_latt={C_latt:.4f}')
    print(f'  [{label}] sigma={sigma:.4f}, omega={omega:.4f}')

    # One-sample z-tests: is C_obs > C_rand? Is L_obs ~ L_rand?
    # (informally — proper test is just inspecting sigma/omega)
    z_C = (C_obs - C_rand) / C_rand_std if C_rand_std > 0 else np.nan
    z_L = (L_obs - L_rand) / L_rand_std if L_rand_std > 0 else np.nan

    return {
        'label':        label,
        'n_lcc':        n_lcc,
        'lcc_fraction': round(lcc_frac, 4),
        'C_obs':        round(C_obs, 5),
        'L_obs':        round(L_obs, 4),
        'C_rand':       round(C_rand, 5),
        'L_rand':       round(L_rand, 4),
        'C_rand_std':   round(C_rand_std, 5),
        'L_rand_std':   round(L_rand_std, 4),
        'C_latt':       round(C_latt, 5),
        'sigma':        round(sigma, 4),
        'omega':        round(omega, 4),
        'z_C':          round(z_C, 3),
        'z_L':          round(z_L, 3),
        'C_null_vals':  C_null_vals,
        'L_null_vals':  L_null_vals,
    }


sw_results = {}
for label, G in graphs.items():
    print(f'\nSmall-world analysis: {label}')
    sw_results[label] = small_world_stats(G, label=label, n_null=N_NULL_SW)


# =============================================================================
# SECTION 7 — SMALL-WORLD NULL DISTRIBUTIONS (plot)
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Small-world null distributions (configuration model)', fontsize=14)

for col, (label, res) in enumerate(sw_results.items()):
    ax_c = axes[0, col]
    ax_l = axes[1, col]

    # Clustering coefficient
    ax_c.hist(res['C_null_vals'], bins=20, color='steelblue', alpha=0.75, label='null C')
    ax_c.axvline(res['C_obs'],  color='darkred',    lw=2, linestyle='--', label=f'observed C={res["C_obs"]:.4f}')
    ax_c.axvline(res['C_rand'], color='steelblue',  lw=1, linestyle=':',  label=f'null mean={res["C_rand"]:.4f}')
    ax_c.set_title(f'{label} — clustering', fontsize=11)
    ax_c.set_xlabel('Mean clustering coefficient', fontsize=9)
    ax_c.set_ylabel('Count', fontsize=9)
    ax_c.legend(fontsize=8, frameon=False)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # Average path length
    ax_l.hist(res['L_null_vals'], bins=20, color='darkorange', alpha=0.75, label='null L')
    ax_l.axvline(res['L_obs'],  color='darkred',   lw=2, linestyle='--', label=f'observed L={res["L_obs"]:.3f}')
    ax_l.axvline(res['L_rand'], color='darkorange', lw=1, linestyle=':',  label=f'null mean={res["L_rand"]:.3f}')
    ax_l.set_title(f'{label} — path length', fontsize=11)
    ax_l.set_xlabel('Mean shortest path length (sampled)', fontsize=9)
    ax_l.set_ylabel('Count', fontsize=9)
    ax_l.legend(fontsize=8, frameon=False)
    ax_l.spines['top'].set_visible(False)
    ax_l.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(out_dir / 'small_world_null_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved small_world_null_distributions.png')


# =============================================================================
# SECTION 8 — SUMMARY TABLE
# =============================================================================

summary_rows = []
for label in graphs:
    sw  = sw_results[label]
    com = community_results[label]
    G   = graphs[label]
    row = {
        'Dataset':              label,
        'Nodes':                G.number_of_nodes(),
        'Edges':                G.number_of_edges(),
        'LCC_nodes':            sw['n_lcc'],
        'LCC_fraction':         sw['lcc_fraction'],
        # Community
        'N_communities':        com['n_communities'],
        'Modularity_Q':         com['modularity_Q'],
        'Giant_comm_fraction':  com['giant_community_fraction'],
        'Median_comm_size':     com['median_community_size'],
        # Small-world
        'C_obs':                sw['C_obs'],
        'C_rand_mean':          sw['C_rand'],
        'C_rand_std':           sw['C_rand_std'],
        'L_obs':                sw['L_obs'],
        'L_rand_mean':          sw['L_rand'],
        'L_rand_std':           sw['L_rand_std'],
        'C_latt':               sw['C_latt'],
        'sigma':                sw['sigma'],
        'omega':                sw['omega'],
        'z_C':                  sw['z_C'],
        'z_L':                  sw['z_L'],
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index('Dataset')
print('\n', summary_df.T.to_string())

summary_df.to_csv(out_dir / 'topology_summary.csv')
print('\nSaved topology_summary.csv')


# =============================================================================
# SECTION 9 — CROSS-DATASET COMPARISON FIGURE
# =============================================================================
# Four-panel figure: modularity Q, sigma, omega, and C_obs/C_rand ratio

datasets  = list(summary_df.index)
x         = np.arange(len(datasets))
bar_width = 0.55
colours   = ['steelblue', 'darkgreen', 'firebrick']

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# --- Panel A: Modularity Q ---
ax_q = fig.add_subplot(gs[0, 0])
bars = ax_q.bar(x, summary_df['Modularity_Q'], width=bar_width,
                color=colours, alpha=0.85, edgecolor='none')
ax_q.axhline(0.3, color='grey', lw=1, linestyle='--', label='Q=0.3 threshold')
ax_q.set_xticks(x); ax_q.set_xticklabels(datasets, fontsize=10)
ax_q.set_ylabel('Modularity Q', fontsize=11)
ax_q.set_title('Community modularity (Leiden)', fontsize=12)
ax_q.set_ylim(0, max(summary_df['Modularity_Q'].max() * 1.3, 0.5))
ax_q.legend(fontsize=9, frameon=False)
ax_q.spines['top'].set_visible(False); ax_q.spines['right'].set_visible(False)
for bar, val in zip(bars, summary_df['Modularity_Q']):
    ax_q.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
              f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# --- Panel B: sigma ---
ax_sigma = fig.add_subplot(gs[0, 1])
bars_s = ax_sigma.bar(x, summary_df['sigma'], width=bar_width,
                      color=colours, alpha=0.85, edgecolor='none')
ax_sigma.axhline(1.0, color='grey', lw=1, linestyle='--', label='sigma=1 (random)')
ax_sigma.set_xticks(x); ax_sigma.set_xticklabels(datasets, fontsize=10)
ax_sigma.set_ylabel('sigma (small-world coefficient)', fontsize=11)
ax_sigma.set_title('Small-world sigma\n(sigma > 1 = small-world)', fontsize=12)
ax_sigma.legend(fontsize=9, frameon=False)
ax_sigma.spines['top'].set_visible(False); ax_sigma.spines['right'].set_visible(False)
for bar, val in zip(bars_s, summary_df['sigma']):
    ax_sigma.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                  f'{val:.2f}', ha='center', va='bottom', fontsize=9)

# --- Panel C: omega ---
ax_omega = fig.add_subplot(gs[1, 0])
bars_o = ax_omega.bar(x, summary_df['omega'], width=bar_width,
                      color=colours, alpha=0.85, edgecolor='none')
ax_omega.axhline(0.0, color='grey', lw=1, linestyle='--', label='omega=0 (small-world)')
ax_omega.set_xticks(x); ax_omega.set_xticklabels(datasets, fontsize=10)
ax_omega.set_ylabel('omega', fontsize=11)
ax_omega.set_title('Small-world omega\n(near 0 = small-world; −1 = lattice; +1 = random)', fontsize=12)
ax_omega.legend(fontsize=9, frameon=False)
ax_omega.spines['top'].set_visible(False); ax_omega.spines['right'].set_visible(False)
for bar, val in zip(bars_o, summary_df['omega']):
    ypos = val + 0.01 if val >= 0 else val - 0.03
    ax_omega.text(bar.get_x() + bar.get_width()/2, ypos,
                  f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# --- Panel D: C_obs vs C_rand (error bars from null std) ---
ax_c = fig.add_subplot(gs[1, 1])
w = 0.3
x_obs   = x - w/2
x_rand  = x + w/2
ax_c.bar(x_obs,  summary_df['C_obs'],      width=w, color=colours, alpha=0.9,
         edgecolor='none', label='Observed C')
ax_c.bar(x_rand, summary_df['C_rand_mean'], width=w,
         color=[c for c in colours], alpha=0.4,
         edgecolor='none', label='Null C (config model)')
ax_c.errorbar(x_rand, summary_df['C_rand_mean'],
              yerr=summary_df['C_rand_std'],
              fmt='none', color='black', capsize=4, lw=1.2)
ax_c.set_xticks(x); ax_c.set_xticklabels(datasets, fontsize=10)
ax_c.set_ylabel('Mean clustering coefficient', fontsize=11)
ax_c.set_title('Observed vs null clustering\n(solid = observed, faded = null ± SD)', fontsize=12)
ax_c.legend(fontsize=9, frameon=False)
ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)

plt.suptitle('Domain architecture network topology — cross-dataset comparison', fontsize=14, y=1.01)
plt.savefig(out_dir / 'cross_dataset_topology_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved cross_dataset_topology_comparison.png')







# =============================================================================
# SECTION 10 — INTERPRETATION GUIDE (prints to console)
# =============================================================================

print('\n' + '='*70)
print('INTERPRETATION GUIDE')
print('='*70)
print("""
Community structure (Leiden modularity Q):
  Q > 0.3  — meaningful community structure present
  Q > 0.5  — strong community structure
  Values near 0 — little or no community organisation

Small-world sigma:
  sigma >> 1  — small-world (high clustering, short paths vs random)
  sigma ~ 1   — random-graph-like (no small-world property)
  sigma < 1   — lattice-like (long paths, high clustering)

Small-world omega (Telesford 2011):
  omega ~ 0   — small-world
  omega ~ -1  — lattice-like
  omega ~ +1  — random-graph-like
  omega is more robust than sigma for heterogeneous (scale-free) networks.

z_C: z-score of observed C vs null C distribution.
  Large positive z_C means observed graph is MORE clustered than expected
  by degree alone — the key signature of small-world organisation.

z_L: z-score of observed L vs null L distribution.
  Small absolute z_L means path lengths are similar to random — also
  a key small-world signature.
""")

print('All outputs written to:', out_dir)
print('Files:')
for f in sorted(out_dir.iterdir()):
    print(f'  {f.name}')























import polars as pl
import math

data_dir = Path(os.path.join(os.getcwd(),'plasmid_motif_network/intermediate'))
files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))

df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
df_merged = df_merged.with_columns(
    pl.col('strand').cast(pl.Int32).alias('strand')
)

domain_df = pd.read_csv('Pfam-A.clans.tsv', sep='\t', header=None)
domain_dict = dict(zip(domain_df[3].tolist(), domain_df[2].tolist()))


domain_dict_clean = {
    k: (None if isinstance(v, float) and math.isnan(v) else v)
    for k, v in domain_dict.items()
}

df_merged = df_merged.with_columns(
    pl.col('target_name').replace(domain_dict_clean).alias('clan')
)

df_merged_pfam_beta = df_merged.filter(pl.col('clan') == 'Beta-lactamase')