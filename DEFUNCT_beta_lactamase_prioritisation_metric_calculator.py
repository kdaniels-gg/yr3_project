#Gene-level centrality is defined as the maximum centrality among Pfam domain nodes associated with that gene. This avoids inflation caused by multi-domain proteins contributing multiple nodes to the graph.
#alternative would be gene-name level collapse.

#the pooled position filtering in block 8 uses plasmid-level matching to restrict which domain_positions entries belong to this gene's PIDs
# if two different genes share the same pfam domain and the same plasmid, some positions may be excluded.
#(you don't want TEM-1's centrality polluted by SHV-1's occurrences of the same pfam domain
#ie does this domain position belong to a PID mapped to this gene? - where is the PID on the domain ordered list


#betweeness centrality - betweenness of node v
# sum over other node pairs of number of shortest paths passing through v / nunber of shortest paths between those nodes
# ie the proportion of shortest paths of other nodes that goes through that hub

## ============================================================
## PER-BETA-LACTAMASE-GENE METRICS
## For each AMRFinder gene name (e.g. TEM-1, CTX-M-15, SHV-1):
##   - Raw Shannon entropy of genomic neighbourhood contexts
##   - Relative Shannon entropy (vs background domains on same plasmids)
##   - Duplication event rate
##   - Betweenness centrality  (structural bottleneck / bridge)
##   - Eigenvector centrality  (neighbourhood quality / hub embeddedness)
##   - Closeness centrality    (geometric integration into domain landscape)
##
## NOTE ON HITS / LOADER-CARGO:
##   HITS is intentionally excluded. The directed graph encodes upstream-in-
##   coordinate-space, not biological transfer direction. Without integrating
##   per-domain strand information and a validated model of which orientation
##   constitutes a functional MGE-passenger pair, the hub/authority labels are
##   not interpretable as loader/cargo. Betweenness + eigenvector + closeness
##   on the undirected LCC are the three most informative and interpretable
##   metrics for this purpose.
##
## NOTE ON DEGREE:
##   degree_centrality (normalised total degree) is included as a fourth metric
##   because it is the simplest to interpret and complements the three above,
##   but is computed on the full directed graph (consistent with pfam_hubs.py).
## ============================================================

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


# ── 0. Paths and parameters ───────────────────────────────────────────────────

data_dir   = Path(os.path.join(os.getcwd(), 'plasmid_motif_network/intermediate'))
graph_dir  = Path(os.path.join(os.getcwd(), 'plasmid_batched_graphs'))
out_dir    = Path(os.path.join(os.getcwd(), 'bl_gene_metrics'))
out_dir.mkdir(exist_ok=True)

MERGED_FASTA_DIR = Path('merged_nonoverlapping_fastas')


MIN_OBS    = 10    # minimum copies for entropy/relative-entropy to be computed
HUB_PCT    = 95   # top (100-HUB_PCT)% = hub threshold, consistent with pfam_hubs.py


# ── 1. Load df_merged ─────────────────────────────────────────────────────────

print("Loading parquet files...")
files = sorted(data_dir.glob('parsed_selected_nonoverlap_*.parquet'))
df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
df_merged = df_merged.with_columns(pl.col('strand').cast(pl.Int32))

# Build ordered domain list per plasmid (consistent with entropy_dist.py)
ordered_df = df_merged.sort(['plasmid', 'start', 'ali_from']).select(['plasmid', 'target_name'])

print("Building circular domain lists...")
plasmid_to_domains = defaultdict(list)
for row in ordered_df.iter_rows(named=True):
    plasmid_to_domains[row['plasmid']].append(row['target_name'])

# domain_positions[domain] = list of (plasmid, index_within_plasmid)
domain_positions = defaultdict(list)
for plasmid, doms in plasmid_to_domains.items():
    n = len(doms)
    if n <= 1:
        continue
    for i, dom in enumerate(doms):
        domain_positions[dom].append((plasmid, i))

print(f"Loaded {len(domain_positions)} unique domains across {len(plasmid_to_domains)} plasmids.")


# ── 2. Load AMRFinder gene mapping ────────────────────────────────────────────

merged_kept_PIDs = set('.'.join(x.split('.')[:-1]) for x in os.listdir(MERGED_FASTA_DIR))


#PID_nogene_pattern  = re.compile(r'^(.+?)_(\d+)_(\d+)$')
#any(PID_nogene_pattern.match(x) == None for x in test['query_id'].tolist())
#only change in new test and old test is the pfam domains


test_new = pd.read_csv('amrfindermapped_beta_lactamases_new.csv', low_memory=False)
test_new = test_new.loc[test_new['query_id'].isin(merged_kept_PIDs)].copy()

prev_mapped_names = ['TEM-1', 'CTX-M-140', 'CMY-4', 'CTX-M-9', 'IMP-22', 'CTX-M-2', 'PAD', 'VEB', 'CTX-M-44', 'NDM-31', 'SFO-1', 'ROB-1', 'OXA-926', 'SHV', 'AFM-1', 'TEM-181', 'CTX-M-59', 'NDM-4', 'DHA-7', 'VIM', 'CTX-M-63', 'OXA-1041', 'CTX-M-125', 'CTX-M-123', 'OXA-1203', 'IMP-19', 'CTX-M-243', 'AFM-3', 'OXA-1204', 'bla-A2', 'LMB-1', 'GES-11', 'mecR1', 'OXA-1', 'BKC-1', 'FOX', 'CTX-M-136', 'OXA-58', 'NDM-1', 'CTX-M-32', 'TEM-171', 'KPC-6', 'KPC-49', 'CTX-M-30', 'CARB-16', 'KPC-93', 'L2', 'LAP-1', 'OXA-567', 'TEM-215', 'CMY-23', 'IMI', 'CMY-111', 'SHV-2A', 'RCP', 'OXA-19', 'OXA-436', 'VEB-18', 'OXA-237', 'NDM-19', 'CTX-M-17', 'IMI-16', 'CMY-6', 'CMY-172', 'VEB-5', 'KPC-109', 'FOX-5', 'IMP-100', 'OXA-655', 'PAU-1', 'TEM-21', 'OXA-96', 'VEB-16', 'VIM-19', 'IMP-56', 'OXA-2', 'SHV-30', 'CTX-M-25', 'SHV-28', 'IMP-45', 'IMP-26', 'CMY-148', 'OKP-B', 'VIM-2', 'CTX-M-40', 'KPC-204', 'OXA-932', 'TEM-4', 'mecA', 'OXA-420', 'KPC-121', 'NDM-5', 'NPS-1', 'KPC-3', 'TEM-12', 'ELC', 'KPC-113', 'MOX', 'OXA-164', 'HBL', 'PDC-16', 'CARB-2', 'OXA-653', 'PER-4', 'CTX-M-104', 'SHV-11', 'TEM-156', 'R39', 'PSV', 'GES-20', 'NDM-27', 'PEN-B', 'DIM-1', 'OXA-9', 'IMP-69', 'OXA-246', 'PER-1', 'VIM-7', 'OXA-198', 'CTX-M-173', 'TEM-61', 'OXA-101', 'TEM-34', 'CAE-1', 'MUN-1', 'NDM-29', 'TEM-3', 'MYO-1', 'SHV-7', 'OXA-97', 'VIM-84', 'OXA-438', 'CMY', 'VEB-2', 'KPC-33', 'GES-12', 'RAHN', 'NDM', 'IMP-14', 'VIM-11', 'IMP-63', 'CTX-M-226', 'VEB-8', 'CARB-8', 'CMY-97', 'ROB', 'CTX-M-53', 'KPC-40', 'OXA-244', 'SHV-5', 'mecB', 'ROB-11', 'KLUC-5', 'VIM-61', 'TEM-84', 'KPC-154', 'CMY-185', 'OXY-2-16', 'TEM-169', 'cfiA2', 'MCA', 'TEM-168', 'RSC1', 'IMP-23', 'CTX-M-62', 'OXA-732', 'CTX-M-195', 'NDM-9', 'CMY-166', 'VIM-85', 'ADC-30', 'NDM-7', 'TEM-116', 'TMB-1', 'VIM-86', 'CTX-M-174', 'bla-C', 'KPC-29', 'IMP-18', 'OXA-256', 'FRI-3', 'OXA-162', 'NPS', 'TEM-54', 'BIM-1', 'CTX-M-90', 'KPC-125', 'KPC-66', 'RAA-1', 'OXA-66', 'OXA-21', 'CMY-16', 'CTX-M-55', 'CMY-146', 'VEB-9', 'CTX-M-8', 'VHW', 'KPC-17', 'OXA-24', 'SHV-2', 'ADC-176', 'TEM-176', 'cdiA', 'CARB-4', 'KPC-4', 'KPC-14', 'GES-5', 'bla-A', 'pbp2m', 'OXA-232', 'VEB-3', 'CMY-36', 'TEM-20', 'CTX-M-39', 'VEB-25', 'CTX-M-65', 'IMP-11', 'ACC-1', 'OXA-181', 'OXY', 'mecI_of_mecA', 'NDM-17', 'SHV-31', 'KPC-78', 'CTX-M-5', 'IMP-38', 'CMY-44', 'CTX-M-134', 'LCR', 'GES-51', 'IMP-89', 'OXA-779', 'SHV-18', 'CMY-174', 'GIM-1', 'TER', 'GES-1', 'IMP-31', 'CMY-145', 'SHV-1', 'VIM-60', 'CTX-M-130', 'TEM-30', 'TEM-7', 'LAP-2', 'VIM-1', 'GES-44', 'CMY2-MIR-ACT-EC', 'LAP', 'CMY-2', 'RAHN-3', 'FRI-7', 'OXA-1391', 'OXA-82', 'FRI-4', 'SHV-12', 'bla-A_firm', 'CTX-M-64', 'OXA-209', 'OXA', 'DHA-15', 'BKC-2', 'IMI-23', 'TEM-24', 'bla-B1', 'R1', 'CTX-M-15', 'OXA-893', 'ADC', 'CMY-13', 'TEM-40', 'FRI-11', 'CTX-M-215', 'OXA-4', 'IMI-6', 'OXA-517', 'CMY-136', 'CTX-M-1', 'KPC-5', 'TEM-10', 'IMI-5', 'CTX-M-38', 'CTX-M-71', 'OXA-139', 'DHA-27', 'BRO', 'KPC-21', 'OXA-921', 'CMY-10', 'OXA-23', 'KHM-1', 'TEM-57', 'CTX-M-132', 'CTX-M-131', 'OXA-32', 'IMP-10', 'TEM-144', 'FRI-9', 'SCO-1', 'CAE', 'KPC-8', 'LCR-1', 'IMP-1', 'OXA-48', 'RTG', 'KPC-79', 'CMY-141', 'FRI-5', 'OXA-17', 'TEM-237', 'CTX-M-234', 'GES-14', 'NDM-13', 'VIM-27', 'CTX-M-27', 'NDM-36', 'KPC-112', 'KPC-111', 'NDM-14', 'GES-4', 'KPC-53', 'VEB-17', 'CARB-12', 'TEM-52', 'OXA-207', 'TEM-32', 'IMP-94', 'KPC-31', 'OXA-427', 'CTX-M-3', 'CTX-M', 'GES-6', 'SIM-2', 'OXA-520', 'OXA-897', 'bla1', 'VEB-1', 'PER-7', 'CTX-M-58', 'TEM-6', 'PSE', 'BES-1', 'CMY-178', 'NDM-6', 'BEL-1', 'Z', 'ADC-130', 'IMP-13', 'OXA-347', 'OXA-484', 'CTX-M-255', 'ACT-9', 'VIM-24', 'OXA-519', 'I', 'FRI-8', 'OXA-656', 'IMP', 'HER-3', 'PER-2', 'NDM-16b', 'CMY-5', 'OXA-29', 'IMP-64', 'IMP-6', 'TEM-256', 'VAM-1', 'CTX-M-236', 'PEN-bcc', 'ROB-2', 'KPC-84', 'TEM-37', 'bla-A_carba', 'CTX-M-102', 'ACC-4', 'VIM-66', 'FONA', 'CTX-M-14', 'KPC-18', 'OXA-1397', 'cfxA_fam', 'IMI-2', 'MOX-1', 'KPC-12', 'KPC-74', 'KPC-90', 'CTX-M-105', 'TEM-31', 'CTX-M-199', 'CTX-M-24', 'OXA-695', 'TEM-135', 'TEM-26', 'OXA-935', 'PER', 'TEM', 'IMP-96', 'IMP-8', 'NDM-23', 'KPC-67', 'PER-3', 'OXA-47', 'CTX-M-121', 'III', 'PSZ', 'KPC-189', 'CTX-M-98', 'CTX-M-101', 'OXA-1042', 'VCC-1', 'CMY-8', 'ACC', 'AFM-4', 'KPC-70', 'PEN-J', 'DIM', 'NDM-37', 'SHV-44', 'GES-24', 'PAU', 'VIM-23', 'IMI-22', 'OXA-392', 'KPC-110', 'KPC-2', 'CMY-31', 'TEM-33', 'MOX-18', 'VIM-4', 'OXA-10', 'KPC-71', 'KPC-44', 'GMA-1', 'OXA-235', 'CTX-M-37', 'CMY-42', 'OXA-204', 'DHA-1', 'FOX-7', 'VMB-1', 'TEM-210', 'OXA-796', 'PC1', 'OXA-900', 'GES-19', 'TEM-238', 'FRI-6', 'FLC-1', 'CTX-M-253', 'GES-7', 'SIM-1', 'TEM-206', 'IMP-4', 'KPC-87', 'FRI-2', 'OXA-72', 'KPC', 'SHV-102', 'OXA-1202', 'TLA-3', 'OXA-163', 'DHA', 'TEM-190', 'TEM-2', 'OXA-129', 'GES', 'VIM-6', 'KPC-24', 'PNC', 'AFM-2', 'KPC-55', 'PSZ-1', 'CTX-M-251', 'IMP-34']
new_mapped_names = ['NDM-3', 'NDM-11', 'NDM-20', 'NDM-21', 'VIM-5']
all_bl_mapped_names = prev_mapped_names + new_mapped_names 

test_new = test_new.loc[test_new['gene_name'].isin(all_bl_mapped_names)]
test_old = pd.read_csv('amrfindermapped_beta_lactamases_old.csv', low_memory=False)
test_old = test_old.loc[test_old['query_id'].isin(merged_kept_PIDs)].copy()

test = pd.concat([test_old, test_new]).drop_duplicates(keep='first')
test.to_csv('amrfindermapped_beta_lactamases.csv', index=False)



all_gene_names = [x for x in test['gene_name'].unique() if isinstance(x, str)]
gene_to_family = {
    gname: test.loc[test['gene_name'] == gname, 'gene_family'].iloc[0]
    for gname in all_gene_names
}
# gene_name → set of query_names (PIDs)
gene_to_pids = defaultdict(set)
for _, row in test.iterrows():
    if isinstance(row['gene_name'], str):
        gene_to_pids[row['gene_name']].add(row['query_id'])

print(f"Loaded {len(all_gene_names)} BL gene names from AMRFinder mapping.")


#get beta-lactamase domains

bl_df_merged = df_merged.filter(pl.col('target_name').str.contains('lactamase')|pl.col('target_name').str.contains('Lactamase')|pl.col('query_name').is_in(test['query_id'].tolist()))
beta_lac_pfam_domains = set(list(bl_df_merged['target_name']))



# ── 4. Entropy and duplication functions (from entropy_dist.py) ───────────────

def context_entropy_score(contexts):
    """
    contexts : list of (left_domain, right_domain) tuples
    Returns  : (penalised_score, raw_H, N, K)
      raw_H  = Shannon entropy of the context distribution
      N      = total observations
      K      = unique context types
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


def get_contexts_and_dup_rate(plasmid_idx_list, dom):
    """
    For all occurrences of dom across plasmids, collect (left, right) context
    tokens and detect tandem duplications (same domain immediately adjacent).
    """
    neighbour_tokens  = []
    plasmids_seen     = set()
    duplication_events = 0
    for plasmid, idx in plasmid_idx_list:
        entries = plasmid_to_domains[plasmid]
        n = len(entries)
        if n <= 1:
            continue
        left  = entries[(idx - 1) % n]
        right = entries[(idx + 1) % n]
        neighbour_tokens.append((left, right))
        plasmids_seen.add(plasmid)
        if left  == dom:
            duplication_events += 1
        if right == dom:
            duplication_events += 1
    dup_rate = duplication_events / len(neighbour_tokens) if neighbour_tokens else 0.0
    return neighbour_tokens, plasmids_seen, duplication_events, dup_rate


# ── 5. Compute entropy metrics for ALL domains (needed for relative entropy) ──
# This replicates the background computation from entropy_dist.py.

print("\nComputing entropy for all domains (background for relative entropy)...")

all_domain_data   = {}   # dom → full stats dict
all_domain_raw_H  = {}   # dom → raw H  (for background lookup, MIN_OBS filtered)

for dom, pos_list in tqdm(domain_positions.items(), desc='All-domain entropy'):
    tokens, plas_seen, dup_events, dup_rate = get_contexts_and_dup_rate(pos_list, dom)
    _, H, n_copies, n_unique = context_entropy_score(tokens)
    all_domain_data[dom] = {
        'plas_seen'  : plas_seen,
        'raw_entropy': H,
        'n_copies'   : n_copies,
        'n_unique'   : n_unique,
        'dup_events' : dup_events,
        'dup_rate'   : dup_rate,
    }
    if n_copies >= MIN_OBS and not np.isnan(H):
        all_domain_raw_H[dom] = H


# ── 6. Load the full-dataset directed graph for centrality ────────────────────
# Use the largest batch (consistent with pfam_hubs.py load_max_batch_overall).

print("\nLoading domain architecture graph...")
batch_files = sorted(
    graph_dir.glob('*_domain_architecture_signed_network.graphml'),
    key=lambda p: int(p.name.split('_')[0])
)
if not batch_files:
    raise FileNotFoundError(f'No graphml files found in {graph_dir}')
G_directed = nx.read_graphml(str(batch_files[-1]))
print(f"Graph loaded: {G_directed.number_of_nodes():,} nodes, {G_directed.number_of_edges():,} edges")

# Build undirected LCC — used for betweenness, eigenvector, closeness
U         = G_directed.to_undirected()
lcc_nodes = max(nx.connected_components(U), key=len)
U_lcc     = U.subgraph(lcc_nodes).copy()
lcc_frac  = len(lcc_nodes) / G_directed.number_of_nodes()
print(f"Undirected LCC: {U_lcc.number_of_nodes():,} nodes ({lcc_frac:.1%} of total)")


# ── 7. Compute centrality metrics ─────────────────────────────────────────────

print("\nComputing centrality metrics...")

# Degree centrality — full directed graph
print("  [1/4] Degree centrality...")
deg_cent  = nx.degree_centrality(G_directed)       # normalised (in+out) / (n-1)
in_deg    = dict(G_directed.in_degree())
out_deg   = dict(G_directed.out_degree())
total_deg = {n: in_deg[n] + out_deg[n] for n in G_directed.nodes()}

# Betweenness — undirected LCC
print("  [2/4] Betweenness centrality (Brandes, undirected LCC)...")
betw = nx.betweenness_centrality(U_lcc, normalized=True)

# Eigenvector — undirected LCC
print("  [3/4] Eigenvector centrality...")
try:
    eig = nx.eigenvector_centrality(U_lcc, max_iter=1000, tol=1e-6)
except nx.PowerIterationFailedConvergence:
    print("    WARNING: eigenvector did not converge — using normalised degree as proxy")
    raw = {n: U_lcc.degree(n) for n in U_lcc.nodes()}
    mx  = max(raw.values()) or 1
    eig = {n: v / mx for n, v in raw.items()}

# Closeness — undirected LCC, Wasserman-Faust correction
print("  [4/4] Closeness centrality...")
close = nx.closeness_centrality(U_lcc, wf_improved=True)

# Hub thresholds (top HUB_PCT percentile on each metric, across ALL graph nodes)
all_nodes = list(G_directed.nodes())
_deg_vals   = [deg_cent.get(n, 0.0) for n in all_nodes]
_betw_vals  = [betw.get(n, 0.0)     for n in all_nodes]
_eig_vals   = [eig.get(n, 0.0)      for n in all_nodes]
_close_vals = [close.get(n, 0.0)    for n in all_nodes]

thresh_deg   = np.percentile(_deg_vals,   HUB_PCT)
thresh_betw  = np.percentile(_betw_vals,  HUB_PCT)
thresh_eig   = np.percentile(_eig_vals,   HUB_PCT)
thresh_close = np.percentile(_close_vals, HUB_PCT)

print(f"Hub thresholds (top {100-HUB_PCT}%): "
      f"deg={thresh_deg:.4f}, betw={thresh_betw:.4f}, "
      f"eig={thresh_eig:.4f}, close={thresh_close:.4f}")


# ── 8. Per-gene metric calculation (OPTIMISED) ───────────────────────────────

print("\nPreparing lookup tables for fast per-gene computation...")

# Build PID → positions lookup
# query_name → list of (plasmid, index, domain)
pid_to_positions = defaultdict(list)

ordered_full = (
    df_merged
    .sort(['plasmid', 'start', 'ali_from'])
    .select(['query_name', 'plasmid', 'target_name'])
    .to_pandas()
)

# Track domain index while iterating plasmids
plasmid_index_counter = defaultdict(int)

for row in ordered_full.itertuples(index=False):
    plasmid = row.plasmid
    idx = plasmid_index_counter[plasmid]
    plasmid_index_counter[plasmid] += 1

    pid_to_positions[row.query_name].append((plasmid, idx, row.target_name))

print(f"Built PID lookup for {len(pid_to_positions):,} query_ids")

print("\nCalculating per-gene metrics...")

records = []

for gene_name in tqdm(all_gene_names, desc='Per-gene'):

    pids_this_gene = gene_to_pids[gene_name]
    family = gene_to_family.get(gene_name, '')

    pooled_positions = []
    pfam_domains_this_gene = set()

    # Collect positions directly from PID lookup
    for pid in pids_this_gene:
        for plasmid, idx, dom in pid_to_positions.get(pid, []):
            pooled_positions.append((plasmid, idx))
            pfam_domains_this_gene.add(dom)

    if not pooled_positions:
        records.append({
            'gene_name': gene_name, 'gene_family': family,
            'n_pfam_hits': 0, 'pfam_domains': '',
            'n_copies': 0, 'n_unique_contexts': 0,
            'raw_entropy': np.nan, 'relative_raw_entropy': np.nan,
            'dup_events': 0, 'dup_rate': np.nan,
            'degree_centrality': np.nan,
            'betweenness': np.nan, 'eigenvector': np.nan, 'closeness': np.nan,
            'is_hub_degree': False, 'is_hub_betweenness': False,
            'is_hub_eigenvector': False, 'is_hub_closeness': False,
            'is_hub_any': False,
            'in_lcc': False,
        })
        continue

    pfam_domains_this_gene = sorted(pfam_domains_this_gene)

    # ── Entropy and duplication ───────────────────────────────────────────────
    tokens = []
    dup_events = 0
    plasmids_seen = set()

    for plasmid, idx in pooled_positions:

        entries = plasmid_to_domains[plasmid]
        n = len(entries)
        if n <= 1:
            continue

        left  = entries[(idx - 1) % n]
        right = entries[(idx + 1) % n]

        tokens.append((left, right))
        plasmids_seen.add(plasmid)

        if left  in pfam_domains_this_gene or left  in beta_lac_pfam_domains:
            dup_events += 1
        if right in pfam_domains_this_gene or right in beta_lac_pfam_domains:
            dup_events += 1

    dup_rate = dup_events / len(tokens) if tokens else 0.0
    _, H, n_copies, n_unique = context_entropy_score(tokens)

    # ── Relative entropy ──────────────────────────────────────────────────────

    bg_domains = set()

    for plas in plasmids_seen:
        for d in plasmid_to_domains.get(plas, []):
            if d not in pfam_domains_this_gene:
                bg_domains.add(d)

    bg_raw = [all_domain_raw_H[d] for d in bg_domains if d in all_domain_raw_H]

    rel_raw_H = (
        round(H - np.mean(bg_raw), 6)
        if len(bg_raw) >= 3 and not np.isnan(H)
        else np.nan
    )

    # ── Centrality ────────────────────────────────────────────────────────────

    cent_deg   = max((deg_cent.get(d, 0.0)  for d in pfam_domains_this_gene
                      if d in G_directed.nodes()), default=np.nan)

    cent_betw  = max((betw.get(d, 0.0) for d in pfam_domains_this_gene
                      if d in U_lcc.nodes()), default=np.nan)

    cent_eig   = max((eig.get(d, 0.0) for d in pfam_domains_this_gene
                      if d in U_lcc.nodes()), default=np.nan)

    cent_close = max((close.get(d, 0.0) for d in pfam_domains_this_gene
                      if d in U_lcc.nodes()), default=np.nan)

    in_lcc = any(d in lcc_nodes for d in pfam_domains_this_gene)

    is_hub_deg   = (not np.isnan(cent_deg))   and cent_deg   >= thresh_deg
    is_hub_betw  = (not np.isnan(cent_betw))  and cent_betw  >= thresh_betw
    is_hub_eig   = (not np.isnan(cent_eig))   and cent_eig   >= thresh_eig
    is_hub_close = (not np.isnan(cent_close)) and cent_close >= thresh_close

    is_hub_any = is_hub_deg or is_hub_betw or is_hub_eig or is_hub_close

    records.append({
        'gene_name': gene_name,
        'gene_family': family,
        'n_pfam_hits': len(pooled_positions),
        'pfam_domains': ';'.join(pfam_domains_this_gene),
        'n_copies': n_copies,
        'n_unique_contexts': n_unique,
        'raw_entropy': round(H,6) if not np.isnan(H) else np.nan,
        'relative_raw_entropy': rel_raw_H,
        'dup_events': dup_events,
        'dup_rate': round(dup_rate,6),
        'degree_centrality': round(cent_deg,6) if not np.isnan(cent_deg) else np.nan,
        'betweenness': round(cent_betw,6) if not np.isnan(cent_betw) else np.nan,
        'eigenvector': round(cent_eig,6) if not np.isnan(cent_eig) else np.nan,
        'closeness': round(cent_close,6) if not np.isnan(cent_close) else np.nan,
        'is_hub_degree': is_hub_deg,
        'is_hub_betweenness': is_hub_betw,
        'is_hub_eigenvector': is_hub_eig,
        'is_hub_closeness': is_hub_close,
        'is_hub_any': is_hub_any,
        'in_lcc': in_lcc
    })

df_results = (
    pd.DataFrame(records)
    .sort_values('raw_entropy', ascending=False)
    .reset_index(drop=True)
)

# ── 9. Print summary and save ─────────────────────────────────────────────────

out_dir = Path('bl_gene_metrics')


print(f"\n── Results summary (n={len(df_results)} genes) ──")
print(f"Genes with raw_entropy computed  : {df_results['raw_entropy'].notna().sum()}")
print(f"Genes with rel. entropy computed : {df_results['relative_raw_entropy'].notna().sum()}")
print(f"Genes in LCC                     : {df_results['in_lcc'].sum()}")
print(f"Genes flagged as hub (any metric): {df_results['is_hub_any'].sum()}")

print(f"\nTop 20 by raw entropy:")
print(df_results[['gene_name','gene_family','n_copies','raw_entropy',
                  'relative_raw_entropy','dup_rate',
                  'betweenness','eigenvector','closeness']].head(20).to_string(index=False))

df_results.to_csv(out_dir / 'per_gene_bl_metrics.csv', index=False)
print(f"\nSaved to {out_dir / 'per_gene_bl_metrics.csv'}")










#!!! AT THIS POINT









import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# ── Paths ───────────────────────────────────────────────────────────────
metrics_csv = Path('bl_gene_metrics/per_gene_bl_metrics.csv')
mge_csv     = Path('mge_association_results/per_gene_mge_association.csv')
out_dir     = Path('bl_gene_correlation_plots')
out_dir.mkdir(exist_ok=True)

# ── Load and merge ──────────────────────────────────────────────────────
df_metrics = pd.read_csv(metrics_csv)
df_mge     = pd.read_csv(mge_csv)

if 'label' in df_mge.columns and 'gene_name' not in df_mge.columns:
    df_mge = df_mge.rename(columns={'label': 'gene_name'})

df = df_metrics.merge(
    df_mge[['gene_name', 'A_rate', 'B_rate', 'pool_rate']],
    on='gene_name', how='inner'
)
print(f"Merged dataset: {len(df)} genes")

# ── Metrics setup ──────────────────────────────────────────────────────
METRICS = [
    'raw_entropy', 'relative_raw_entropy', 'dup_rate',
    'degree_centrality', 'betweenness', 'eigenvector', 'closeness',
    'A_rate', 'B_rate', 'pool_rate'
]

METRIC_LABELS = {
    'raw_entropy': 'Raw entropy',
    'relative_raw_entropy': 'Relative entropy',
    'dup_rate': 'Duplication rate',
    'degree_centrality': 'Degree centrality',
    'betweenness': 'Betweenness',
    'eigenvector': 'Eigenvector',
    'closeness': 'Closeness',
    'A_rate': 'MGE assoc. (nt dist)',
    'B_rate': 'MGE assoc. (adjacency)',
    'pool_rate': 'MGE assoc. (pooled)',
}

df_full = df[METRICS + ['gene_name', 'gene_family', 'n_copies']].dropna(subset=['raw_entropy', 'pool_rate'])
print(f"Genes with full metrics: {len(df_full)}")

# ── Compute Spearman correlations ───────────────────────────────────────
n = len(METRICS)
rho_mat = np.full((n, n), np.nan)
p_mat   = np.full((n, n), np.nan)

for i, j in combinations(range(n), 2):
    x, y = df_full[METRICS[i]], df_full[METRICS[j]]
    mask = ~(x.isna() | y.isna())
    if mask.sum() >= 5:
        r, p = spearmanr(x[mask], y[mask])
        rho_mat[i, j] = rho_mat[j, i] = r
        p_mat[i, j] = p_mat[j, i] = p

np.fill_diagonal(rho_mat, 1)
np.fill_diagonal(p_mat, 0)

# BH correction
off_diag_idx = np.triu_indices(n, k=1)
p_vals = p_mat[off_diag_idx]
_, p_adj, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')

adj_p_mat = np.full_like(p_mat, np.nan)
adj_p_mat[off_diag_idx] = p_adj
adj_p_mat = adj_p_mat + adj_p_mat.T
np.fill_diagonal(adj_p_mat, 0)

# ── Plot 1: Correlation heatmap ─────────────────────────────────────────
plt.figure(figsize=(10, 8))
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
sns.heatmap(rho_mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            xticklabels=[METRIC_LABELS[m] for m in METRICS],
            yticklabels=[METRIC_LABELS[m] for m in METRICS],
            cbar_kws={'label': "Spearman ρ"}, square=True)
plt.title(f"Spearman correlation (BH-corrected p < 0.05)", fontsize=12)
plt.xticks(rotation=40, ha='right')
plt.tight_layout()
plt.savefig(out_dir / 'spearman_heatmap.png', dpi=200)
plt.savefig(out_dir / 'spearman_heatmap.pdf', dpi=200)
plt.close()
print("Saved spearman heatmap.")

# ── Identify significant pairs for scatter plots ────────────────────────
sig_pairs = [(i, j) for i, j in combinations(range(n), 2) if adj_p_mat[i, j] < 0.05]
sig_pairs.sort(key=lambda x: abs(rho_mat[x]), reverse=True)
print(f"Significant pairs: {len(sig_pairs)}")

# ── Plot 2: Scatter plots for top significant pairs ────────────────────
TOP_N = min(12, len(sig_pairs))
if TOP_N > 0:
    ncols = 3
    nrows = int(np.ceil(TOP_N / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))
    axes = axes.flatten()

    families = df_full['gene_family'].fillna('Unknown').unique()
    palette = dict(zip(families, sns.color_palette('tab10', len(families))))

    for ax_idx, (i, j) in enumerate(sig_pairs[:TOP_N]):
        ax = axes[ax_idx]
        xcol, ycol = METRICS[i], METRICS[j]
        sub = df_full[[xcol, ycol, 'gene_family', 'n_copies']].dropna()

        for fam, grp in sub.groupby('gene_family'):
            ax.scatter(grp[xcol], grp[ycol],
                       s=grp['n_copies'].clip(upper=500)/5+10,
                       alpha=0.7, label=fam, color=palette.get(fam, '#888888'))

        # Trend line
        m, b = np.polyfit(sub[xcol], sub[ycol], 1)
        xs = np.linspace(sub[xcol].min(), sub[xcol].max(), 200)
        ax.plot(xs, m*xs+b, color='black', lw=1.2, ls='--')

        r_val, p_val = spearmanr(sub[xcol], sub[ycol])
        ax.set_xlabel(METRIC_LABELS[xcol])
        ax.set_ylabel(METRIC_LABELS[ycol])
        ax.set_title(f'ρ={r_val:.2f}, p={p_val:.3g}')

    # Hide unused axes
    for ax in axes[TOP_N:]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.savefig(out_dir / 'scatter_top_pairs.png', dpi=200)
    plt.savefig(out_dir / 'scatter_top_pairs.pdf', dpi=200)
    plt.close()
    print("Saved scatter plots for top significant pairs.")

# ── Save correlation table ─────────────────────────────────────────────
corr_list = []
for i, j in combinations(range(n), 2):
    corr_list.append({
        'metric_1': METRICS[i],
        'metric_2': METRICS[j],
        'rho': rho_mat[i, j],
        'p_raw': p_mat[i, j],
        'p_adj_BH': adj_p_mat[i, j],
        'significant': adj_p_mat[i, j] < 0.05
    })

pd.DataFrame(corr_list).to_csv(out_dir / 'spearman_correlations.csv', index=False)
print("Saved spearman_correlations.csv")


################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
## ============================================================
## MGE ASSOCIATION ANALYSIS FOR BETA-LACTAMASES
## Two strategies:
##   A) Nucleotide distance between beta-lactamase and MGE domains (via PID indices)
##   B) Domain adjacency: direct neighbour or sandwiched (MGE|x|BL|y|MGE) on plasmid
## Comparisons:
##   - BL overall vs non-BL/non-MGE "background" genes
##   - Per beta-lactamase gene name (from AMRFinder mapping)
## ============================================================

import os
import re
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ── 0. Load df_merged ────────────────────────────────────────────────────────

data_dir = Path(os.path.join(os.getcwd(), 'plasmid_motif_network/intermediate'))
files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))
df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
df_merged = df_merged.with_columns(
    pl.col('strand').cast(pl.Int32).alias('strand')
)

# ── 1. MGE detection machinery ───────────────────────────────────────────────

MGE_PFAM_ACCESSIONS = frozenset({
    'PF00665','PF01527','PF01609','PF01610','PF02003','PF02022','PF02316',
    'PF02371','PF00239','PF07508','PF01764','PF04754','PF05699','PF10551',
    'PF12761','PF13385','PF13586','PF13840','PF00872','PF01398','PF02914',
    'PF03050','PF03184','PF04827','PF07592','PF09184','PF09811','PF10407',
    'PF12728','PF13843','PF14815','PF15706','PF17921','PF00589','PF09424',
    'PF02899','PF13102','PF02902','PF07022','PF01371','PF13612','PF03108',
    'PF01797','PF06276','PF00078','PF00552','PF07727','PF00075','PF05986',
    'PF03354','PF04364','PF04589','PF07195','PF09068','PF05135','PF10145',
    'PF06143','PF01424','PF03389','PF08751','PF03004','PF13009',
})

MGE_TARGET_NAMES_EXACT = frozenset({
    'Transposase_1','Transposase_2','Transposase_IS200_or_IS605','Transposase_mut',
    'Transposase_21','DDE_Tnp_IS1','DDE_Tnp_IS240','DDE_Tnp_1','DDE_Tnp_4',
    'DDE_Tnp_ISAZ013','DDE_3','IS1_InsA','IS1_InsB_1','IS3_IS911','IS30_Tnp',
    'IS66','IS66_Orf2','IS66_Orf3','IS200_IS605','IS701','InsB','IS3_transposase',
    'IS481','ISTron','Resolvase','Recombinase','Phage_integrase',
    'Integrase_recombinase_phage','Integrase','Int_C','Int_AP2','Integrase_Zn',
    'rve','rve_2','rve_3','RVT_1','RVT_2','RVT_3','RVT_N','RVT_thumb',
    'RNase_H','RNase_H2','Phage_terminase_1','Phage_terminase_2','Terminase_6',
    'Phage_cap_E','Phage_pRha','Phage_Mu_Gam','Phage_GPA','GPW_gp25',
    'Relaxase','MobA_MobL','MobA','MOB_NHLP','Xis','RDF','Tn3_res','TnpR',
    'HTH_Tnp_IS630','HTH_Tnp_Tc3_2','HTH_Tnp_1','HTH_Tnp_Mu_2','HTH_Tnp_Mu_1',
    'Tnp_zf-ribbon_2','MULE','FLINT','Zn_Tnp_IS1',
})

_MGE_REGEX_RAW = [
    r'transpos', r'\bIS\d', r'\bTn\d', r'\bICE\b',
    r'integrase', r'resolvase', r'invertase(?!.*sugar)', r'recombinas',
    r'retrotranspos', r'retroelem', r'retrovir', r'reverse.transcriptas',
    r'RVT', r'RNase_H', r'DDE.tnp', r'DDE_Tnp', r'Tnp[AB_]',
    r'phage.*integras', r'phage.*capsid', r'phage.*terminase',
    r'phage.*portal', r'phage.*tail', r'phage.*baseplate', r'phage.*sheath',
    r'mob[A-Z_]', r'relaxase', r'mobilisa', r'\bXis\b',
    r'recombination.directionality', r'RDF\b', r'insertion.seq',
    r'insertion.element', r'Tn3_res', r'MULE', r'HTH_Tnp', r'Zn_Tnp',
    r'INE\d', r'ISCR', r'IS_Mu',
]
MGE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _MGE_REGEX_RAW]

def is_mge_domain(target_name: str, target_accession: str) -> bool:
    acc_base = str(target_accession).split('.')[0]
    if acc_base in MGE_PFAM_ACCESSIONS:
        return True
    if str(target_name) in MGE_TARGET_NAMES_EXACT:
        return True
    name = str(target_name)
    for pat in MGE_PATTERNS:
        if pat.search(name):
            return True
    return False

def is_betalactamase_domain(target_name: str) -> bool:
    return 'lactamase' in str(target_name).lower()


# ── 2. Classify all rows once ────────────────────────────────────────────────

df_pd = df_merged.to_pandas()

mge_mask   = df_pd.apply(
    lambda row: is_mge_domain(row['target_name'], row['target_accession']), axis=1
)
bl_mask    = df_pd['target_name'].apply(is_betalactamase_domain)
other_mask = ~mge_mask & ~bl_mask

df_mge   = df_pd[mge_mask].copy()
df_bl    = df_pd[bl_mask].copy()
df_other = df_pd[other_mask].copy()

print(f"MGE domain rows        : {len(df_mge):,}")
print(f"Beta-lactamase rows    : {len(df_bl):,}")
print(f"Other (background) rows: {len(df_other):,}")


# ── 3. Load AMRFinder per-gene mapping ───────────────────────────────────────

MERGED_FASTA_DIR = Path('merged_nonoverlapping_fastas')
merged_kept_PIDs = set('.'.join(x.split('.')[:-1]) for x in os.listdir(MERGED_FASTA_DIR))

test = pd.read_csv('amrfindermapped_beta_lactamases.csv', low_memory=False)
test = test.loc[test['query_id'].isin(merged_kept_PIDs)].copy()

pid_to_gene   = dict(zip(test['query_id'], test['gene_name']))
pid_to_family = dict(zip(test['query_id'], test['gene_family']))

all_gene_names = [x for x in test['gene_name'].unique() if isinstance(x, str)]
print(f"Unique BL gene names   : {len(all_gene_names):,}")

gene_to_family = {
    gname: test.loc[test['gene_name'] == gname, 'gene_family'].iloc[0]
    for gname in all_gene_names
}

gene_to_qnames = defaultdict(set)
for qid, gname in pid_to_gene.items():
    if isinstance(gname, str):
        gene_to_qnames[gname].add(qid)


# ── 4. Shared infrastructure ──────────────────────────────────────────────────

DIST_THRESHOLD = 5000   # nucleotides  #$$$
CIRCULAR       = True   # plasmids are always circular
SANDWICH_GAP   = 1      # max intervening domains on each side  #$$$

# mge_flat: one row per MGE domain hit, used by Strategy A merges
mge_flat = df_mge[['plasmid', 'start', 'stop']].copy()
mge_flat.columns = ['plasmid', 'mge_start', 'mge_stop']
mge_flat = mge_flat.reset_index(drop=True)

# gene_tags: one row per (plasmid, query_name), with is_mge / is_bl flags
df_pd['is_mge'] = mge_mask.values
df_pd['is_bl']  = bl_mask.values
gene_tags = (
    df_pd.groupby(['plasmid', 'query_name', 'start', 'stop'])
    .agg(is_mge=('is_mge', 'any'), is_bl=('is_bl', 'any'))
    .reset_index()
)
gene_tags = gene_tags.sort_values(['plasmid', 'start']).reset_index(drop=True)

# Per-plasmid arrays for fast Strategy B access
plasmid_gene_data = {}
for plasmid, grp in gene_tags.groupby('plasmid', sort=False):
    grp = grp.reset_index(drop=True)
    plasmid_gene_data[plasmid] = {
        'query_names': grp['query_name'].tolist(),
        'starts'     : grp['start'].tolist(),
        'stops'      : grp['stop'].tolist(),
        'mge'        : grp['is_mge'].to_numpy(),
    }


# ── 5. Strategy A helper (Memory-Safe & Vectorised) ──────────────────────────

def run_strategy_A(df_query: pd.DataFrame) -> pd.DataFrame:
    """
    Safely scores genes by chunking plasmids to prevent cross-join memory explosions.
    """
    q = df_query[['query_name', 'plasmid', 'start', 'stop']].copy()
    q = q.rename(columns={'start': 'bl_start', 'stop': 'bl_stop'})
    
    plasmids_all = q['plasmid'].unique()
    chunk_results = []
    CHUNK_SIZE = 5000 # Number of plasmids to process at once
    
    # Process in chunks to prevent memory blow-outs
    for start_idx in tqdm(range(0, len(plasmids_all), CHUNK_SIZE), desc='Strategy A processing', leave=False):
        plas_chunk = plasmids_all[start_idx : start_idx + CHUNK_SIZE]
        
        q_chunk = q[q['plasmid'].isin(plas_chunk)]
        mge_chunk = mge_flat[mge_flat['plasmid'].isin(plas_chunk)]
        
        merged = q_chunk.merge(mge_chunk, on='plasmid', how='left')
        
        # Vectorised interval gap
        merged['dist'] = np.maximum(
            0,
            np.maximum(
                merged['mge_start'].values - merged['bl_stop'].values,
                merged['bl_start'].values  - merged['mge_stop'].values,
            )
        )
        merged['dist'] = merged['dist'].fillna(np.inf)
        
        # Group immediately to shrink size before moving to next chunk
        chunk_res = (
            merged.groupby(['query_name', 'plasmid', 'bl_start', 'bl_stop'])['dist']
            .min()
            .reset_index()
            .rename(columns={'dist': 'min_dist_nt', 'bl_start': 'start', 'bl_stop': 'stop'})
        )
        chunk_results.append(chunk_res)

    result = pd.concat(chunk_results, ignore_index=True) if chunk_results else pd.DataFrame()
    if not result.empty:
        result['mge_assoc_A'] = result['min_dist_nt'] <= DIST_THRESHOLD
    else:
        result['mge_assoc_A'] = False
    return result


# ── 6. Strategy B helper (Memory-Optimised) ──────────────────────────────────

def run_strategy_B(query_names_set: set, desc: str) -> pd.DataFrame:
    """
    Scores genes in query_names_set. Uses columnar lists which are massively 
    more memory-efficient than building millions of dictionaries.
    """
    # Pre-allocate column lists
    q_out, p_out, start_out, stop_out = [], [], [], []
    da_out, sw_out, assoc_out = [], [], []
    
    reach = SANDWICH_GAP + 1
    
    for plasmid, data in tqdm(plasmid_gene_data.items(), desc=desc, leave=False):
        qnames  = data['query_names']
        mge_arr = data['mge']
        n       = len(qnames)
        target_positions = [i for i, qn in enumerate(qnames) if qn in query_names_set]
        
        if not target_positions:
            continue
            
        mge_positions = set(np.where(mge_arr)[0])
        
        for i in target_positions:
            directly_adjacent = (
                ((i - 1) % n) in mge_positions or
                ((i + 1) % n) in mge_positions
            )
            left_mge_dists  = [d for d in range(1, reach + 1) if ((i - d) % n) in mge_positions]
            right_mge_dists = [d for d in range(1, reach + 1) if ((i + d) % n) in mge_positions]
            sandwiched = bool(left_mge_dists) and bool(right_mge_dists)
            mge_assoc_B = directly_adjacent or sandwiched
            
            # Append directly to flat lists (saves huge memory)
            q_out.append(qnames[i])
            p_out.append(plasmid)
            start_out.append(data['starts'][i])
            stop_out.append(data['stops'][i])
            da_out.append(directly_adjacent)
            sw_out.append(sandwiched)
            assoc_out.append(mge_assoc_B)

    return pd.DataFrame({
        'query_name': q_out,
        'plasmid': p_out,
        'gene_start': start_out,
        'gene_stop': stop_out,
        'directly_adjacent': da_out,
        'sandwiched': sw_out,
        'mge_assoc_B': assoc_out
    })


# ── 7. Summarise helpers ──────────────────────────────────────────────────────

def summarise(df_A: pd.DataFrame, df_B: pd.DataFrame, label: str) -> dict:
    """For BL overall, per-gene calls, and background."""
    n_A    = len(df_A)
    asc_A  = int(df_A['mge_assoc_A'].sum()) if n_A > 0 else 0
    rate_A = asc_A / n_A if n_A > 0 else np.nan

    n_B    = len(df_B)
    asc_B  = int(df_B['mge_assoc_B'].sum()) if n_B > 0 else 0
    rate_B = asc_B / n_B if n_B > 0 else np.nan

    qn_A   = set(df_A.loc[df_A['mge_assoc_A'], 'query_name']) if n_A > 0 else set()
    qn_B   = set(df_B.loc[df_B['mge_assoc_B'], 'query_name']) if n_B > 0 else set()
    all_qn = (set(df_A['query_name']) if n_A > 0 else set()) | \
             (set(df_B['query_name']) if n_B > 0 else set())
    pooled = qn_A | qn_B
    n_pt   = len(all_qn)
    n_pa   = len(pooled)
    rate_p = n_pa / n_pt if n_pt > 0 else np.nan

    print(f"\n{'─'*58}")
    print(f"  {label}")
    print(f"{'─'*58}")
    print(f"  Strategy A (nt dist ≤ {DIST_THRESHOLD:,} nt) : "
          f"{asc_A:,} / {n_A:,}  ({rate_A*100:.2f}%)")
    print(f"  Strategy B (adjacency, gap ≤ {SANDWICH_GAP}) : "
          f"{asc_B:,} / {n_B:,}  ({rate_B*100:.2f}%)")
    if n_B > 0:
        print(f"    direct adjacent : {int(df_B['directly_adjacent'].sum()):,}")
        print(f"    sandwiched only : "
              f"{int((df_B['sandwiched'] & ~df_B['directly_adjacent']).sum()):,}")
    print(f"  Pooled (unique genes, either) : "
          f"{n_pa:,} / {n_pt:,}  ({rate_p*100:.2f}%)")

    return dict(
        label=label,
        A_total=n_A,  A_assoc=asc_A,  A_rate=rate_A,
        B_total=n_B,  B_assoc=asc_B,  B_rate=rate_B,
        pool_total=n_pt, pool_assoc=n_pa, pool_rate=rate_p,
    )


# ── 8. BL overall ─────────────────────────────────────────────────────────────

print("\nRunning BL overall...")
df_A_bl  = run_strategy_A(df_bl)
df_B_bl  = run_strategy_B(set(df_bl['query_name']), desc='BL overall — Strategy B')
stats_bl = summarise(df_A_bl, df_B_bl, 'Beta-lactamases overall (pfam)')


# ── 9. Non-BL / non-MGE background ───────────────────────────────────────────

print("\nRunning Background...")
df_A_other = run_strategy_A(df_other)
df_B_other = run_strategy_B(set(df_other['query_name']), desc='Background — Strategy B')
stats_other = summarise(df_A_other, df_B_other, 'Non-BL / non-MGE background (pfam)')


# ── 10. Per-gene (AMRFinder gene names) ───────────────────────────────────────

per_gene_stats = []

for gene_name in tqdm(all_gene_names, desc='Per-gene loop'):
    qnames_this = gene_to_qnames[gene_name]
    df_bl_gene  = df_bl.loc[df_bl['query_name'].isin(qnames_this)]

    if len(df_bl_gene) == 0:
        per_gene_stats.append(dict(
            gene_name=gene_name, gene_family=gene_to_family.get(gene_name, ''),
            A_total=0, A_assoc=0, A_rate=np.nan,
            B_total=0, B_assoc=0, B_rate=np.nan,
            pool_total=0, pool_assoc=0, pool_rate=np.nan,
        ))
        continue

    df_A_gene = run_strategy_A(df_bl_gene)
    df_B_gene = run_strategy_B(qnames_this, desc=gene_name)

    s = summarise(df_A_gene, df_B_gene, gene_name)
    s['gene_name']   = gene_name
    s['gene_family'] = gene_to_family.get(gene_name, '')
    per_gene_stats.append(s)

df_per_gene = (
    pd.DataFrame(per_gene_stats)
    .sort_values('pool_rate', ascending=False)
    .reset_index(drop=True)
)

print(f"\n── Per-gene summary (top 20 by pooled MGE association rate) ──")
print(df_per_gene[['gene_name','gene_family','A_rate','B_rate',
                    'pool_rate','pool_assoc','pool_total']].head(20).to_string(index=False))

df_per_gene = df_per_gene.loc[df_per_gene['label'].isin(list(set(test['gene_name'].tolist())))]


# ── 11. Save all results ──────────────────────────────────────────────────────

output_dir = Path(os.path.join(os.getcwd(), 'mge_association_results'))
output_dir.mkdir(exist_ok=True)

df_A_bl.to_csv( output_dir / 'BL_overall_strategy_A.csv', index=False)
df_B_bl.to_csv( output_dir / 'BL_overall_strategy_B.csv', index=False)
df_A_other.to_csv( output_dir / 'other_genes_strategy_A.csv', index=False)
df_B_other.to_csv( output_dir / 'other_genes_strategy_B.csv', index=False)
df_per_gene.to_csv(output_dir / 'per_gene_mge_association.csv', index=False)

df_summary = pd.DataFrame([stats_bl, stats_other])[
    ['label','A_total','A_assoc','A_rate',
     'B_total','B_assoc','B_rate',
     'pool_total','pool_assoc','pool_rate']
]

df_summary.to_csv(output_dir / 'group_comparison_summary.csv', index=False)

print(f"\nAll results written to {output_dir}")
print(df_summary.to_string(index=False))




################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################


## ============================================================
#
#ALL PROTEIN METRICS
#
## ============================================================

import os
import re
import math
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from itertools import combinations


# ─── patterns ────────────────────────────────────────────────────────────────
PID_nuccore_pattern = re.compile(r'^(.+?)_\d+_\d+')
PID_nogene_pattern  = re.compile(r'^(.+?)_(\d+)_(\d+)$')
MIN_OBS = 5

def pid_to_plasmid(pid):
    m = PID_nuccore_pattern.match(pid)
    return m.group(1) if m else None

# ─── PID lists ───────────────────────────────────────────────────────────────
FASTA_DIR        = Path('fastas')
PFAM_FASTA_DIR   = Path('pfam_fastas')
MERGED_FASTA_DIR = Path('merged_nonoverlapping_fastas')

merged_kept_PIDs = ['.'.join(x.split('.')[:-1]) for x in os.listdir(MERGED_FASTA_DIR)]
kept_pfam_PIDs   = [p for p in merged_kept_PIDs if PID_nogene_pattern.match(p)]
kept_fasta_PIDs  = [p for p in merged_kept_PIDs if not PID_nogene_pattern.match(p)]
total_PIDs = len(merged_kept_PIDs)

# ─── beta-lactamase hits — now using _final CSV ──────────────────────────────
test = pd.read_csv('amrfindermapped_beta_lactamases.csv', low_memory=False)
test = test.loc[test['query_id'].isin(merged_kept_PIDs)]
all_betas = [x for x in test['gene_name'].unique() if isinstance(x, str)]

betas_to_PIDs       = {}
betas_to_plas       = {}
pfam_betas_to_PIDs  = {}
fasta_betas_to_PIDs = {}

for gene in all_betas:
    queries = list(set(test.loc[test['gene_name'] == gene, 'query_id']))
    betas_to_PIDs[gene]       = queries
    pfam_betas_to_PIDs[gene]  = [x for x in queries if PID_nogene_pattern.match(x)]
    fasta_betas_to_PIDs[gene] = [x for x in queries if not PID_nogene_pattern.match(x)]
    betas_to_plas[gene]       = list(set(
        m.group(1) for x in queries if (m := PID_nuccore_pattern.match(x))
    ))

# ─── CARD prevalence (informational) ─────────────────────────────────────────
card_prevalence = pd.read_csv('card_prevalence.txt', sep='\t')
(card_prevalence.loc[card_prevalence['Name'].isin(all_betas)]
                .sort_values('NCBI Plasmid', ascending=False)
                .to_csv('card_prevalence_betas_redone_final.txt'))

prev_rows = []
for gene in all_betas:
    n_pids  = len(betas_to_PIDs.get(gene, []))
    n_plas  = len(betas_to_plas.get(gene, []))
    prev_rows.append({
        'gene':            gene,
        'n_PIDs':          n_pids,
        'pct_of_all_PIDs': round(n_pids / total_PIDs, 6),
        'n_plasmids':      n_plas,
    })

prev_df = pd.DataFrame(prev_rows).sort_values('n_PIDs', ascending=False)

card_sub = (card_prevalence.loc[card_prevalence['Name'].isin(all_betas),
                                ['Name', 'NCBI Plasmid', 'NCBI Chromosome']]
                            .rename(columns={'Name': 'gene',
                                             'NCBI Plasmid':    'card_ncbi_plasmid',
                                             'NCBI Chromosome': 'card_ncbi_chrom'}))

prev_df = prev_df.merge(card_sub, on='gene', how='left')
prev_df.to_csv('beta_lactamase_prevalence_final.csv', index=False)
print(prev_df.head(30).to_string())


# ─── PLSDB metadata ──────────────────────────────────────────────────────────
plsdb_meta_path = Path('plsdb_meta')

nuc_df  = pd.read_csv(plsdb_meta_path / 'nuccore_only.csv')
typ_df  = pd.read_csv(plsdb_meta_path / 'typing_only.csv')
bio_df  = pd.read_csv(plsdb_meta_path / 'biosample.csv', low_memory=False)
tax_df  = pd.read_csv(plsdb_meta_path / 'taxonomy.csv')

nuc_tax = dict(zip(nuc_df['NUCCORE_ACC'], nuc_df['TAXONOMY_UID']))
nuc_bio = dict(zip(nuc_df['NUCCORE_ACC'], nuc_df['BIOSAMPLE_UID']))
nuc_mob = dict(zip(typ_df['NUCCORE_ACC'], typ_df['predicted_mobility']))
tax_spc = dict(zip(tax_df['TAXONOMY_UID'], tax_df['TAXONOMY_species']))
nuc_spc = {k: tax_spc.get(v) for k, v in nuc_tax.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — MOBILITY
# ═══════════════════════════════════════════════════════════════════════════════
print('\n--- MOBILITY ---')

categories = ['conjugative', 'mobilizable', 'non-mobilizable']

mob_rows = []
for gene, plasmids in betas_to_plas.items():
    for plas in set(plasmids):
        mob = nuc_mob.get(plas)
        if mob is not None:
            mob_rows.append({'gene': gene, 'mobility': mob})

gene_counts = (
    pd.DataFrame(mob_rows)
    .groupby(['gene', 'mobility']).size()
    .unstack(fill_value=0)
    .reindex(columns=categories, fill_value=0)
)
gene_counts['total'] = gene_counts.sum(axis=1)
gene_counts = gene_counts[gene_counts['total'] >= MIN_OBS]

mob_results = []
for gene, row in gene_counts.iterrows():
    total    = row['total']
    pct      = row[categories] / total
    n_mobile = int(row['conjugative']) + int(row['mobilizable'])
    mob_results.append({
        'gene':                gene,
        'n_plasmids':          int(total),
        'pct_conjugative':     round(pct['conjugative'],     4),
        'pct_mobilizable':     round(pct['mobilizable'],     4),
        'pct_non_mobilizable': round(pct['non-mobilizable'], 4),
        'pct_mobile':          round(n_mobile / total,       4),
    })

mob_df = pd.DataFrame(mob_results)

gene_conj_obs = {
    r['gene']: [1] * int(r['n_plasmids'] * r['pct_conjugative'])
             + [0] * (r['n_plasmids'] - int(r['n_plasmids'] * r['pct_conjugative']))
    for _, r in mob_df.iterrows() if r['n_plasmids'] >= MIN_OBS
}
med_val  = mob_df['pct_conjugative'].median()
med_gene = (mob_df['pct_conjugative'] - med_val).abs().idxmin()
med_gene = mob_df.loc[med_gene, 'gene']
ref_conj = gene_conj_obs[med_gene]

mw_mob = []
for gene, obs in gene_conj_obs.items():
    if gene == med_gene or len(obs) < MIN_OBS:
        mw_mob.append({'gene': gene, 'p_mw_mobility': np.nan})
        continue
    _, p = mannwhitneyu(obs, ref_conj, alternative='two-sided')
    mw_mob.append({'gene': gene, 'p_mw_mobility': p})

mw_mob_df = pd.DataFrame(mw_mob)
valid = mw_mob_df['p_mw_mobility'].notna()
if valid.sum() > 1:
    mw_mob_df.loc[valid, 'p_adj_mobility'] = multipletests(
        mw_mob_df.loc[valid, 'p_mw_mobility'], method='fdr_bh'
    )[1]

mob_df = (mob_df.merge(mw_mob_df[['gene', 'p_adj_mobility']], on='gene', how='left')
                .sort_values('pct_mobile', ascending=False))
mob_df.to_csv('beta_lactamase_mobility_stats_final.csv', index=False)
print(mob_df.to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — SPECIES BREADTH
# ═══════════════════════════════════════════════════════════════════════════════
print('\n--- SPECIES BREADTH ---')

betas_to_species = {}
for gene, PIDs in betas_to_PIDs.items():
    plas_list    = [pid_to_plasmid(x) for x in PIDs]
    species_list = [nuc_spc.get(p) for p in plas_list if p]
    betas_to_species[gene] = [s for s in species_list if s is not None]

spc_results = []
for gene in all_betas:
    slist = betas_to_species.get(gene, [])
    if len(slist) < MIN_OBS:
        continue
    n_plas  = len(slist)
    counts  = Counter(slist)
    n_spc   = len(counts)
    simpson = 1.0 - sum((c / n_plas) ** 2 for c in counts.values())
    max_simp = 1.0 - 1.0 / n_spc if n_spc > 1 else 0.0
    spread   = round(simpson / max_simp, 5) if max_simp > 0 else 0.0
    spc_results.append({
        'gene':              gene,
        'simpson_diversity': round(simpson, 5),
        'spread_score':      spread,
    })

spc_df = pd.DataFrame(spc_results).sort_values('spread_score', ascending=False)
spc_df.to_csv('beta_lactamase_species_breadth_final.csv', index=False)
print(spc_df.to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — SHANNON ENTROPY AND RELATIVE ENTROPY
#
# Entropy is calculated using context_entropy_score from hub_proneness_and_entropy.py:
#   raw_entropy        = raw Shannon H over (left, right) neighbour-pair tokens
#   relative_raw_entropy = raw_entropy minus mean(raw_entropy of background domains
#                          on the same plasmids), analogous to entropy_dist.py
#
# The per-copy and dedup variants from the old script are replaced by these two
# metrics, which are more directly interpretable and consistent with the rest of
# the analysis pipeline.
#
# Flow: amrfindermapped_beta_lactamases.csv → gene_name → pfam PIDs →
#       df_merged (query_name lookup) → plasmid position → (left, right) context
#
# beta_lac_pfam_domains is now derived from df_merged directly:
#   target_name contains 'lactamase'/'Lactamase' AND query_name is in
#   amrfindermapped_beta_lactamases.csv
# This avoids the clan-based approach which includes non-BL structural homologues.
# ═══════════════════════════════════════════════════════════════════════════════
print('\n--- SHANNON ENTROPY ---')

data_dir   = Path('plasmid_motif_network/intermediate')
clust_path = Path('clustering_results/umap_hdbscan_clusters.csv')
OUT_DIR    = Path('protein_prioritisation_metrics_final')
OUT_DIR.mkdir(exist_ok=True)

files     = sorted(data_dir.glob('parsed_selected_nonoverlap_*.parquet'))
df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
df_merged = df_merged.with_columns(pl.col('strand').cast(pl.Int32))

clust_df           = pd.read_csv(clust_path)
plasmid_to_cluster = dict(zip(clust_df['plasmid'], clust_df['cluster']))

# Ordered domain list per plasmid (includes strand and start for position lookup)
ordered_df = df_merged.sort(['plasmid', 'start', 'ali_from']).select(
    ['plasmid', 'target_name', 'strand', 'start']
)

plasmid_to_domains = defaultdict(list)   # plasmid -> [(domain, strand, start), ...]
for row in ordered_df.iter_rows(named=True):
    plasmid_to_domains[row['plasmid']].append(
        (row['target_name'], row['strand'], row['start'])
    )

plasmid_starts = {
    plas: np.array([t[2] for t in entries])
    for plas, entries in plasmid_to_domains.items()
}

# ── beta_lac_pfam_domains: derived from df_merged, not from clan table ────────
# Criterion: target_name contains 'lactamase'/'Lactamase' AND query_name is
# a PID present in amrfindermapped_beta_lactamases.csv.
# This ensures only confirmed BL proteins contribute to the domain name set.
amr_pids_set = set(test['query_id'].tolist())

bl_domain_rows = (
    df_merged
    .filter(
        (pl.col('target_name').str.contains('lactamase') |
         pl.col('target_name').str.contains('Lactamase')) &
        pl.col('query_name').is_in(list(amr_pids_set))
    )
    .select(['target_name'])
    .to_pandas()
)
beta_lac_pfam_domains = set(bl_domain_rows['target_name'].unique())
print(f'Beta-lactamase Pfam domain names ({len(beta_lac_pfam_domains)}): '
      f'{sorted(beta_lac_pfam_domains)}')

COORD_TOL = 50

def resolve_pid_index(pid):
    """Return (plasmid, idx) for a PID using nearest-coordinate matching."""
    plasmid = pid_to_plasmid(pid)
    m = PID_nogene_pattern.match(pid)
    if not plasmid or not m:
        return None, None
    pid_start = int(m.group(2))
    starts = plasmid_starts.get(plasmid)
    if starts is None or len(starts) == 0:
        return plasmid, None
    idx = int(np.argmin(np.abs(starts - pid_start)))
    if abs(starts[idx] - pid_start) > COORD_TOL:
        pid_stop = int(m.group(3))
        idx2 = int(np.argmin(np.abs(starts - pid_stop)))
        if abs(starts[idx2] - pid_stop) > COORD_TOL:
            return plasmid, None
        idx = idx2
    return plasmid, idx

def get_neighbours_by_index(plasmid, idx):
    entries = plasmid_to_domains.get(plasmid, [])
    n = len(entries)
    if n <= 1:
        return None, None
    left  = entries[(idx - 1) % n][0]
    right = entries[(idx + 1) % n][0]
    return left, right


# ── Entropy function (from hub_proneness_and_entropy.py) ─────────────────────
def context_entropy_score(contexts):
    """
    contexts : list of (left_domain, right_domain) tuples
    Returns  : (penalised_score, raw_H, N, K)
      raw_H  = raw Shannon entropy (bits) — this is what we report
      N      = total observations (copies)
      K      = unique context types
    The penalised_score is retained for completeness but not used in outputs.
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


# ── domain_positions: index for background entropy computation ────────────────
domain_positions = defaultdict(list)   # domain -> [(plasmid, idx), ...]
for plasmid, entries in plasmid_to_domains.items():
    n = len(entries)
    if n <= 1:
        continue
    for i, (dom, _, _) in enumerate(entries):
        domain_positions[dom].append((plasmid, i))

# ── Background cache: raw_entropy per domain (all plasmids) ──────────────────
# Used for relative_raw_entropy: focal gene H minus mean background H.
print('Building background domain raw entropies...')

all_domain_raw_H = {}   # domain -> raw H (only if >= MIN_OBS copies)

for dom, pos_list in tqdm(domain_positions.items(), desc='Background entropy'):
    tokens = []
    for plasmid, idx in pos_list:
        entries = plasmid_to_domains[plasmid]
        n = len(entries)
        if n <= 1:
            continue
        left  = entries[(idx - 1) % n][0]
        right = entries[(idx + 1) % n][0]
        tokens.append((left, right))
    _, H, n_copies, _ = context_entropy_score(tokens)
    if n_copies >= MIN_OBS and not np.isnan(H):
        all_domain_raw_H[dom] = H

print(f'  {len(all_domain_raw_H):,} domains with >= {MIN_OBS} raw observations.')

# ── resolve all pfam-source BL PIDs to positions ──────────────────────────────
pfam_test    = test.loc[test['query_id'].isin(kept_pfam_PIDs)]
pid_to_index = {}
for qid in pfam_test['query_id']:
    plasmid, idx = resolve_pid_index(qid)
    if idx is not None:
        pid_to_index[qid] = (plasmid, idx)

found    = len(pid_to_index)
notfound = len(pfam_test) - found
print(f'PID position resolved: {found}  unresolved: {notfound}')


# ── Fast PID → position lookup (from hub_proneness_and_entropy.py) ────────────
print('\nBuilding fast PID → position lookup...')

pid_to_positions = defaultdict(list)   # query_name -> [(plasmid, idx, domain), ...]

ordered_full = (
    df_merged
    .sort(['plasmid', 'start', 'ali_from'])
    .select(['query_name', 'plasmid', 'target_name'])
    .to_pandas()
)

plasmid_index_counter = defaultdict(int)
for row in ordered_full.itertuples(index=False):
    plasmid = row.plasmid
    idx = plasmid_index_counter[plasmid]
    plasmid_index_counter[plasmid] += 1
    pid_to_positions[row.query_name].append((plasmid, idx, row.target_name))

print(f'Built PID lookup for {len(pid_to_positions):,} query_ids')


# ── Per-gene entropy, relative entropy, duplication rate ──────────────────────
print('\nCalculating per-gene entropy metrics...')

gene_to_pids_map = defaultdict(set)
gene_to_family   = {}
for _, row in test.iterrows():
    if isinstance(row['gene_name'], str):
        gene_to_pids_map[row['gene_name']].add(row['query_id'])
        gene_to_family[row['gene_name']] = row.get('gene_family', '')

all_gene_names = list(gene_to_pids_map.keys())

overall_records = []

for gene_name in tqdm(all_gene_names, desc='Per-gene entropy'):
    pids_this_gene = gene_to_pids_map[gene_name]
    # Only pfam PIDs (have coordinate info for position lookup)
    pids_pfam = [p for p in pids_this_gene if PID_nogene_pattern.match(p)]

    pooled_positions    = []
    pfam_domains_this_gene = set()

    for pid in pids_pfam:
        for plasmid, idx, dom in pid_to_positions.get(pid, []):
            pooled_positions.append((plasmid, idx))
            pfam_domains_this_gene.add(dom)

    if not pooled_positions:
        continue

    pfam_domains_this_gene = sorted(pfam_domains_this_gene)
    n_dedup = len(pooled_positions)
    if n_dedup < MIN_OBS:
        continue

    # Collect (left, right) context tokens and duplication events
    tokens        = []
    dup_events    = 0
    plasmids_seen = set()

    for plasmid, idx in pooled_positions:
        entries = plasmid_to_domains[plasmid]
        n = len(entries)
        if n <= 1:
            continue
        left  = entries[(idx - 1) % n][0]
        right = entries[(idx + 1) % n][0]
        tokens.append((left, right))
        plasmids_seen.add(plasmid)
        # Duplication: adjacent domain is a BL domain (same or any other BL)
        if left  in pfam_domains_this_gene or left  in beta_lac_pfam_domains:
            dup_events += 1
        if right in pfam_domains_this_gene or right in beta_lac_pfam_domains:
            dup_events += 1

    dup_rate = dup_events / len(tokens) if tokens else 0.0
    _, H, n_copies, n_unique = context_entropy_score(tokens)

    if np.isnan(H):
        continue

    # Focal pfam domain names for background exclusion
    focal_domains = set(pfam_domains_this_gene)

    # Background: all domains on the same plasmids, excluding focal gene's domains
    bg_domains = set()
    for plas in plasmids_seen:
        for dom, _, _ in plasmid_to_domains.get(plas, []):
            if dom not in focal_domains:
                bg_domains.add(dom)

    bg_raw = [all_domain_raw_H[d] for d in bg_domains if d in all_domain_raw_H]
    relative_raw_entropy = (
        round(H - float(np.mean(bg_raw)), 5)
        if len(bg_raw) >= 3 else np.nan
    )

    overall_records.append({
        'gene':                  gene_name,
        'gene_family':           gene_to_family.get(gene_name, ''),
        'pfam_domains':          ';'.join(pfam_domains_this_gene),
        'n_copies':              n_copies,
        'n_unique_contexts':     n_unique,
        'raw_entropy':           round(H, 6),
        'relative_raw_entropy':  relative_raw_entropy,
        'dup_events':            dup_events,
        'dup_rate':              round(dup_rate, 6),
        'n_bg_domains':          len(bg_raw),
    })

overall_df = pd.DataFrame(overall_records).sort_values('raw_entropy', ascending=False)
overall_df.to_csv(OUT_DIR / 'recombination_overall_final.csv', index=False)
overall_df.to_csv(OUT_DIR / 'recombination_summary_final.csv', index=False)
print(overall_df.to_string())
print('\nDone.')


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS — recombination/entropy metrics
# ═══════════════════════════════════════════════════════════════════════════════

PLOT_DIR = Path('metric_plots_final')
PLOT_DIR.mkdir(exist_ok=True)

prev_df_plot   = pd.read_csv('beta_lactamase_prevalence_final.csv')
mob_df_plot    = pd.read_csv('beta_lactamase_mobility_stats_final.csv')
spc_df_plot    = pd.read_csv('beta_lactamase_species_breadth_final.csv')
recomb_df_plot = pd.read_csv(OUT_DIR / 'recombination_summary_final.csv')

master = (recomb_df_plot
    .merge(mob_df_plot[['gene', 'pct_mobile', 'pct_conjugative', 'p_adj_mobility']],
           on='gene', how='left')
    .merge(spc_df_plot[['gene', 'simpson_diversity', 'spread_score']],
           on='gene', how='left')
    .merge(prev_df_plot[['gene', 'n_PIDs', 'pct_of_all_PIDs', 'n_plasmids',
                          'card_ncbi_plasmid']], on='gene', how='left')
)

plt.rcParams.update({
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False
})

BLUE   = '#2166ac'
ORANGE = '#d6604d'
GREEN  = '#4dac26'
GREY   = '#969696'

# ── 1. Histograms ─────────────────────────────────────────────────────────────
hist_metrics = [
    ('raw_entropy',          'Raw Shannon entropy (H)',             BLUE),
    ('relative_raw_entropy', 'Relative raw entropy',                BLUE),
    ('dup_rate',             'Duplication rate',                    BLUE),
    ('pct_mobile',           'Fraction mobile plasmids',            ORANGE),
    ('simpson_diversity',    'Simpson species diversity',           GREEN),
    ('spread_score',         'Species spread score',                GREEN),
    ('pct_of_all_PIDs',      'Gene prevalence (fraction of PIDs)',  GREY),
    ('card_ncbi_plasmid',    'CARD NCBI plasmid prevalence',        GREY),
]

ncols_h, nrows_h = 3, 3
fig, axes = plt.subplots(nrows_h, ncols_h, figsize=(11, 9))
axes = axes.flatten()

for ax, (col, label, colour) in zip(axes, hist_metrics):
    data = master[col].dropna()
    ax.hist(data, bins=25, color=colour, edgecolor=colour, linewidth=0.5)
    ax.set_xlabel(label)
    ax.set_ylabel('Genes')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.set_title(f'n = {len(data)}', fontsize=8, color=GREY)

for ax in axes[len(hist_metrics):]:
    ax.set_visible(False)

fig.suptitle('Distribution of beta-lactamase gene metrics', fontsize=11)
fig.tight_layout()
fig.savefig(PLOT_DIR / '1_histograms.pdf', bbox_inches='tight')
fig.savefig(PLOT_DIR / '1_histograms.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('saved 1_histograms')


# ── 2. Scatter pairs ──────────────────────────────────────────────────────────
scatter_pairs = [
    ('raw_entropy',          'relative_raw_entropy', 'Raw entropy',      'Relative entropy'),
    ('raw_entropy',          'dup_rate',             'Raw entropy',       'Dup. rate'),
    ('raw_entropy',          'pct_mobile',           'Raw entropy',       'pct_mobile'),
    ('raw_entropy',          'spread_score',         'Raw entropy',       'spread_score'),
    ('raw_entropy',          'simpson_diversity',    'Raw entropy',       'simpson div.'),
    ('relative_raw_entropy', 'pct_mobile',           'Rel. entropy',      'pct_mobile'),
    ('relative_raw_entropy', 'spread_score',         'Rel. entropy',      'spread_score'),
    ('dup_rate',             'pct_mobile',           'Dup. rate',         'pct_mobile'),
    ('dup_rate',             'spread_score',         'Dup. rate',         'spread_score'),
    ('pct_mobile',           'spread_score',         'pct_mobile',        'spread_score'),
    ('pct_mobile',           'simpson_diversity',    'pct_mobile',        'simpson div.'),
    ('spread_score',         'simpson_diversity',    'spread_score',      'simpson div.'),
    ('pct_of_all_PIDs',      'raw_entropy',          'prevalence',        'Raw entropy'),
    ('pct_of_all_PIDs',      'relative_raw_entropy', 'prevalence',        'Rel. entropy'),
    ('pct_of_all_PIDs',      'spread_score',         'prevalence',        'spread_score'),
]

results = []
for xcol, ycol, *_ in scatter_pairs:
    sub = master[[xcol, ycol]].dropna()
    if len(sub) > 2:
        r, p = pearsonr(sub[xcol], sub[ycol])
        results.append((r, p, len(sub)))
    else:
        results.append((np.nan, np.nan, len(sub)))

pvals = [r[1] for r in results if not np.isnan(r[1])]
_, pvals_bh_sub, _, _ = multipletests(pvals, method='fdr_bh')
pvals_bh = []
adj_idx = 0
for r, p, n in results:
    if not np.isnan(p):
        pvals_bh.append(pvals_bh_sub[adj_idx])
        adj_idx += 1
    else:
        pvals_bh.append(np.nan)

ncols = 3
nrows = int(np.ceil(len(scatter_pairs) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4.5, nrows*4))
axes = axes.flatten()

for i, ax in enumerate(axes[:len(scatter_pairs)]):
    xcol, ycol, xlabel, ylabel = scatter_pairs[i]
    sub = master[[xcol, ycol]].dropna()
    x, y = sub[xcol], sub[ycol]
    ax.scatter(x, y, s=20, color='black', alpha=1)
    if len(x) > 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m*xs + b, color='black', linewidth=1)
    r, p, n = results[i]
    p_adj = pvals_bh[i]
    annot = f'r = {r:.2f}\np = {p_adj:.2g}' if not np.isnan(r) else 'n/a'
    ax.text(0.05, 0.95, annot, transform=ax.transAxes,
            ha='left', va='top', fontsize=8, color='red', zorder=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

for ax in axes[len(scatter_pairs):]:
    ax.set_visible(False)

fig.suptitle('Pairwise metric comparisons - beta-lactamase genes', fontsize=12)
fig.tight_layout()
fig.savefig(PLOT_DIR / '2_scatter_pairs.pdf', bbox_inches='tight')
fig.savefig(PLOT_DIR / '2_scatter_pairs.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('saved 2_scatter_pairs')


# ── 3. Bar plots — top 15 genes per metric ────────────────────────────────────
print('Plotting bar plots...')

TOP15 = 15
COLORS = ['#FF6B6B', '#FF8E72', '#FFB17A', '#FFD3A5', '#FFE6B5', '#F0F0C0',
          '#C4E0D9', '#98D0E5', '#6CB0E0', '#4A90D9', '#5D6FB0', '#6F4F87',
          '#8B3A5E', '#B24C6C', '#D45E7A']

bar_metrics = [
    ('raw_entropy',          'Raw Shannon entropy (H)'),
    ('relative_raw_entropy', 'Relative raw entropy'),
    ('dup_rate',             'Duplication rate'),
    ('pct_mobile',           'Fraction mobile plasmids'),
    ('pct_conjugative',      'Fraction conjugative plasmids'),
    ('simpson_diversity',    'Simpson species diversity'),
    ('spread_score',         'Species spread score'),
    ('card_ncbi_plasmid',    'CARD NCBI plasmid prevalence'),
    ('pct_of_all_PIDs',      'Prevalence (fraction of all PIDs)'),
]

master_clean = master.groupby('gene', as_index=False).mean(numeric_only=True)

ncols_b = 3
nrows_b = int(np.ceil(len(bar_metrics) / ncols_b))
fig, axes = plt.subplots(nrows_b, ncols_b, figsize=(ncols_b*5, nrows_b*5))
axes = axes.flatten()

for i, (ax, (col, title)) in enumerate(zip(axes, bar_metrics)):
    sub = master_clean[['gene', col]].dropna()
    sub = sub.nlargest(TOP15, col).sort_values(col)
    colours = [COLORS[j % len(COLORS)] for j in range(len(sub))]
    ax.barh(sub['gene'], sub[col], height=0.7, color=colours)
    ax.set_xlabel(col)
    ax.set_title(f'Top {TOP15}: {title}', fontsize=10)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

for ax in axes[len(bar_metrics):]:
    ax.set_visible(False)

fig.tight_layout()
fig.savefig(PLOT_DIR / '3_top15_barplots.png', dpi=150, bbox_inches='tight')
fig.savefig(PLOT_DIR / '3_top15_barplots.pdf', bbox_inches='tight')
plt.close(fig)
print('saved 3_top15_barplots')


# ── 4. Combined ranking heatmap ───────────────────────────────────────────────
print('Plotting combined ranking heatmap...')

rank_cols   = ['raw_entropy', 'relative_raw_entropy', 'dup_rate', 'pct_mobile', 'spread_score']
rank_labels = ['Raw H', 'Rel. H', 'Dup. rate', 'pct_mobile', 'spread']

rank_df = master.groupby('gene', as_index=False).mean(numeric_only=True)
rank_df = rank_df[['gene'] + rank_cols].dropna()

for col in rank_cols:
    rank_df[col + '_rank'] = rank_df[col].rank(ascending=False, method='min')

rank_df['mean_rank'] = rank_df[[c + '_rank' for c in rank_cols]].mean(axis=1)
rank_df = rank_df.sort_values('mean_rank').head(40)

genes = rank_df['gene'].tolist()
mat   = rank_df[[c + '_rank' for c in rank_cols]].values

fig, ax = plt.subplots(figsize=(7, len(genes)*0.3 + 2))
im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto')

ax.set_xticks(range(len(rank_cols)))
ax.set_xticklabels(rank_labels)
ax.set_yticks(range(len(genes)))
ax.set_yticklabels(genes)
ax.set_xlabel('Metric')
ax.set_title('Combined gene ranking (top 40 by mean rank)')

for i, r in enumerate(rank_df['mean_rank']):
    ax.text(len(rank_cols)-0.3, i, f'{r:.1f}', ha='left', va='center', fontsize=7)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Rank')

fig.tight_layout()
fig.savefig(PLOT_DIR / '4_combined_ranking_heatmap.pdf', bbox_inches='tight')
fig.savefig(PLOT_DIR / '4_combined_ranking_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('saved 4_combined_ranking_heatmap')


# ═══════════════════════════════════════════════════════════════════════════════
# HUB ANALYSIS
# Builds BL-plasmid-restricted centrality graph and maps back to gene names.
# HITS (hub/authority scores) are excluded — the directed graph encodes
# upstream-in-coordinate-space, not biological transfer direction, so the
# loader/cargo interpretation is not valid without strand + orientation data.
# Hub is defined as top (100-HUB_PERCENTILE)% on any of:
#   degree, betweenness, eigenvector, closeness.
# ═══════════════════════════════════════════════════════════════════════════════

graph_dir        = Path('plasmid_batched_graphs')
species_base_dir = Path('species_specific_plasmid_analysis/big_species')
hospital_dir     = Path('hospital_analysis/graphml')
out_dir          = Path('bl_hub_results_final')
out_dir.mkdir(exist_ok=True)

ECOLI_SPECIES_LABEL = 'Escherichia_coli'
HOSPITAL_LABEL      = 'hospital'
HUB_PERCENTILE      = 95
TOP_N               = 30

print('Hub output dir:', out_dir)


# ── Functional category classifier ───────────────────────────────────────────
# beta_lac_pfam_domains was derived above from df_merged (no clan table needed)

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

def classify_domain(domain_name):
    if domain_name in beta_lac_pfam_domains:
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
    'Beta-lactamase':        '#d62728',
    'Transposon/MGE':        '#ff7f0e',
    'Conjugation/T4SS':      '#1f77b4',
    'Replication/Stability': '#2ca02c',
    'Resistance (non-BL)':   '#9467bd',
    'Other/Unknown':         '#aec7e8',
}
CATEGORY_ORDER = list(CATEGORY_COLOURS.keys())
print('Domain classifier ready.')


# ── Graph loaders ─────────────────────────────────────────────────────────────

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
    batch_files = sorted(sp_dir.glob(pattern), key=lambda p: int(p.name.split('_')[1]))
    if not batch_files:
        raise FileNotFoundError(f'No species graphml files in {sp_dir}')
    path = batch_files[-1]
    print(f'E. coli: loading {path.name}')
    return nx.read_graphml(str(path)), path

def load_max_batch_hospital(hospital_dir, hospital_label):
    pattern = f'batch_*_{hospital_label}_domain_architecture_signed_network.graphml'
    batch_files = sorted(hospital_dir.glob(pattern), key=lambda p: int(p.name.split('_')[1]))
    if not batch_files:
        raise FileNotFoundError(f'No hospital graphml files in {hospital_dir}')
    path = batch_files[-1]
    print(f'Hospital: loading {path.name}')
    return nx.read_graphml(str(path)), path

G_overall,  _ = load_max_batch_overall(graph_dir)
G_ecoli,    _ = load_max_batch_species(species_base_dir, ECOLI_SPECIES_LABEL)
G_hospital, _ = load_max_batch_hospital(hospital_dir, HOSPITAL_LABEL)

graphs = {'Overall': G_overall, 'E. coli': G_ecoli, 'Hospital': G_hospital}
for lbl, G in graphs.items():
    print(f'{lbl:10s}  n={G.number_of_nodes():>6,}  m={G.number_of_edges():>7,}')


# ── BL plasmid domain set ─────────────────────────────────────────────────────
bl_plasmids = set()
for plas_list in betas_to_plas.values():
    bl_plasmids.update(plas_list)

bl_plasmid_domains = set()
for plasmid in bl_plasmids:
    for dom, _, _ in plasmid_to_domains.get(plasmid, []):
        bl_plasmid_domains.add(dom)

print(f'\nDomains on BL-carrying plasmids: {len(bl_plasmid_domains):,}')


# ── Centrality computation (HITS removed) ────────────────────────────────────

def compute_centralities_restricted(G_full, bl_domain_set, label='', hub_pct=HUB_PERCENTILE):
    """
    Restrict G_full to nodes in bl_domain_set, then compute four centrality
    metrics. HITS is excluded — see module header for rationale.
    """
    nodes_to_keep = [n for n in G_full.nodes() if n in bl_domain_set]
    G = G_full.subgraph(nodes_to_keep).copy()
    print(f'\n[{label}] Restricted graph: {G.number_of_nodes():,} nodes, '
          f'{G.number_of_edges():,} edges')

    if G.number_of_nodes() == 0:
        print(f'  WARNING: no nodes remain — returning empty DataFrame')
        return pd.DataFrame()

    U         = G.to_undirected()
    lcc_nodes = max(nx.connected_components(U), key=len)
    U_lcc     = U.subgraph(lcc_nodes).copy()
    lcc_frac  = len(lcc_nodes) / G.number_of_nodes()
    print(f'  Undirected LCC: {U_lcc.number_of_nodes():,} nodes ({lcc_frac:.1%})')

    print('  [1/4] Degree ...')
    in_deg    = dict(G.in_degree())
    out_deg   = dict(G.out_degree())
    total_deg = {n: in_deg[n] + out_deg[n] for n in G.nodes()}
    deg_cent  = nx.degree_centrality(G)

    print('  [2/4] Betweenness ...')
    betw = nx.betweenness_centrality(U_lcc, normalized=True)

    print('  [3/4] Eigenvector ...')
    try:
        eig = nx.eigenvector_centrality(U_lcc, max_iter=1000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        print('    WARNING: eigenvector did not converge — using degree proxy')
        raw = {n: U_lcc.degree(n) for n in U_lcc.nodes()}
        mx  = max(raw.values()) or 1
        eig = {n: v / mx for n, v in raw.items()}

    print('  [4/4] Closeness ...')
    close = nx.closeness_centrality(U_lcc, wf_improved=True)

    rows = []
    for node in G.nodes():
        cat     = classify_domain(node)
        is_beta = (cat == 'Beta-lactamase')
        in_lcc  = node in lcc_nodes
        rows.append({
            'domain':            node,
            'category':          cat,
            'is_beta_lactamase': is_beta,
            'in_lcc':            in_lcc,
            'total_degree':      total_deg.get(node, 0),
            'in_degree':         in_deg.get(node, 0),
            'out_degree':        out_deg.get(node, 0),
            'degree_centrality': deg_cent.get(node, 0.0),
            'betweenness':       betw.get(node, 0.0),
            'eigenvector':       eig.get(node, 0.0),
            'closeness':         close.get(node, 0.0),
        })
    df = pd.DataFrame(rows)

    # Hub flags: union across degree, betweenness, eigenvector, closeness
    for m in ['total_degree', 'betweenness', 'eigenvector', 'closeness']:
        thresh = np.percentile(df[m], hub_pct)
        df[f'hub_by_{m}'] = df[m] >= thresh
    df['is_hub'] = df[['hub_by_total_degree', 'hub_by_betweenness',
                        'hub_by_eigenvector',  'hub_by_closeness']].any(axis=1)

    n_hubs      = df['is_hub'].sum()
    n_beta_hubs = df[df['is_hub'] & df['is_beta_lactamase']].shape[0]
    print(f'  Hubs (top {100-hub_pct}%, any metric): {n_hubs}  |  BL hubs: {n_beta_hubs}')

    return df.sort_values('total_degree', ascending=False).reset_index(drop=True)


centrality_results = {}
for lbl, G in graphs.items():
    centrality_results[lbl] = compute_centralities_restricted(
        G, bl_plasmid_domains, label=lbl)

for lbl, df in centrality_results.items():
    fname = out_dir / f'centrality_bl_restricted_{lbl.replace(" ", "_").replace(".", "")}.csv'
    df.to_csv(fname, index=False)
    print(f'Saved {fname.name}')


# ── Resolve pfam PIDs to domain names ────────────────────────────────────────

plasmid_starts_hub = {
    plas: np.array([t[2] for t in entries])
    for plas, entries in plasmid_to_domains.items()
}

def resolve_pid_to_domain(pid):
    plasmid = pid_to_plasmid(pid)
    m = PID_nogene_pattern.match(pid)
    if not plasmid or not m:
        return None, None
    pid_start = int(m.group(2))
    starts = plasmid_starts_hub.get(plasmid)
    if starts is None or len(starts) == 0:
        return plasmid, None
    idx = int(np.argmin(np.abs(starts - pid_start)))
    if abs(starts[idx] - pid_start) > COORD_TOL:
        pid_stop = int(m.group(3))
        idx2 = int(np.argmin(np.abs(starts - pid_stop)))
        if abs(starts[idx2] - pid_stop) > COORD_TOL:
            return plasmid, None
        idx = idx2
    domain_name = plasmid_to_domains[plasmid][idx][0]
    return plasmid, domain_name

gene_to_domains = defaultdict(set)
domain_to_genes = defaultdict(set)

for gene, pids in pfam_betas_to_PIDs.items():
    for pid in pids:
        _, dom = resolve_pid_to_domain(pid)
        if dom is not None:
            gene_to_domains[gene].add(dom)
            domain_to_genes[dom].add(gene)


# ── Map domain centrality to gene names ───────────────────────────────────────

CENTRALITY_METRICS = [
    'total_degree', 'degree_centrality',
    'betweenness', 'eigenvector', 'closeness',
]

gene_records = []

for lbl, cent_df in centrality_results.items():
    if cent_df.empty:
        continue
    dom_cent = cent_df.set_index('domain')

    for gene in all_betas:
        pids    = pfam_betas_to_PIDs.get(gene, [])
        domains = gene_to_domains.get(gene, set())
        present_domains = [d for d in domains if d in dom_cent.index]
        if not present_domains:
            continue
        n_pids = len(pids)
        if n_pids < MIN_OBS:
            continue

        rec = {
            'gene':         gene,
            'scope':        lbl,
            'n_pids':       n_pids,
            'n_domains':    len(present_domains),
            'domain_names': '; '.join(sorted(present_domains)),
            'is_hub':       any(dom_cent.at[d, 'is_hub'] for d in present_domains),
        }
        for m in CENTRALITY_METRICS:
            vals = [dom_cent.at[d, m] for d in present_domains if m in dom_cent.columns]
            rec[m] = float(np.mean(vals)) if vals else np.nan

        gene_records.append(rec)

gene_df = pd.DataFrame(gene_records)
gene_df.to_csv(out_dir / 'gene_centrality_bl_restricted_final.csv', index=False)
print(f'\nGene-level centrality table: {len(gene_df):,} rows  '
      f'({gene_df["gene"].nunique():,} unique genes)')
print(gene_df.groupby('scope')[['gene']].count().rename(columns={'gene': 'n_genes'}))


# ── Console summary ───────────────────────────────────────────────────────────

for lbl in centrality_results:
    sub = gene_df[gene_df['scope'] == lbl].sort_values('total_degree', ascending=False)
    if sub.empty:
        continue
    print(f'\n--- {lbl}: top {min(TOP_N, len(sub))} genes by total degree ---')
    print(sub.head(TOP_N)[
        ['gene', 'n_pids', 'total_degree', 'betweenness',
         'eigenvector', 'closeness', 'is_hub']
    ].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — BARPLOTS PER SCOPE PER METRIC
# HITS metrics removed; plots for degree, betweenness, eigenvector, closeness only
# ═══════════════════════════════════════════════════════════════════════════════

BAR_COLOR = '#4C72B0'

PLOT_METRICS = [
    ('total_degree',      'Total Degree'),
    ('degree_centrality', 'Degree Centrality'),
    ('betweenness',       'Betweenness Centrality'),
    ('eigenvector',       'Eigenvector Centrality'),
    ('closeness',         'Closeness Centrality'),
]

for scope in gene_df['scope'].unique():
    df_scope = gene_df[gene_df['scope'] == scope].copy()

    for metric, metric_label in PLOT_METRICS:
        if metric not in df_scope.columns:
            continue
        sub = (df_scope[df_scope[metric].notna()]
               .sort_values(metric, ascending=False)
               .head(TOP_N)
               .sort_values(metric, ascending=True))
        if sub.empty:
            continue

        fig_height = max(6, len(sub) * 0.35)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        bars = ax.barh(sub['gene'], sub[metric],
                       color=BAR_COLOR, edgecolor='none', height=0.7)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color='#EEEEEE', linestyle='-')
        ax.yaxis.grid(False)
        ax.tick_params(axis='y', length=0, labelsize=10)
        ax.tick_params(axis='x', colors='#333333', labelsize=10)

        max_val = sub[metric].max()
        offset  = max_val * 0.015
        for bar in bars:
            w = bar.get_width()
            ax.text(w + offset, bar.get_y() + bar.get_height() / 2,
                    f'{w:.4f}', va='center', ha='left', fontsize=9, color='#333333')

        ax.set_xlim(0, max_val * 1.15)
        ax.set_xlabel(metric_label, labelpad=12, fontsize=12,
                      fontweight='bold', color='#333333')
        ax.set_title(
            f'Top {len(sub)} {scope} genes — {metric_label}\n'
            f'(BL-plasmid restricted graph, hub threshold = top {100-HUB_PERCENTILE}%)',
            pad=16, fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        scope_safe = scope.replace(' ', '_').replace('.', '')
        save_path  = out_dir / f'barplot_{metric}_{scope_safe}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved {save_path.name}')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — HUB FLAG SUMMARY BARPLOT
# Simplified: only is_hub (HITS hub/auth removed)
# ═══════════════════════════════════════════════════════════════════════════════

scopes = [s for s in gene_df['scope'].unique() if not gene_df[gene_df['scope'] == s].empty]

fig, axes = plt.subplots(1, len(scopes), figsize=(5 * len(scopes), 5), sharey=False)
if len(scopes) == 1:
    axes = [axes]

fig.suptitle(
    f'Beta-lactamase gene hub classification\n'
    f'(BL-plasmid restricted graph, top {100-HUB_PERCENTILE}% on any of '
    f'degree / betweenness / eigenvector / closeness)',
    fontsize=12, y=1.02
)

for ax, scope in zip(axes, scopes):
    sub    = gene_df[gene_df['scope'] == scope]
    count  = int(sub['is_hub'].sum())
    bar    = ax.bar(['Hub'], [count], color='#d62728', edgecolor='none',
                    alpha=0.85, width=0.4)
    ax.text(bar[0].get_x() + bar[0].get_width() / 2,
            count + max(1, count * 0.02),
            str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title(f'{scope}\n(n genes = {len(sub)})', fontsize=11)
    ax.set_ylabel('Number of hub genes', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(out_dir / 'gene_hub_classification_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved gene_hub_classification_summary.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — CROSS-SCOPE LINE PLOT
# HITS panels removed; 2x2 grid for degree, betweenness, eigenvector, closeness
# ═══════════════════════════════════════════════════════════════════════════════

dataset_labels = list(centrality_results.keys())
x              = np.arange(len(dataset_labels))

metric_titles = {
    'total_degree': 'Total degree',
    'betweenness':  'Betweenness',
    'eigenvector':  'Eigenvector',
    'closeness':    'Closeness',
}

gene_scope_counts = gene_df.groupby('gene')['scope'].nunique()
genes_multi_scope = gene_scope_counts[gene_scope_counts >= 2].index.tolist()
print(f'\nGenes in >= 2 scopes: {len(genes_multi_scope):,}')

if genes_multi_scope:
    colours = plt.cm.tab20(np.linspace(0, 1, min(len(genes_multi_scope), 20)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        'Beta-lactamase gene centrality across plasmid scopes\n'
        '(BL-plasmid restricted graph)',
        fontsize=14, y=1.01
    )

    for ax, (metric, title) in zip(axes.flat, metric_titles.items()):
        for gene, col in zip(genes_multi_scope[:20], colours):
            vals = []
            for scope in dataset_labels:
                row = gene_df[(gene_df['gene'] == gene) & (gene_df['scope'] == scope)]
                vals.append(float(row[metric].values[0]) if len(row) > 0 else np.nan)
            ax.plot(x, vals, marker='o', linewidth=1.5, markersize=6,
                    color=col, alpha=0.8, label=gene)
            last_valid = next((v for v in reversed(vals) if not np.isnan(v)), None)
            last_idx   = next((i for i in range(len(vals)-1, -1, -1)
                               if not np.isnan(vals[i])), None)
            if last_valid is not None:
                ax.annotate(gene, xy=(x[last_idx], last_valid),
                            xytext=(4, 0), textcoords='offset points',
                            fontsize=6, color=col, va='center')

        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, fontsize=10)
        ax.set_ylabel('Score', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_dir / 'gene_centrality_cross_scope.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved gene_centrality_cross_scope.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — FILE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

for d in [OUT_DIR, PLOT_DIR, out_dir]:
    print(f'\nOutputs in {d}/')
    for f in sorted(d.iterdir()):
        if f.is_file():
            print(f'  {f.name:<60s}  {f.stat().st_size/1024:>8.1f} KB')
