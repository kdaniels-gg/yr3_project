
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

#
## ── 8. Per-gene metric calculation ────────────────────────────────────────────
## For each AMRFinder gene name:
##   - Identify all pfam domain hits whose query_name is in that gene's PID set
##   - Pool all (plasmid, index) positions for those domain hits
##   - Compute entropy, duplication rate, and centrality for the aggregate
##     pfam-level node(s) that represent this gene's domain(s)
#
#print("\nCalculating per-gene metrics...")
#
#records = []
#
#for gene_name in tqdm(all_gene_names, desc='Per-gene'):
#    pids_this_gene = gene_to_pids[gene_name]
#    family         = gene_to_family.get(gene_name, '')
#
#    # Find which pfam domain target_names appear in this gene's PID rows
#    # A given PID maps to one or more pfam domain hits (multiple rows in df_merged)
#    gene_df_pd = (
#        df_merged
#        .filter(pl.col('query_name').is_in(list(pids_this_gene)))
#        .select(['query_name', 'target_name', 'plasmid', 'start', 'stop'])
#        .to_pandas()
#    )
#
#    if gene_df_pd.empty:
#        records.append({
#            'gene_name': gene_name, 'gene_family': family,
#            'n_pfam_hits': 0, 'pfam_domains': '',
#            'n_copies': 0, 'n_unique_contexts': 0,
#            'raw_entropy': np.nan, 'relative_raw_entropy': np.nan,
#            'dup_events': 0, 'dup_rate': np.nan,
#            'degree_centrality': np.nan,
#            'betweenness': np.nan, 'eigenvector': np.nan, 'closeness': np.nan,
#            'is_hub_degree': False, 'is_hub_betweenness': False,
#            'is_hub_eigenvector': False, 'is_hub_closeness': False,
#            'is_hub_any': False,
#            'in_lcc': False,
#        })
#        continue
#
#    # Unique pfam domain names associated with this gene
#    pfam_domains_this_gene = sorted(gene_df_pd['target_name'].unique())
#
#    # Pool all (plasmid, index) positions across all pfam domain hits for this gene.
#    # This gives the full set of genomic occurrences of this gene's domain(s).
#    pooled_positions = []
#    for dom in pfam_domains_this_gene:
#        if dom in domain_positions:
#            # Only include positions whose plasmid+start are in this gene's PID set
#            # (i.e. don't pull in the same pfam domain from unrelated genes)
#            pids_plasmid_start = set(
#                gene_df_pd.loc[gene_df_pd['target_name'] == dom, 'query_name']
#            )
#            for plasmid, idx in domain_positions[dom]:
#                # Reconstruct the query_name from plasmid + start
#                # query_name format is plasmid_start_stop; start is in domain_positions indirectly
#                # — use the plasmid_to_domains list to confirm the domain at that index
#                # and cross-check against known PIDs for this gene
#                reconstructed_pids = {
#                    r['query_name'] for r in gene_df_pd.to_dict('records')
#                    if r['plasmid'] == plasmid
#                }
#                if reconstructed_pids & pids_this_gene:
#                    pooled_positions.append((plasmid, idx))
#
#    if not pooled_positions:
#        records.append({
#            'gene_name': gene_name, 'gene_family': family,
#            'n_pfam_hits': len(gene_df_pd), 'pfam_domains': ';'.join(pfam_domains_this_gene),
#            'n_copies': 0, 'n_unique_contexts': 0,
#            'raw_entropy': np.nan, 'relative_raw_entropy': np.nan,
#            'dup_events': 0, 'dup_rate': np.nan,
#            'degree_centrality': np.nan,
#            'betweenness': np.nan, 'eigenvector': np.nan, 'closeness': np.nan,
#            'is_hub_degree': False, 'is_hub_betweenness': False,
#            'is_hub_eigenvector': False, 'is_hub_closeness': False,
#            'is_hub_any': False,
#            'in_lcc': False,
#        })
#        continue
#
#    # ── Entropy and duplication ───────────────────────────────────────────────
#    # Use the first pfam domain as the 'dom' identifier for duplication detection
#    # (tandem copy = the same gene's domain appearing in the adjacent position).
#    # We use gene_name as the token for duplication detection here so that
#    # co-localised domains from the same gene count as duplications.
#    tokens = []
#    dup_events = 0
#    plasmids_seen = set()
#
#    for plasmid, idx in pooled_positions:
#        entries = plasmid_to_domains[plasmid]
#        n = len(entries)
#        if n <= 1:
#            continue
#        left  = entries[(idx - 1) % n]
#        right = entries[(idx + 1) % n]
#        tokens.append((left, right))
#        plasmids_seen.add(plasmid)
#        # Duplication: adjacent domain is also a beta-lactamase pfam domain
#        if left  in pfam_domains_this_gene or left  in beta_lac_pfam_domains:
#            dup_events += 1
#        if right in pfam_domains_this_gene or right in beta_lac_pfam_domains:
#            dup_events += 1
#
#    dup_rate = dup_events / len(tokens) if tokens else 0.0
#    _, H, n_copies, n_unique = context_entropy_score(tokens)
#
#    # ── Relative raw entropy ──────────────────────────────────────────────────
#    # Background: all domains on the same plasmids as this gene, excluding
#    # the gene's own pfam domains. Consistent with entropy_dist.py.
#    bg_domains = set()
#    for plas in plasmids_seen:
#        for d in plasmid_to_domains.get(plas, []):
#            if d not in pfam_domains_this_gene:
#                bg_domains.add(d)
#
#    bg_raw = [all_domain_raw_H[d] for d in bg_domains if d in all_domain_raw_H]
#    rel_raw_H = (
#        round(H - np.mean(bg_raw), 6)
#        if len(bg_raw) >= 3 and not np.isnan(H)
#        else np.nan
#    )
#
#    # ── Centrality — aggregate across pfam domain nodes ──────────────────────
#    # Each pfam domain name is a node in the graph. A gene may map to one or
#    # more pfam domain nodes (e.g. both Beta-lactamase_A and Beta-lactamase_B).
#    # We take the max across nodes (most central representation of the gene).
#    cent_deg   = max((deg_cent.get(d, 0.0)  for d in pfam_domains_this_gene
#                      if d in G_directed.nodes()), default=np.nan)
#    cent_betw  = max((betw.get(d, 0.0)      for d in pfam_domains_this_gene
#                      if d in U_lcc.nodes()), default=np.nan)
#    cent_eig   = max((eig.get(d, 0.0)       for d in pfam_domains_this_gene
#                      if d in U_lcc.nodes()), default=np.nan)
#    cent_close = max((close.get(d, 0.0)     for d in pfam_domains_this_gene
#                      if d in U_lcc.nodes()), default=np.nan)
#
#    in_lcc = any(d in lcc_nodes for d in pfam_domains_this_gene)
#
#    is_hub_deg   = (not np.isnan(cent_deg))   and cent_deg   >= thresh_deg
#    is_hub_betw  = (not np.isnan(cent_betw))  and cent_betw  >= thresh_betw
#    is_hub_eig   = (not np.isnan(cent_eig))   and cent_eig   >= thresh_eig
#    is_hub_close = (not np.isnan(cent_close)) and cent_close >= thresh_close
#    is_hub_any   = is_hub_deg or is_hub_betw or is_hub_eig or is_hub_close
#
#    records.append({
#        'gene_name'            : gene_name,
#        'gene_family'          : family,
#        'n_pfam_hits'          : len(gene_df_pd),
#        'pfam_domains'         : ';'.join(pfam_domains_this_gene),
#        'n_copies'             : n_copies,
#        'n_unique_contexts'    : n_unique,
#        'raw_entropy'          : round(H, 6)         if not np.isnan(H)         else np.nan,
#        'relative_raw_entropy' : rel_raw_H,
#        'dup_events'           : dup_events,
#        'dup_rate'             : round(dup_rate, 6),
#        'degree_centrality'    : round(cent_deg,   6) if not np.isnan(cent_deg)   else np.nan,
#        'betweenness'          : round(cent_betw,  6) if not np.isnan(cent_betw)  else np.nan,
#        'eigenvector'          : round(cent_eig,   6) if not np.isnan(cent_eig)   else np.nan,
#        'closeness'            : round(cent_close, 6) if not np.isnan(cent_close) else np.nan,
#        'is_hub_degree'        : is_hub_deg,
#        'is_hub_betweenness'   : is_hub_betw,
#        'is_hub_eigenvector'   : is_hub_eig,
#        'is_hub_closeness'     : is_hub_close,
#        'is_hub_any'           : is_hub_any,
#        'in_lcc'               : in_lcc,
#    })
#
#df_results = pd.DataFrame(records).sort_values('raw_entropy', ascending=False).reset_index(drop=True)
#


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