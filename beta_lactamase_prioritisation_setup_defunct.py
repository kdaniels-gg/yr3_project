import os
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.cluster import DBSCAN
 
 
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
 
# ─── beta-lactamase hits ─────────────────────────────────────────────────────
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
                .to_csv('card_prevalence_betas_redone.txt'))
 
 
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
 
card_prevalence = pd.read_csv('card_prevalence.txt', sep='\t')
card_sub = (card_prevalence.loc[card_prevalence['Name'].isin(all_betas),
                                ['Name', 'NCBI Plasmid', 'NCBI Chromosome']]
                            .rename(columns={'Name': 'gene',
                                             'NCBI Plasmid':     'card_ncbi_plasmid',
                                             'NCBI Chromosome':  'card_ncbi_chrom'}))
 
prev_df = prev_df.merge(card_sub, on='gene', how='left')
prev_df.to_csv('beta_lactamase_prevalence.csv', index=False)
print(prev_df.head(30).to_string())
 
 
# ─── PLSDB metadata ───────────────────────────────────────────────────────────
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
# Per-plasmid counts (one row per unique plasmid per gene, not per protein
# instance) so multi-gene plasmids are not over-represented.
# pct_mobile = (conjugative + mobilisable) / total plasmids for that gene.
# Mann-Whitney: each gene's binary conjugative vector vs the median gene.
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
    total      = row['total']
    pct        = row[categories] / total
    n_mobile   = int(row['conjugative']) + int(row['mobilizable'])
    mob_results.append({
        'gene':               gene,
        'n_plasmids':         int(total),
        'pct_conjugative':    round(pct['conjugative'],     4),
        'pct_mobilizable':    round(pct['mobilizable'],     4),
        'pct_non_mobilizable':round(pct['non-mobilizable'], 4),
        'pct_mobile':         round(n_mobile / total,       4),
    })
 
mob_df = pd.DataFrame(mob_results)
 
# Mann-Whitney: each gene vs the gene with pct_conjugative closest to median
gene_conj_obs = {
    r['gene']: [1] * int(r['n_plasmids'] * r['pct_conjugative'])
             + [0] * (r['n_plasmids'] - int(r['n_plasmids'] * r['pct_conjugative']))
    for _, r in mob_df.iterrows() if r['n_plasmids'] >= MIN_OBS
}
med_val   = mob_df['pct_conjugative'].median()
med_gene  = (mob_df['pct_conjugative'] - med_val).abs().idxmin()
med_gene  = mob_df.loc[med_gene, 'gene']
ref_conj  = gene_conj_obs[med_gene]
 
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
mob_df.to_csv('beta_lactamase_mobility_stats.csv', index=False)
print(mob_df.to_string())
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — SPECIES BREADTH
# For each gene, compute:
#   simpson_diversity       — 1 - sum(p_i^2): prob two random obs differ in species
#   spread_score            — simpson / (1 - 1/n_species): simpson relative to
#                             what it would be if plasmids were perfectly spread
#                             across the observed species.
#
# n_species and species_per_plas are omitted: n_species is strongly confounded
# by prevalence (more observations → more species simply by sampling), and
# species_per_plas is its ratio form which inherits the same problem.
# Simpson and spread_score both account for prevalence implicitly through
# the probability structure, so they are retained.
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
    n_plas   = len(slist)
    counts   = Counter(slist)
    n_spc    = len(counts)
    simpson  = 1.0 - sum((c / n_plas) ** 2 for c in counts.values())
    # max possible simpson for this n_spc = 1 - 1/n_spc (perfectly even)
    max_simp = 1.0 - 1.0 / n_spc if n_spc > 1 else 0.0
    spread   = round(simpson / max_simp, 5) if max_simp > 0 else 0.0
    spc_results.append({
        'gene':              gene,
        'simpson_diversity': round(simpson, 5),
        'spread_score':      spread,
    })
 
spc_df = pd.DataFrame(spc_results).sort_values('spread_score', ascending=False)
spc_df.to_csv('beta_lactamase_species_breadth.csv', index=False)
print(spc_df.to_string())
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — RECOMBINATION / NEIGHBOUR ENTROPY
#
# Data model: signed adjacency matrix of ordered Pfam domain hits on circular
# plasmids. Each gene (beta-lactamase) maps to one or more PIDs; each PID maps
# to one position on one plasmid; each position has a (left, right) neighbour
# pair drawn from the circular domain order.
#
# Three metrics are computed:
#
# ── norm_H (plasmid-deduplicated) ────────────────────────────────────────────
#   Collapse to unique (plasmid, context_token) pairs, then compute:
#     H     = Shannon entropy over context-type frequencies
#     norm_H = H / log2(n_unique_context_types)  in [0, 1]
#   A gene appearing k times on one plasmid with one context contributes 1
#   token, not k. Prevents within-plasmid duplication inflating entropy.
#
# ── relative_H (cross-plasmid, deduplicated background) ──────────────────────
#   focal_norm_H (deduplicated) minus the mean deduplicated norm_H of all other
#   domains appearing on the same set of plasmids. Answers: does this gene show
#   more contextual diversity across plasmids than the typical domain on those
#   plasmids? Kept because per-plasmid duplication is itself biologically
#   informative (see normH_per_copy below for the complementary copy-aware view).
#
# ── normH_per_copy ────────────────────────────────────────────────────────────
#   Uses ALL PID instances (including multiple copies on the same plasmid).
#   Intentionally does NOT deduplicate: duplication is part of the signal.
#
#   For each gene, pool all n_copies context tokens (one per resolved PID).
#   Compute:
#     H_raw       = Shannon entropy over the context-type frequency distribution
#     norm_H_raw  = H_raw / log2(n_unique_context_types)   (maps H to [0,1])
#     normH_per_copy = norm_H_raw / n_copies
#
#   Division by n_copies penalises genes that need many copies to achieve their
#   contextual diversity. A gene appearing 100 times with the same entropy as
#   one appearing 5 times scores ~20x lower. This rewards genes that achieve
#   high context diversity efficiently — i.e. are present in varied
#   neighbourhoods without extreme copy-number amplification.
#
#   relative_normH_per_copy: the same quantity made relative to background.
#   For each other domain on the same set of plasmids, compute its own
#   normH_per_copy (all PID instances, same formula). Take the mean of those
#   background values, then:
#     relative_normH_per_copy = focal_normH_per_copy - mean(background)
#   Both focal and background use the same n_copies denominator logic, so the
#   comparison is like-for-like.
#
# ═══════════════════════════════════════════════════════════════════════════════
#As a heads up the flow is test (i.e., gene name mapping) -> PIDs with gene name -> df_merged (i.e., pfam domain hits) -> pfam query_ids -> circular ordered domain hit list
#The gene names aren't known for the other components on the plasmid, these are just treated as per being a node on the network (i.e., pfam domain)

print('\n--- RECOMBINATION / NEIGHBOUR ENTROPY ---')
 
data_dir   = Path('plasmid_motif_network/intermediate')
clust_path = Path('clustering_results/umap_hdbscan_clusters.csv')
OUT_DIR    = Path('recombination_results')
OUT_DIR.mkdir(exist_ok=True)
 
files     = sorted(data_dir.glob('parsed_selected_nonoverlap_*.parquet'))
df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
df_merged = df_merged.with_columns(pl.col('strand').cast(pl.Int32))
 
clust_df           = pd.read_csv(clust_path)
plasmid_to_cluster = dict(zip(clust_df['plasmid'], clust_df['cluster']))
 
# ordered domain list per plasmid, including start coordinate
ordered_df = df_merged.sort(['plasmid', 'start', 'ali_from']).select(['plasmid', 'target_name', 'strand', 'start'])
 
# plasmid -> [(domain, strand, start), ...]
plasmid_to_domains = defaultdict(list)
for row in ordered_df.iter_rows(named=True):
    plasmid_to_domains[row['plasmid']].append((row['target_name'], row['strand'], row['start']))
 
# plasmid -> sorted array of start coords for fast nearest-neighbour lookup
plasmid_starts = {
    plas: np.array([t[2] for t in entries])
    for plas, entries in plasmid_to_domains.items()
}
 
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
        # try stop coord
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
    #treats plasmids as circular
    left  = entries[(idx - 1) % n][0]
    right = entries[(idx + 1) % n][0]
    return left, right
 
def shannon_entropy_norm(context_list):
    """
    Compute normalised Shannon entropy over a list of (hashable) context tokens.
    Returns (norm_H, n_tokens, n_unique).
    norm_H = H / log2(n_unique); 0 if n_unique <= 1 or n_tokens <= 1.
    """
    counts = Counter(context_list)
    total  = sum(counts.values())
    n_uniq = len(counts)
    if total <= 1 or n_uniq <= 1:
        return 0.0, total, n_uniq
    H = -sum((c / total) * np.log2(c / total) for c in counts.values())
    norm_H = H / np.log2(n_uniq)
    return round(norm_H, 5), total, n_uniq
 
 
# resolve all pfam-source beta-lactamase PIDs to positions
pfam_test    = test.loc[test['query_id'].isin(kept_pfam_PIDs)]
pid_to_index = {}   # pid -> (plasmid, idx)
for qid in pfam_test['query_id']:
    plasmid, idx = resolve_pid_index(qid)
    if idx is not None:
        pid_to_index[qid] = (plasmid, idx)
 
found    = len(pid_to_index)
notfound = len(pfam_test) - found
print(f'PID position resolved: {found}  unresolved: {notfound}')
 
 
def deduplicated_contexts(pid_list, pid_to_index):
    """
    Collapse to unique (plasmid, context_token) pairs.
    Returns (tokens, plasmids_seen).
    Each (plasmid, context_token) pair contributes at most once — prevents
    within-plasmid identical copies inflating entropy.
    """
    seen          = set()
    tokens        = []
    plasmids_seen = set()
    for pid in pid_list:
        if pid not in pid_to_index:
            continue
        plasmid, idx = pid_to_index[pid]
        left, right  = get_neighbours_by_index(plasmid, idx)
        if left is None:
            continue
        token = (left, right)
        key   = (plasmid, token)
        if key not in seen:
            seen.add(key)
            tokens.append(token)
            plasmids_seen.add(plasmid)
    return tokens, plasmids_seen
 
 
def raw_contexts(pid_list, pid_to_index):
    """
    Collect ALL context tokens for a list of PIDs — no deduplication.
    Each resolved PID contributes one token regardless of copy number.
    Also returns the set of plasmids represented.
    Used for normH_per_copy where duplication is part of the signal.
    """
    tokens        = []
    plasmids_seen = set()
    for pid in pid_list:
        if pid not in pid_to_index:
            continue
        plasmid, idx = pid_to_index[pid]
        left, right  = get_neighbours_by_index(plasmid, idx)
        if left is None:
            continue
        tokens.append((left, right))
        plasmids_seen.add(plasmid)
    return tokens, plasmids_seen
 
 
def H_per_copy_score(token_list):
    """
    Compute H_per_copy = H / log2(n_copies).
 
    n_copies = len(token_list)  - all PID instances, no deduplication.
    H        = raw Shannon entropy (bits) over context-type frequencies.
               NOT divided by log2(n_unique): each PID is an independent
               opportunity to land in a new context, so growth of H with
               n_unique contexts is the signal, not noise to be suppressed.
 
    log2(n_copies) is the maximum possible H for n_copies observations
    (achieved when every copy has a unique context), so:
 
        H / log2(n_copies)  in [0, 1]
 
    gives the fraction of maximum possible contextual diversity actually
    achieved, directly comparable across genes with different copy numbers.
    A gene with 5 copies and one with 500 copies are on the same scale:
    both are asked "given your copy count, how diverse were your contexts?"
 
    Edge case: n_copies <= 1 -> log2 undefined or zero -> returns NaN.
    MIN_OBS filter upstream handles this for the focal gene; the background
    cache applies the same MIN_OBS guard.
 
    Returns (H_per_copy, H_raw, n_copies, n_unique).
    """
    n_copies = len(token_list)
    if n_copies <= 1:
        return np.nan, np.nan, n_copies, 0
    counts = Counter(token_list)
    n_uniq = len(counts)
    if n_uniq <= 1:
        return 0.0, 0.0, n_copies, n_uniq
    H = -sum((c / n_copies) * np.log2(c / n_copies) for c in counts.values())
    score = round(H / np.log2(n_copies), 6)
    return score, round(H, 6), n_copies, n_uniq
 
 
# ── build domain position index (shared by both metrics) ─────────────────────
# domain -> list of (plasmid, idx)   across ALL plasmids
domain_positions = defaultdict(list)
for plasmid, entries in plasmid_to_domains.items():
    n = len(entries)
    if n <= 1:
        continue
    for i, (dom, _, _) in enumerate(entries):
        domain_positions[dom].append((plasmid, i))
 
 
# ── background caches ─────────────────────────────────────────────────────────
# Two separate caches, one per metric family, built once up front.
 
# 1. Deduplicated norm_H  — for relative_H background
print('Building deduplicated background domain entropies (for relative_H)...')
 
def _domain_dedup_entropy(plasmid_idx_list):
    seen   = set()
    tokens = []
    for plasmid, idx in plasmid_idx_list:
        entries = plasmid_to_domains[plasmid]
        n = len(entries)
        if n <= 1:
            continue
        left  = entries[(idx - 1) % n][0]
        right = entries[(idx + 1) % n][0]
        token = (left, right)
        key   = (plasmid, token)
        if key not in seen:
            seen.add(key)
            tokens.append(token)
    h, _, _ = shannon_entropy_norm(tokens)
    return h, len(tokens)
 
domain_dedup_H = {}    # domain -> dedup norm_H  (only if >= MIN_OBS tokens)
for dom, pos_list in domain_positions.items():
    h, n = _domain_dedup_entropy(pos_list)
    if n >= MIN_OBS:
        domain_dedup_H[dom] = h
print(f'  {len(domain_dedup_H):,} domains with >= {MIN_OBS} deduplicated observations.')
 
# 2. H_per_copy  — for relative_H_per_copy background
print('Building H_per_copy background domain scores...')
 
def _domain_raw_H_per_copy(plasmid_idx_list):
    tokens = []
    for plasmid, idx in plasmid_idx_list:
        entries = plasmid_to_domains[plasmid]
        n = len(entries)
        if n <= 1:
            continue
        left  = entries[(idx - 1) % n][0]
        right = entries[(idx + 1) % n][0]
        tokens.append((left, right))
    score, _, n_copies, _ = H_per_copy_score(tokens)
    return score, n_copies
 
domain_H_per_copy = {}   # domain -> H_per_copy  (only if >= MIN_OBS copies)
for dom, pos_list in domain_positions.items():
    score, n = _domain_raw_H_per_copy(pos_list)
    if n >= MIN_OBS and not np.isnan(score):
        domain_H_per_copy[dom] = score
print(f'  {len(domain_H_per_copy):,} domains with >= {MIN_OBS} raw observations.')
 
 
# ── overall scores ────────────────────────────────────────────────────────────
print('Computing overall scores...')
overall_records = []
 
for gene, pids in pfam_betas_to_PIDs.items():
 
    # ── norm_H (deduplicated) ─────────────────────────────────────────────────
    dedup_tokens, gene_plasmids = deduplicated_contexts(pids, pid_to_index)
    norm_H, n_dedup, n_uniq = shannon_entropy_norm(dedup_tokens)
    if n_dedup < MIN_OBS:
        continue
 
    # focal pfam domain names (a gene may hit multiple domains)
    focal_domains = set()
    for pid in pids:
        if pid in pid_to_index:
            plasmid, idx = pid_to_index[pid]
            focal_domains.add(plasmid_to_domains[plasmid][idx][0])
 
    # background domains: present on the same plasmids, not the focal gene
    bg_domains = set()
    for plas in gene_plasmids:
        for dom, _, _ in plasmid_to_domains.get(plas, []):
            if dom not in focal_domains:
                bg_domains.add(dom)
 
    # ── relative_H (deduplicated focal vs deduplicated background) ────────────
    bg_dedup = [domain_dedup_H[d] for d in bg_domains if d in domain_dedup_H]
    if len(bg_dedup) >= 3:
        relative_H   = round(norm_H - float(np.mean(bg_dedup)), 5)
        n_bg_dedup   = len(bg_dedup)
    else:
        relative_H   = np.nan
        n_bg_dedup   = 0
 
    # ── H_per_copy (all PID instances, no dedup, raw H) ──────────────────────
    all_tokens, _ = raw_contexts(pids, pid_to_index)
    hpc, H_raw, n_copies, _ = H_per_copy_score(all_tokens)
 
    # ── relative_H_per_copy (per-copy focal vs per-copy background) ───────────
    # Background: each other domain's H_per_copy computed globally across all
    # plasmids (same formula: raw H / n_copies_total), restricted to domains
    # present on the same plasmids as this gene for comparability.
    bg_hpc = [domain_H_per_copy[d] for d in bg_domains if d in domain_H_per_copy]
    if len(bg_hpc) >= 3 and not np.isnan(hpc):
        rel_hpc    = round(hpc - float(np.mean(bg_hpc)), 8)
        n_bg_hpc   = len(bg_hpc)
    else:
        rel_hpc    = np.nan
        n_bg_hpc   = 0
 
    overall_records.append({
        'gene':                 gene,
        'n_dedup_contexts':     n_dedup,
        'n_unique_contexts':    n_uniq,
        'norm_neighbour_H':     norm_H,
        'relative_H':           relative_H,
        'n_bg_domains_dedup':   n_bg_dedup,
        'n_copies':             n_copies,
        'H_raw':                H_raw,
        'H_per_copy':           hpc,
        'relative_H_per_copy':  rel_hpc,
        'n_bg_domains_per_copy':n_bg_hpc,
    })
 
overall_df = pd.DataFrame(overall_records).sort_values('norm_neighbour_H', ascending=False)
overall_df.to_csv(OUT_DIR / 'recombination_overall.csv', index=False)
print(overall_df.to_string())
 
# ── combined summary ──────────────────────────────────────────────────────────
overall_df.to_csv(OUT_DIR / 'recombination_summary.csv', index=False)
print(overall_df.to_string())
 
print('\nDone.')
 
 
 
 
#PLOTS
 
import re
from pathlib import Path
from collections import Counter, defaultdict
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
 
OUT_DIR  = Path('recombination_results')
PLOT_DIR = Path('metric_plots')
PLOT_DIR.mkdir(exist_ok=True)
 
# load all metric tables
prev_df   = pd.read_csv('beta_lactamase_prevalence.csv')
mob_df    = pd.read_csv('beta_lactamase_mobility_stats.csv')
spc_df    = pd.read_csv('beta_lactamase_species_breadth.csv')
recomb_df = pd.read_csv(OUT_DIR / 'recombination_summary.csv')
 
# master merge on gene
master = (recomb_df
    .merge(mob_df[['gene', 'pct_mobile', 'pct_conjugative', 'p_adj_mobility']], on='gene', how='left')
    .merge(spc_df[['gene', 'simpson_diversity', 'spread_score']], on='gene', how='left')
    .merge(prev_df[['gene', 'n_PIDs', 'pct_of_all_PIDs', 'n_plasmids',
                    'card_ncbi_plasmid']], on='gene', how='left')
)
 
# plot style
plt.rcParams.update({
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False
})
 
BLUE   = '#2166ac'
ORANGE = '#d6604d'
GREEN  = '#4dac26'
GREY   = '#969696'
 
# ═══════════════════════════════════════════════════════════════════════════════
# 1.  HISTOGRAMS
# Metrics: H_per_copy, relative_H_per_copy, pct_mobile, simpson_diversity,
#          spread_score, pct_of_all_PIDs, card_ncbi_plasmid
# 2x3 grid matching original layout + one extra panel = 3x3, hide unused
# ═══════════════════════════════════════════════════════════════════════════════
metrics = [
    ('H_per_copy',          'H per copy  [H / log2(n copies)]',  BLUE),
    ('relative_H_per_copy', 'Relative H per copy',               BLUE),
    ('pct_mobile',          'Fraction mobile plasmids',          ORANGE),
    ('simpson_diversity',   'Simpson diversity',                  GREEN),
    ('spread_score',        'Species spread score',               GREEN),
    ('pct_of_all_PIDs',     'Gene prevalence (fraction of PIDs)', GREY),
    ('card_ncbi_plasmid',   'CARD NCBI plasmid prevalence',       GREY),
]
 
ncols_h, nrows_h = 3, 3   # 3x3 = 9 panels, 7 used, 2 hidden
fig, axes = plt.subplots(nrows_h, ncols_h, figsize=(11, 9))
axes = axes.flatten()
 
for ax, (col, label, colour) in zip(axes, metrics):
    data = master[col].dropna()
    ax.hist(data, bins=25, color=colour, edgecolor=colour, linewidth=0.5)
    ax.set_xlabel(label)
    ax.set_ylabel('Genes')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.set_title(f'n = {len(data)}', fontsize=8, color=GREY)
 
for ax in axes[len(metrics):]:
    ax.set_visible(False)
 
fig.suptitle('Distribution of beta-lactamase gene metrics', fontsize=11)
fig.tight_layout()
fig.savefig(PLOT_DIR / '1_histograms.pdf', bbox_inches='tight')
fig.savefig(PLOT_DIR / '1_histograms.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('saved 1_histograms')
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SCATTER PAIRS
# ═══════════════════════════════════════════════════════════════════════════════
scatter_pairs = [
    ('H_per_copy',         'relative_H_per_copy', 'H per copy',    'relative H per copy'),
    ('H_per_copy',         'pct_mobile',           'H per copy',    'pct_mobile'),
    ('H_per_copy',         'spread_score',         'H per copy',    'spread_score'),
    ('H_per_copy',         'simpson_diversity',    'H per copy',    'simpson div.'),
    ('relative_H_per_copy','pct_mobile',           'rel H per copy','pct_mobile'),
    ('relative_H_per_copy','spread_score',         'rel H per copy','spread_score'),
    ('pct_mobile',         'spread_score',         'pct_mobile',    'spread_score'),
    ('pct_mobile',         'simpson_diversity',    'pct_mobile',    'simpson div.'),
    ('spread_score',       'simpson_diversity',    'spread_score',  'simpson div.'),
    ('pct_of_all_PIDs',    'H_per_copy',           'prevalence',    'H per copy'),
    ('pct_of_all_PIDs',    'relative_H_per_copy',  'prevalence',    'rel H per copy'),
    ('pct_of_all_PIDs',    'spread_score',         'prevalence',    'spread_score'),
]
 
results = []
for xcol, ycol, *_ in scatter_pairs:
    sub = master[[xcol, ycol]].dropna()
    r, p = pearsonr(sub[xcol], sub[ycol])
    results.append((r, p, len(sub)))
 
pvals = [r[1] for r in results]
_, pvals_bh, _, _ = multipletests(pvals, method='fdr_bh')
 
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
    ax.text(
        0.05, 0.95,
        f'r = {r:.2f}\np = {p_adj:.2g}',
        transform=ax.transAxes,
        ha='left', va='top', fontsize=8, color='red', zorder=10,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2)
    )
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
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 3.  BAR PLOTS - top 15 genes for each key metric
# ═══════════════════════════════════════════════════════════════════════════════
print('Plotting bar plots...')
 
TOP15 = 15
COLORS = ['#FF6B6B', '#FF8E72', '#FFB17A', '#FFD3A5', '#FFE6B5', '#F0F0C0',
          '#C4E0D9', '#98D0E5', '#6CB0E0', '#4A90D9', '#5D6FB0', '#6F4F87',
          '#8B3A5E', '#B24C6C', '#D45E7A']
 
bar_metrics = [
    ('H_per_copy',          'H per copy [H / log2(n copies)]'),
    ('relative_H_per_copy', 'Relative H per copy'),
    ('pct_mobile',          'Fraction mobile plasmids'),
    ('pct_conjugative',     'Fraction conjugative plasmids'),
    ('simpson_diversity',   'Simpson species diversity'),
    ('spread_score',        'Species spread score'),
    ('card_ncbi_plasmid',   'CARD NCBI plasmid prevalence'),
    ('pct_of_all_PIDs',     'Prevalence (fraction of all PIDs)'),
]
 
master_clean = master.groupby('gene', as_index=False).mean(numeric_only=True)
 
fig, axes = plt.subplots(4, 2, figsize=(12, 18))
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
 
fig.tight_layout()
fig.savefig(PLOT_DIR / '3_top15_barplots.png', dpi=150, bbox_inches='tight')
fig.savefig(PLOT_DIR / '3_top15_barplots.pdf', bbox_inches='tight')
plt.close(fig)
print('saved 3_top15_barplots')
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 4.  COMBINED RANKING HEATMAP
# Metrics: H_per_copy, relative_H_per_copy, pct_mobile, spread_score
# ═══════════════════════════════════════════════════════════════════════════════
print('Plotting combined ranking heatmap...')
 
rank_cols   = ['H_per_copy', 'relative_H_per_copy', 'pct_mobile', 'spread_score']
rank_labels = ['H per copy', 'rel H per copy',       'pct_mobile', 'spread_score']
 
rank_df = master.groupby('gene', as_index=False).mean(numeric_only=True)
rank_df = rank_df[['gene'] + rank_cols].dropna()
 
for col in rank_cols:
    rank_df[col + '_rank'] = rank_df[col].rank(ascending=False, method='min')
 
rank_df['mean_rank'] = rank_df[[c + '_rank' for c in rank_cols]].mean(axis=1)
rank_df = rank_df.sort_values('mean_rank').head(40)
 
genes = rank_df['gene'].tolist()
mat   = rank_df[[c + '_rank' for c in rank_cols]].values
 
fig, ax = plt.subplots(figsize=(6, len(genes)*0.3 + 2))
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












#NOT WHAT I NEED 
#NEED TO JUST GET PIDS FOR GENE NAME, GET PLASMIDS FOR THESE PIDS, AND CALCULATE CENTRALITY ETC., NORMALLY TO IDENTIFY HUBS
#AS LIKE IN COMPARE_HUBS, BUT ONLY RETURN THE VALUES FOR THE BETA-LACTAMASES BY GENE NAME
import os
import re
import math
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─── patterns (from chad_protein.py) ─────────────────────────────────────────
PID_nuccore_pattern = re.compile(r'^(.+?)_\d+_\d+')
PID_nogene_pattern  = re.compile(r'^(.+?)_(\d+)_(\d+)$')
MIN_OBS = 5

def pid_to_plasmid(pid):
    m = PID_nuccore_pattern.match(pid)
    return m.group(1) if m else None


# =============================================================================
# SECTION 1 — PATHS & PARAMETERS
# =============================================================================

data_dir         = Path('plasmid_motif_network/intermediate')
graph_dir        = Path('plasmid_batched_graphs')
species_base_dir = Path('species_specific_plasmid_analysis/big_species')
hospital_dir     = Path('hospital_analysis/graphml')
amr_csv          = Path('amrfindermapped_beta_lactamases.csv')
merged_fasta_dir = Path('merged_nonoverlapping_fastas')
pfam_clans_tsv   = 'Pfam-A.clans.tsv'
out_dir          = Path('bl_hub_results')
out_dir.mkdir(exist_ok=True)

ECOLI_SPECIES_LABEL = 'Escherichia_coli'
HOSPITAL_LABEL      = 'hospital'
HUB_PERCENTILE      = 95    # top 5% = hub
TOP_N               = 30    # bars shown per plot
COORD_TOL           = 50    # bp tolerance for PID coordinate matching

print('Paths set. Output:', out_dir)


# =============================================================================
# SECTION 2 — PFAM CLAN ANNOTATIONS  (from compare_hubs.py)
# =============================================================================

domain_df         = pd.read_csv(pfam_clans_tsv, sep='\t', header=None)
domain_dict       = dict(zip(domain_df[3].tolist(), domain_df[2].tolist()))
domain_dict_clean = {
    k: (None if isinstance(v, float) and math.isnan(v) else v)
    for k, v in domain_dict.items()
}
beta_lac_domains = {k for k, v in domain_dict_clean.items() if v == 'Beta-lactamase'}
print(f'Beta-lactamase Pfam domains ({len(beta_lac_domains)}): {sorted(beta_lac_domains)}')


# =============================================================================
# SECTION 3 — FUNCTIONAL CATEGORY LABELLER  (from compare_hubs.py)
# =============================================================================

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
    if domain_name in beta_lac_domains:
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


# =============================================================================
# SECTION 4 — LOAD AMR TABLE + BUILD GENE→PID MAPS  (from chad_protein.py)
# =============================================================================

merged_kept_PIDs = ['.'.join(x.split('.')[:-1]) for x in os.listdir(merged_fasta_dir)]
kept_pfam_PIDs   = [p for p in merged_kept_PIDs if PID_nogene_pattern.match(p)]

test = pd.read_csv(amr_csv, low_memory=False)
test = test.loc[test['query_id'].isin(merged_kept_PIDs)]
all_betas = [x for x in test['gene_name'].unique() if isinstance(x, str)]

pfam_betas_to_PIDs = {}
betas_to_plas      = {}
for gene in all_betas:
    queries = list(set(test.loc[test['gene_name'] == gene, 'query_id']))
    pfam_betas_to_PIDs[gene] = [x for x in queries if PID_nogene_pattern.match(x)]
    betas_to_plas[gene] = list(set(
        m.group(1) for x in queries if (m := PID_nuccore_pattern.match(x))
    ))

# All plasmid IDs that carry at least one beta-lactamase PID
bl_plasmids = set()
for plas_list in betas_to_plas.values():
    bl_plasmids.update(plas_list)

print(f'Beta-lactamase genes     : {len(all_betas):,}')
print(f'Plasmids with any BL PID : {len(bl_plasmids):,}')


# =============================================================================
# SECTION 5 — LOAD PARQUET + BUILD PLASMID DOMAIN MAP
# =============================================================================
# Needed to resolve pfam PID coordinates → domain name + index,
# and to map domain centrality scores back to the originating gene.

files     = sorted(data_dir.glob('parsed_selected_nonoverlap_*.parquet'))
df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
df_merged = df_merged.with_columns(pl.col('strand').cast(pl.Int32))

ordered_df = (df_merged
              .sort(['plasmid', 'start', 'ali_from'])
              .select(['plasmid', 'target_name', 'strand', 'start']))

plasmid_to_domains = defaultdict(list)
for row in ordered_df.iter_rows(named=True):
    plasmid_to_domains[row['plasmid']].append(
        (row['target_name'], row['strand'], row['start']))

plasmid_starts = {
    plas: np.array([t[2] for t in entries])
    for plas, entries in plasmid_to_domains.items()
}
print(f'Plasmids in parquet: {len(plasmid_to_domains):,}')


# =============================================================================
# SECTION 6 — RESOLVE PFAM PIDs TO DOMAIN NAMES
# =============================================================================
# Each pfam PID carries coordinates that tell us where in the plasmid it sits.
# We match those coordinates to the ordered domain list to get the domain name.
# This gives us the gene_name → {Pfam domain name(s)} mapping we need later.

def resolve_pid_to_domain(pid):
    """Return (plasmid_id, domain_name) for a pfam PID, or (None, None)."""
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
    domain_name = plasmid_to_domains[plasmid][idx][0]
    return plasmid, domain_name

# Build: gene_name → set of Pfam domain names that represent it
# Also: domain_name → list of gene_names (for the reverse lookup)
gene_to_domains  = defaultdict(set)   # gene_name → {domain_name, ...}
domain_to_genes  = defaultdict(set)   # domain_name → {gene_name, ...}
pid_resolved     = 0
pid_unresolved   = 0

for gene, pids in pfam_betas_to_PIDs.items():
    for pid in pids:
        _, dom = resolve_pid_to_domain(pid)
        if dom is not None:
            gene_to_domains[gene].add(dom)
            domain_to_genes[dom].add(gene)
            pid_resolved += 1
        else:
            pid_unresolved += 1

print(f'PIDs resolved to domain: {pid_resolved:,}  unresolved: {pid_unresolved:,}')
print(f'Genes with >=1 resolved domain: {sum(1 for v in gene_to_domains.values() if v):,}')


# =============================================================================
# SECTION 7 — GRAPH LOADERS  (from compare_hubs.py, unchanged)
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
    sp_dir = species_base_dir / species_label
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


# =============================================================================
# SECTION 8 — RESTRICT GRAPHS TO BL PLASMIDS + COMPUTE CENTRALITY
# =============================================================================
# For each graph scope:
#   1. Keep only nodes (domains) that appear on at least one BL-carrying plasmid.
#   2. Run the same centrality computation as compare_hubs.py on that subgraph.
#
# WHY RESTRICT?
# The full graph has tens of thousands of nodes. Restricting to the plasmid
# context where BL genes actually live gives centrality scores that reflect
# the architectural neighbourhood of BL genes, not the whole pan-plasmidome.
# A domain that is a hub on BL plasmids specifically is more relevant to
# understanding BL gene mobility than a global hub that rarely co-occurs with BL.

# Domains present on BL-carrying plasmids
bl_plasmid_domains = set()
for plasmid in bl_plasmids:
    for dom, _, _ in plasmid_to_domains.get(plasmid, []):
        bl_plasmid_domains.add(dom)

print(f'\nDomains on BL-carrying plasmids: {len(bl_plasmid_domains):,}')


def compute_centralities_restricted(G_full, bl_domain_set, label='', hub_pct=HUB_PERCENTILE):
    """
    Restrict G_full to nodes in bl_domain_set, then compute all five
    centrality metrics using the same logic as compare_hubs.py.
    Returns a DataFrame sorted by total_degree descending.
    """
    nodes_to_keep = [n for n in G_full.nodes() if n in bl_domain_set]
    G = G_full.subgraph(nodes_to_keep).copy()
    print(f'\n[{label}] Restricted graph: {G.number_of_nodes():,} nodes, '
          f'{G.number_of_edges():,} edges  '
          f'(full graph had {G_full.number_of_nodes():,} nodes)')

    if G.number_of_nodes() == 0:
        print(f'  WARNING: no nodes remain after restriction — returning empty DataFrame')
        return pd.DataFrame()

    # ── Undirected LCC ────────────────────────────────────────────────────────
    U         = G.to_undirected()
    lcc_nodes = max(nx.connected_components(U), key=len)
    U_lcc     = U.subgraph(lcc_nodes).copy()
    lcc_frac  = len(lcc_nodes) / G.number_of_nodes()
    print(f'  Undirected LCC: {U_lcc.number_of_nodes():,} nodes ({lcc_frac:.1%} of restricted graph)')

    # ── 1. Degree (directed, restricted graph) ────────────────────────────────
    print('  [1/5] Degree ...')
    in_deg    = dict(G.in_degree())
    out_deg   = dict(G.out_degree())
    total_deg = {n: in_deg[n] + out_deg[n] for n in G.nodes()}
    deg_cent  = nx.degree_centrality(G)

    # ── 2. Betweenness (undirected LCC) ───────────────────────────────────────
    print('  [2/5] Betweenness ...')
    betw = nx.betweenness_centrality(U_lcc, normalized=True)

    # ── 3. Eigenvector (undirected LCC) ───────────────────────────────────────
    print('  [3/5] Eigenvector ...')
    try:
        eig = nx.eigenvector_centrality(U_lcc, max_iter=1000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        print('    WARNING: eigenvector did not converge — using degree-normalised proxy')
        raw = {n: U_lcc.degree(n) for n in U_lcc.nodes()}
        mx  = max(raw.values()) or 1
        eig = {n: v / mx for n, v in raw.items()}

    # ── 4. Closeness (undirected LCC, Wasserman-Faust) ────────────────────────
    print('  [4/5] Closeness ...')
    close = nx.closeness_centrality(U_lcc, wf_improved=True)

    # ── 5. HITS (directed restricted graph) ───────────────────────────────────
    print('  [5/5] HITS ...')
    try:
        hits_hub, hits_auth = nx.hits(G, max_iter=1000, tol=1e-6, normalized=True)
    except nx.PowerIterationFailedConvergence:
        print('    WARNING: HITS did not converge — scores set to 0')
        hits_hub  = {n: 0.0 for n in G.nodes()}
        hits_auth = {n: 0.0 for n in G.nodes()}

    # ── Assemble DataFrame ─────────────────────────────────────────────────────
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
            'total_degree':         total_deg.get(node, 0),
            'in_degree':            in_deg.get(node, 0),
            'out_degree':           out_deg.get(node, 0),
            'degree_centrality':    deg_cent.get(node, 0.0),
            'betweenness':          betw.get(node, 0.0),
            'eigenvector':          eig.get(node, 0.0),
            'closeness':            close.get(node, 0.0),
            'hits_hub_score':       hits_hub.get(node, 0.0),
            'hits_authority_score': hits_auth.get(node, 0.0),
        })
    df = pd.DataFrame(rows)

    # ── Hub flags (same union logic as compare_hubs.py) ───────────────────────
    for m in ['total_degree', 'betweenness', 'eigenvector', 'closeness']:
        thresh = np.percentile(df[m], hub_pct)
        df[f'hub_by_{m}'] = df[m] >= thresh
    df['is_hub'] = df[['hub_by_total_degree', 'hub_by_betweenness',
                        'hub_by_eigenvector',  'hub_by_closeness']].any(axis=1)

    hits_hub_thresh  = np.percentile(df['hits_hub_score'],       hub_pct)
    hits_auth_thresh = np.percentile(df['hits_authority_score'], hub_pct)
    df['is_hits_hub']       = df['hits_hub_score']       >= hits_hub_thresh
    df['is_hits_authority'] = df['hits_authority_score'] >= hits_auth_thresh

    n_hubs      = df['is_hub'].sum()
    n_beta_hubs = df[df['is_hub'] & df['is_beta_lactamase']].shape[0]
    print(f'  Hubs (top {100-hub_pct}%, any metric): {n_hubs}  |  beta-lac hubs: {n_beta_hubs}')

    return df.sort_values('total_degree', ascending=False).reset_index(drop=True)


centrality_results = {}
for lbl, G in graphs.items():
    centrality_results[lbl] = compute_centralities_restricted(
        G, bl_plasmid_domains, label=lbl)

# Save domain-level centrality tables
for lbl, df in centrality_results.items():
    fname = out_dir / f'centrality_bl_restricted_{lbl.replace(" ", "_").replace(".", "")}.csv'
    df.to_csv(fname, index=False)
    print(f'Saved {fname.name}')


# =============================================================================
# SECTION 9 — MAP DOMAIN CENTRALITY BACK TO GENE NAMES
# =============================================================================
# Each beta-lactamase gene may map to one or more Pfam domain types.
# For each gene in each scope, we take the mean centrality across all
# domain names associated with that gene (usually just one domain type,
# occasionally two for closely related family members).
# We also record n_pids (number of pfam PIDs) as the observation count.

CENTRALITY_METRICS = [
    'total_degree', 'degree_centrality',
    'betweenness', 'eigenvector', 'closeness',
    'hits_hub_score', 'hits_authority_score',
]

gene_records = []

for lbl, cent_df in centrality_results.items():
    if cent_df.empty:
        continue
    # domain → centrality row as a dict for fast lookup
    dom_cent = cent_df.set_index('domain')

    for gene in all_betas:
        pids    = pfam_betas_to_PIDs.get(gene, [])
        domains = gene_to_domains.get(gene, set())

        # Only keep domains that actually appear in this scope's restricted graph
        present_domains = [d for d in domains if d in dom_cent.index]
        if not present_domains:
            continue

        n_pids = len(pids)
        if n_pids < MIN_OBS:
            continue

        # Mean centrality across all domain names for this gene
        rec = {
            'gene':          gene,
            'scope':         lbl,
            'n_pids':        n_pids,
            'n_domains':     len(present_domains),
            'domain_names':  '; '.join(sorted(present_domains)),
            'is_hub':        any(dom_cent.at[d, 'is_hub']        for d in present_domains),
            'is_hits_hub':   any(dom_cent.at[d, 'is_hits_hub']   for d in present_domains),
            'is_hits_auth':  any(dom_cent.at[d, 'is_hits_authority'] for d in present_domains),
        }
        for m in CENTRALITY_METRICS:
            vals = [dom_cent.at[d, m] for d in present_domains if m in dom_cent.columns]
            rec[m] = float(np.mean(vals)) if vals else np.nan

        gene_records.append(rec)

gene_df = pd.DataFrame(gene_records)
gene_df.to_csv(out_dir / 'gene_centrality_bl_restricted.csv', index=False)
print(f'\nGene-level centrality table: {len(gene_df):,} rows  '
      f'({gene_df["gene"].nunique():,} unique genes)')
print(gene_df.groupby('scope')[['gene']].count().rename(columns={'gene': 'n_genes'}))


# =============================================================================
# SECTION 10 — CONSOLE SUMMARY
# =============================================================================

for lbl in centrality_results:
    sub = gene_df[gene_df['scope'] == lbl].sort_values('total_degree', ascending=False)
    if sub.empty:
        continue
    print(f'\n--- {lbl}: top {min(TOP_N, len(sub))} genes by total degree ---')
    print(sub.head(TOP_N)[
        ['gene', 'n_pids', 'total_degree', 'betweenness',
         'eigenvector', 'closeness', 'is_hub']
    ].to_string(index=False))


# =============================================================================
# SECTION 11 — BARPLOTS PER SCOPE PER METRIC  (style from chad_protein.py)
# =============================================================================

BAR_COLOR = '#4C72B0'

PLOT_METRICS = [
    ('total_degree',         'Total Degree'),
    ('degree_centrality',    'Degree Centrality'),
    ('betweenness',          'Betweenness Centrality'),
    ('eigenvector',          'Eigenvector Centrality'),
    ('closeness',            'Closeness Centrality'),
    ('hits_hub_score',       'HITS Hub Score'),
    ('hits_authority_score', 'HITS Authority Score'),
]

for scope in gene_df['scope'].unique():
    df_scope = gene_df[gene_df['scope'] == scope].copy()

    for metric, metric_label in PLOT_METRICS:
        if metric not in df_scope.columns:
            continue
        sub = (df_scope[df_scope[metric].notna()]
               .sort_values(metric, ascending=False)
               .head(TOP_N)
               .sort_values(metric, ascending=True))   # ascending so highest is at top
        if sub.empty:
            continue

        fig_height = max(6, len(sub) * 0.35)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        bars = ax.barh(
            sub['gene'], sub[metric],
            color=BAR_COLOR, edgecolor='none', height=0.7
        )

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
            ax.text(
                w + offset, bar.get_y() + bar.get_height() / 2,
                f'{w:.4f}', va='center', ha='left', fontsize=9, color='#333333'
            )

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


# =============================================================================
# SECTION 12 — HUB FLAG SUMMARY BARPLOT  (how many genes are hubs per scope)
# =============================================================================
# One grouped bar chart: for each scope, count of genes classified as hubs
# broken down by whether they are is_hub / is_hits_hub / is_hits_auth.

fig, axes = plt.subplots(1, len(centrality_results), figsize=(5 * len(centrality_results), 6),
                          sharey=False)
if len(centrality_results) == 1:
    axes = [axes]

fig.suptitle(
    f'Beta-lactamase gene hub classification\n'
    f'(BL-plasmid restricted graph, top {100-HUB_PERCENTILE}% threshold)',
    fontsize=13, y=1.02
)

for ax, scope in zip(axes, gene_df['scope'].unique()):
    sub = gene_df[gene_df['scope'] == scope]
    labels   = ['Any hub\n(degree/betw/eig/close)', 'HITS hub\n(loader)', 'HITS authority\n(cargo)']
    cols     = ['is_hub', 'is_hits_hub', 'is_hits_auth']
    counts   = [sub[c].sum() for c in cols]
    colours  = ['#d62728', '#ff7f0e', '#1f77b4']

    bars = ax.bar(labels, counts, color=colours, edgecolor='none', alpha=0.85, width=0.55)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                str(int(val)), ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title(f'{scope}\n(n genes = {len(sub)})', fontsize=11)
    ax.set_ylabel('Number of genes', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=9)

plt.tight_layout()
plt.savefig(out_dir / 'gene_hub_classification_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved gene_hub_classification_summary.png')


# =============================================================================
# SECTION 13 — CROSS-SCOPE LINE PLOT  (same structure as compare_hubs.py Sec 14)
# =============================================================================
# One panel per metric; one line per gene; tracks how each gene's centrality
# (within the BL-restricted graph) varies across the three plasmid scopes.
# Flat = consistent hub position; steep = context-dependent.



all_genes_any_scope = gene_df['gene'].unique()
dataset_labels      = list(centrality_results.keys())
x                   = np.arange(len(dataset_labels))

metric_titles = {
    'total_degree':         'Total degree',
    'betweenness':          'Betweenness',
    'eigenvector':          'Eigenvector',
    'closeness':            'Closeness',
    'hits_hub_score':       'HITS hub score',
    'hits_authority_score': 'HITS authority',
}


# Only plot genes that appear in at least 2 scopes to make lines meaningful
gene_scope_counts = gene_df.groupby('gene')['scope'].nunique()
genes_multi_scope = gene_scope_counts[gene_scope_counts >= 2].index.tolist()
print(f'\nGenes in >= 2 scopes: {len(genes_multi_scope):,}')

if genes_multi_scope:
    colours = plt.cm.tab20(np.linspace(0, 1, min(len(genes_multi_scope), 20)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
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


# =============================================================================
# SECTION 14 — FILE SUMMARY
# =============================================================================

print(f'\nAll outputs written to {out_dir}/')
for f in sorted(out_dir.iterdir()):
    if f.is_file():
        print(f'  {f.name:<60s}  {f.stat().st_size/1024:>8.1f} KB')