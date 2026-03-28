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
# "other": neither BL nor MGE — the background comparison group
other_mask = ~mge_mask & ~bl_mask

df_mge   = df_pd[mge_mask].copy()
df_bl    = df_pd[bl_mask].copy()
df_other = df_pd[other_mask].copy()

print(f"MGE domain rows        : {len(df_mge):,}")
print(f"Beta-lactamase rows    : {len(df_bl):,}")
print(f"Other (background) rows: {len(df_other):,}")


# ── 3. Load AMRFinder per-gene mapping ───────────────────────────────────────
# query_id in test matches query_name in df_merged (PID format plasmid_start_stop)

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

# gene_name → set of query_names (PIDs) for that gene
gene_to_qnames = defaultdict(set)
for qid, gname in pid_to_gene.items():
    if isinstance(gname, str):
        gene_to_qnames[gname].add(qid)


# ── 4. Shared infrastructure — built once, reused everywhere ─────────────────

DIST_THRESHOLD = 5000   # nucleotides  #$$$
CIRCULAR       = True   # plasmids are always circular
SANDWICH_GAP   = 1      # max intervening domains on each side for sandwich  #$$$




mge_flat = df_mge[['plasmid', 'start', 'stop']].copy()
mge_flat.columns = ['plasmid', 'mge_start', 'mge_stop']
mge_flat = mge_flat.reset_index()


# gene_tags: one row per (plasmid, query_name), carrying is_mge / is_bl flags
df_pd['is_mge'] = mge_mask.values
df_pd['is_bl']  = bl_mask.values
gene_tags = (
    df_pd.groupby(['plasmid', 'query_name', 'start', 'stop'])
    .agg(is_mge=('is_mge', 'any'), is_bl=('is_bl', 'any'))
    .reset_index()
)


gene_tags = gene_tags.sort_values(['plasmid', 'start']).reset_index(drop=True)

# Pre-build per-plasmid arrays for fast Strategy B access
plasmid_gene_data = {}
for plasmid, grp in gene_tags.groupby('plasmid', sort=False):
    grp = grp.reset_index(drop=True)
    plasmid_gene_data[plasmid] = {
        'query_names': grp['query_name'].tolist(),
        'starts'     : grp['start'].tolist(),
        'stops'      : grp['stop'].tolist(),
        'mge'        : grp['is_mge'].to_numpy(),
    }


# ── 5. Strategy A helper (vectorised) ────────────────────────────────────────
# Takes a subset of df_pd rows (the gene group to score), merges with mge_flat
# on plasmid, then computes interval gap in one vectorised step.

def run_strategy_A(df_query: pd.DataFrame) -> pd.DataFrame:
    """
    df_query : rows to score — must contain [query_name, plasmid, start, stop]
    Returns  : one row per (query_name, plasmid, start, stop) with min_dist_nt
               and mge_assoc_A.
    """
    q = df_query[['query_name', 'plasmid', 'start', 'stop']].copy()
    q = q.rename(columns={'start': 'bl_start', 'stop': 'bl_stop'})
    merged = q.merge(mge_flat, on='plasmid', how='left')
    # Vectorised interval gap
    merged['dist'] = np.maximum(
        0,
        np.maximum(
            merged['mge_start'].values - merged['bl_stop'].values,
            merged['bl_start'].values  - merged['mge_stop'].values,
        )
    )
    # Genes on plasmids with no MGE get NaN from left join → treat as inf
    merged['dist'] = merged['dist'].fillna(np.inf)
    result = (
        merged.groupby(['query_name', 'plasmid', 'bl_start', 'bl_stop'])['dist']
        .min()
        .reset_index()
        .rename(columns={'dist': 'min_dist_nt', 'bl_start': 'start', 'bl_stop': 'stop'})
    )
    result['mge_assoc_A'] = result['min_dist_nt'] <= DIST_THRESHOLD
    return result


# ── 6. Strategy B helper ──────────────────────────────────────────────────────

def run_strategy_B(query_names_set: set, desc: str) -> pd.DataFrame:
    """
    Scores genes in query_names_set for MGE adjacency / sandwich on circular plasmids.
    Returns one row per gene instance found on any plasmid.
    """
    results = []
    reach = SANDWICH_GAP + 1   # outermost distance checked for sandwich
    for plasmid, data in tqdm(plasmid_gene_data.items(), desc=desc, leave=False):
        qnames  = data['query_names']
        mge_arr = data['mge']
        n       = len(qnames)
        target_positions = [i for i, qn in enumerate(qnames) if qn in query_names_set]
        if not target_positions:
            continue
        mge_positions = set(np.where(mge_arr)[0])
        for i in target_positions:
            # a) Direct adjacency: either circular neighbour is an MGE
            directly_adjacent = (
                ((i - 1) % n) in mge_positions or
                ((i + 1) % n) in mge_positions
            )
            # b) Sandwich: MGE within `reach` steps on BOTH the left and right
            #    circular sides independently
            left_mge_dists  = [d for d in range(1, reach + 1)
                                if ((i - d) % n) in mge_positions]
            right_mge_dists = [d for d in range(1, reach + 1)
                                if ((i + d) % n) in mge_positions]
            sandwiched = bool(left_mge_dists) and bool(right_mge_dists)
            mge_assoc_B = directly_adjacent or sandwiched
            results.append({
                'query_name'       : qnames[i],
                'plasmid'          : plasmid,
                'gene_start'       : data['starts'][i],
                'gene_stop'        : data['stops'][i],
                'directly_adjacent': directly_adjacent,
                'sandwiched'       : sandwiched,
                'mge_assoc_B'      : mge_assoc_B,
            })
    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=['query_name','plasmid','gene_start','gene_stop',
                 'directly_adjacent','sandwiched','mge_assoc_B']
    )


# ── 7. Summarise helper ───────────────────────────────────────────────────────

def summarise(df_A: pd.DataFrame, df_B: pd.DataFrame, label: str) -> dict:
    n_A     = len(df_A)
    asc_A   = int(df_A['mge_assoc_A'].sum()) if n_A > 0 else 0
    rate_A  = asc_A / n_A if n_A > 0 else np.nan

    n_B     = len(df_B)
    asc_B   = int(df_B['mge_assoc_B'].sum()) if n_B > 0 else 0
    rate_B  = asc_B / n_B if n_B > 0 else np.nan

    qn_A    = set(df_A.loc[df_A['mge_assoc_A'], 'query_name']) if n_A > 0 else set()
    qn_B    = set(df_B.loc[df_B['mge_assoc_B'], 'query_name']) if n_B > 0 else set()
    all_qn  = (set(df_A['query_name']) if n_A > 0 else set()) | \
              (set(df_B['query_name']) if n_B > 0 else set())
    pooled  = qn_A | qn_B
    n_pt    = len(all_qn)
    n_pa    = len(pooled)
    rate_p  = n_pa / n_pt if n_pt > 0 else np.nan

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


# ── 8. BL overall ────────────────────────────────────────────────────────────

df_A_bl = run_strategy_A(df_bl)
df_B_bl = run_strategy_B(set(df_bl['query_name']), desc='BL overall — Strategy B')
stats_bl = summarise(df_A_bl, df_B_bl, 'Beta-lactamases overall (pfam)')


# ── 9. Non-BL / non-MGE background ───────────────────────────────────────────

df_A_other = run_strategy_A(df_other)
df_B_other = run_strategy_B(set(df_other['query_name']), desc='Background — Strategy B')
stats_other = summarise(df_A_other, df_B_other, 'Non-BL / non-MGE background (pfam)')


# ── 10. Per-gene (AMRFinder gene names) ──────────────────────────────────────

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

df_A_bl.to_csv(    output_dir / 'BL_overall_strategy_A.csv',    index=False)
df_B_bl.to_csv(    output_dir / 'BL_overall_strategy_B.csv',    index=False)
df_A_other.to_csv( output_dir / 'other_genes_strategy_A.csv',   index=False)
df_B_other.to_csv( output_dir / 'other_genes_strategy_B.csv',   index=False)
df_per_gene.to_csv(output_dir / 'per_gene_mge_association.csv', index=False)

df_summary = pd.DataFrame([stats_bl, stats_other])[
    ['label','A_total','A_assoc','A_rate',
     'B_total','B_assoc','B_rate',
     'pool_total','pool_assoc','pool_rate']
]
df_summary.to_csv(output_dir / 'group_comparison_summary.csv', index=False)

print(f"\nAll results written to {output_dir}")
print(df_summary.to_string(index=False))