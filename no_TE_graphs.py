#get pfam data 
#get ordered domains (circular)
#if the beta-lactamase is within X distance of transposable element, or within them with X distance
#(or other method of prediction)
#score as being within MGE
#calculate percentage of BE prevalence wtihin TE vs overall (as per pfam)
#do so for other pfams domains and compare
#    
#
#could also do per gene name using only mapped instances:
#gene name -> PIDs 
#PIDs -> plasmids
#specfic PID -> specific plasmid specific instance as per pfam
#-> ordered domain list
#-> see if in MGE 
#-> divide by overall number of instances of that gene 
#-> rank genes
# df merged for beta-lactamases can be done with  df_merged.filter(pl.col('target_name').str.contains('lactamase'))







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
import polars as pl
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
import pickle
import igraph as ig
import leidenalg
import random
import sys 
import os
import csv
import re
from collections import defaultdict
from pathlib import Path
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
import pickle
import igraph as ig
import leidenalg
import random
import sys
from multiprocessing import Pool
import signal


############################################################################################################################
############################################################################################################################
############################################################################################################################
#PLASMID BATCHED DEGREE, EDGES, DENSITY ETC.
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

############################################################################################################################
############################################################################################################################
############################################################################################################################
#PROCESS QUALITIES OF NETWORKS FOR DIFFERENT PLASMID BATCH SIZES


#data_dir = Path('/home/kd541/rds/hpc-work/plasmid_motif_network/intermediate')
data_dir = Path(os.path.join(os.getcwd(),'plasmid_motif_network/intermediate'))
files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))

df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()

df_merged = df_merged.with_columns(
    pl.col('strand').cast(pl.Int32).alias('strand')
)


output_path = Path(os.path.join(os.getcwd(), 'no_TE_plasmid_batched_graphs'))
os.makedirs(output_path, exist_ok=True)

all_plasmids = list(set(df_merged['plasmid']))
                    
domain_df = pd.read_csv('Pfam-A.clans.tsv', sep='\t', header=None)
domain_dict = dict(zip(domain_df[3].tolist(), domain_df[2].tolist()))

random.seed(42)


#define ways to get rid of TEs/MGEs per pfam target name, class (map with domain_dict) or accession ;;

MGE_PFAM_ACCESSIONS = {
    # Transposases — IS family DDE-recombinase superfamily
    'PF00665',  # Integrase core domain (retroviral / DDE-transposase)
    'PF01527',  # Transposase DDE domain (IS1-like)
    'PF01609',  # Transposase IS3/IS911 family
    'PF01610',  # Transposase IS3/IS911 DDE domain
    'PF02003',  # Transposase IS30 family
    'PF02022',  # Integrase (Phage int superfamily)
    'PF02316',  # Transposase IS4/IS5 family
    'PF02371',  # Resolvase/invertase N-terminal
    'PF00239',  # Resolvase HTH domain
    'PF07508',  # Transposase IS200/IS605 family (TnpB/ISDra2)
    'PF01764',  # Transposase IS5 orfA
    'PF04754',  # Transposase IS701/IS1031 family
    'PF05699',  # MULE transposase domain
    'PF10551',  # Transposase IS256 family
    'PF12761',  # Transposase zinc ribbon domain IS1
    'PF13385',  # IS1 transposase InsA N-terminal
    'PF13586',  # Transposase (DDE_Tnp_IS1)
    'PF13840',  # IS3/IS911 transposase orfA-like
    'PF00872',  # Transposase IS3/IS911-like catalytic
    'PF01398',  # Transposase IS4 family
    'PF02914',  # Transposase (IS66 family)
    'PF03050',  # Tn3 transposase DDE domain
    'PF03184',  # DDE domain of DDE transposase
    'PF04827',  # Tnp from pSAM2 Streptomyces
    'PF07592',  # ISDra2 / IS200/IS605 transposase TnpA domain
    'PF09184',  # Transposase IS4-like
    'PF09811',  # Transposase (IS200 / IS605 family) YhgA
    'PF10407',  # IS30 family transposase
    'PF12728',  # Transposase (DDE superfamily, novel IS)
    'PF13843',  # Transposase (MULE) – maize-like in bacteria
    'PF14815',  # Transposase (IS_3–IS_481 group)
    'PF15706',  # IS200 transposase Tnp B-type domain
    'PF17921',  # IS1 InsB transposase C-terminal
    # Serine and tyrosine site-specific recombinases on MGEs
    'PF00589',  # Serine recombinase catalytic domain (Tn3 resolvase/invertase)
    'PF09424',  # Serine recombinase N-terminal (large)
    'PF07508',  # IS200/IS605 TnpB (ISDra2, bridge to CRISPR Cas12k)
    'PF00589',  # Tn3 transposase N-terminal domain
    # Tyrosine recombinase / integrase
    'PF00589',  # Recombinase (overlaps above)
    'PF02899',  # Phage integrase N-terminal SAM-like domain
    'PF13102',  # Phage integrase domain
    'PF17921',  # Phage integrase C-terminal (Int_C)
    'PF00589',  # Site-specific recombinase XerC/D
    'PF02899',  # Integrase HTH domain
    'PF00589',  # Cre/Flp recombinase
    'PF02899',  # Int-like HTH
    # Specific integrase/recombinase families
    'PF00589',  # Recombinase
    'PF02902',  # Phage integrase
    'PF07022',  # Transposase-associated (tnpB-like)
    # Insertion sequence accessory
    'PF01371',  # IS1 InsA N-terminal (HTH)
    'PF02914',  # IS66 orfB transposase
    'PF13612',  # OrfC IS66 C-terminal
    'PF03108',  # IS200/IS605 orfA (TnpA)
    # ISCR / rolling-circle transposases
    'PF01797',  # Y1 phage integrase (Tyrosine recombinase)
    'PF04754',  # IS701 family transposase
    'PF06276',  # Relaxase / mobilisation nuclease (but see note below)
    # Retrotransposon / retroelement (rarely on plasmids but present in some)
    'PF00078',  # Reverse transcriptase (RT)
    'PF00552',  # Integrase zinc-binding domain
    'PF07727',  # Reverse transcriptase thumb domain
    'PF00075',  # RNase H domain
    # Phage structural / lysogenic elements on plasmid pathogenicity islands
    'PF05986',  # Phage terminase small subunit
    'PF03354',  # Phage terminase large subunit ATPase
    'PF04364',  # Phage minor tail protein
    'PF04589',  # Phage baseplate assembly
    'PF07195',  # Phage tail tape measure protein (N-term)
    'PF09068',  # Phage major tail sheath
    'PF05135',  # Phage portal protein
    'PF10145',  # Phage major capsid protein (HK97 fold)
    'PF06143',  # Phage integrase SAM-like
    # Conjugative transposon / integrative-conjugative element (ICE)
    'PF01424',  # TraI / MOB relaxase domain
    'PF03389',  # Mobilisation protein (MobA/MobL)
    'PF08751',  # TraG / coupling protein (VirD4-like) N-terminal
    # Recombination directionality factors (RDFs — always associated with integrases)
    'PF04754',  # Xis / RDF family
    'PF02914',  # RDF
    # Transposon-encoded regulatory
    'PF03004',  # Tn10/Tn5 transposase HTH domain
    'PF13009',  # Transposase associated (InsE / IS3 family)
    # CRISPR-associated (Cas) — not MGE sensu stricto but often mobile
    # (comment out if you want to keep Cas proteins)
    # 'PF09039',  # Cas1 (CRISPR-associated)
}
 
# Trim version suffixes — we only need the base accession
MGE_PFAM_ACCESSIONS = frozenset(a.split('.')[0] for a in MGE_PFAM_ACCESSIONS)
 
# ── Layer 2: Exact target_name / family name blacklist ───────────────────────
MGE_TARGET_NAMES_EXACT = frozenset({
    # IS-family transposases (Pfam family names)
    'Transposase_1',
    'Transposase_2',
    'Transposase_IS200_or_IS605',
    'Transposase_mut',
    'Transposase_21',
    'DDE_Tnp_IS1',
    'DDE_Tnp_IS240',
    'DDE_Tnp_1',
    'DDE_Tnp_4',
    'DDE_Tnp_ISAZ013',
    'DDE_3',
    'IS1_InsA',
    'IS1_InsB_1',
    'IS3_IS911',
    'IS30_Tnp',
    'IS66',
    'IS66_Orf2',
    'IS66_Orf3',
    'IS200_IS605',
    'IS701',
    'IS1_InsA',
    'InsB',
    'IS3_transposase',
    'IS481',
    'ISTron',
    # Resolvases / Invertases / Integrases
    'Resolvase',
    'Recombinase',
    'Phage_integrase',
    'Integrase_recombinase_phage',
    'Integrase',
    'Int_C',
    'Int_AP2',
    'Integrase_Zn',
    'rve',           # Retroviral integrase core
    'rve_2',
    'rve_3',
    # Reverse transcriptases
    'RVT_1',
    'RVT_2',
    'RVT_3',
    'RVT_N',
    'RVT_thumb',
    'RNase_H',
    'RNase_H2',
    # Phage structural
    'Phage_terminase_1',
    'Phage_terminase_2',
    'Terminase_6',
    'Phage_cap_E',
    'Phage_pRha',
    'Phage_Mu_Gam',
    'Phage_GPA',
    'GPW_gp25',
    # Relaxase / mobilisation
    'Relaxase',
    'MobA_MobL',
    'MobA',
    'MOB_NHLP',
    # Xis / recombination directionality factors
    'Xis',
    'RDF',
    # Tn3 family
    'Tn3_res',
    'TnpR',
    # RepA (rolling-circle replication initiation — marks ISCR transposons)
    # NOTE: RepA is also used for conjugative plasmid replication; we keep it
    # out of the blacklist to avoid over-filtering.
    # Misc
    'HTH_Tnp_IS630',
    'HTH_Tnp_Tc3_2',
    'HTH_Tnp_1',
    'HTH_Tnp_Mu_2',
    'HTH_Tnp_Mu_1',
    'Tnp_zf-ribbon_2',
    'MULE',
    'FLINT',
    'Zn_Tnp_IS1',
})
 
# ── Layer 3: Regex patterns on target_name (case-insensitive) ────────────────
# These are applied as re.search on the target_name string.
_MGE_REGEX_RAW = [
    r'transpos',          # transposase, transposon, transposition
    r'\bIS\d',            # IS1, IS3, IS10, IS26, IS630 ...
    r'\bTn\d',            # Tn3, Tn10, Tn903 ...
    r'\bICE\b',           # integrative conjugative element
    r'integrase',
    r'resolvase',
    r'invertase(?!.*sugar)',   # exclude sugar invertase (metabolism)
    r'recombinas',        # recombinase, recombination-protein
    r'retrotranspos',
    r'retroelem',
    r'retrovir',
    r'reverse.transcriptas',
    r'RVT',               # reverse transcriptase domain names
    r'RNase_H',
    r'DDE.tnp',           # DDE-type transposase naming
    r'DDE_Tnp',
    r'Tnp[AB_]',          # TnpA, TnpB, Tnp_
    r'phage.*integras',
    r'phage.*capsid',
    r'phage.*terminase',
    r'phage.*portal',
    r'phage.*tail',
    r'phage.*baseplate',
    r'phage.*sheath',
    r'mob[A-Z_]',         # MobA, MobB, MobC, Mob_
    r'relaxase',
    r'mobilisa',          # mobilisation / mobilization
    r'\bXis\b',
    r'recombination.directionality',
    r'RDF\b',
    r'insertion.seq',
    r'insertion.element',
    r'Tn3_res',
    r'MULE',
    r'HTH_Tnp',
    r'Zn_Tnp',
    r'INE\d',             # ISCO / INE-type
    r'ISCR',
    r'IS_Mu',
]
 
MGE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _MGE_REGEX_RAW]
 
 
def is_mge_domain(target_name: str, target_accession: str) -> bool:
    """
    Return True if the domain hit should be treated as a mobile/transposable
    element domain and removed from the architecture.
 
    Checks all three layers in order; returns True on the first match.
    """
    # Layer 1 — accession
    acc_base = str(target_accession).split('.')[0]
    if acc_base in MGE_PFAM_ACCESSIONS:
        return True
 
    # Layer 2 — exact name
    if str(target_name) in MGE_TARGET_NAMES_EXACT:
        return True
 
    # Layer 3 — regex
    name = str(target_name)
    for pat in MGE_PATTERNS:
        if pat.search(name):
            return True
 
    return False
 
 
############################################################################################################################
# DATA LOADING
############################################################################################################################
 
data_dir = Path(os.path.join(os.getcwd(), 'plasmid_motif_network/intermediate'))
files = sorted(data_dir.glob('parsed_selected_nonoverlap_*.parquet'))
 
df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
 
df_merged = df_merged.with_columns(
    pl.col('strand').cast(pl.Int32).alias('strand')
)
 
# ── Identify the domain-name and accession columns ───────────────────────────
# Standard HMMER domtblout column names as used when building df_merged.
# Adjust if your parquet uses different names.
#   target_name       → Pfam family short name  (e.g. "AAA_22")
#   target_accession  → Pfam accession          (e.g. "PF13476.7")
# Both are checked; if only one is present the other is set to '' safely.
 
_cols = df_merged.columns
_HAS_TARGET_NAME = 'target_name' in _cols
_HAS_TARGET_ACC  = 'target_accession' in _cols
_HAS_HMM_NAME    = 'hmm_name' in _cols     # alternative column name
_HAS_HMM_ACC     = 'hmm_acc' in _cols      # alternative column name
 
DOMAIN_NAME_COL = 'target_name'       if _HAS_TARGET_NAME else ('hmm_name' if _HAS_HMM_NAME else None)
DOMAIN_ACC_COL  = 'target_accession'  if _HAS_TARGET_ACC  else ('hmm_acc'  if _HAS_HMM_ACC  else None)
 
if DOMAIN_NAME_COL is None:
    raise ValueError(
        "Cannot find a domain-name column in df_merged. "
        "Expected 'target_name' or 'hmm_name'. "
        f"Available columns: {_cols}"
    )
 
print(f"Domain name column : {DOMAIN_NAME_COL}")
print(f"Domain acc  column : {DOMAIN_ACC_COL}")
 
# ── Filter: remove MGE domains ────────────────────────────────────────────────
# Convert to pandas for the row-wise Python-level filtering, then back to Polars.
df_pd = df_merged.to_pandas()
 
n_before = len(df_pd)
 
if DOMAIN_ACC_COL is not None:
    mge_mask = df_pd.apply(
        lambda row: is_mge_domain(row[DOMAIN_NAME_COL], row[DOMAIN_ACC_COL]),
        axis=1
    )
else:
    # No accession column — name + regex only
    mge_mask = df_pd[DOMAIN_NAME_COL].apply(
        lambda name: is_mge_domain(name, '')
    )
 
# Collect removed domains for inspection / audit
removed_domains = df_pd[mge_mask][[DOMAIN_NAME_COL] + ([DOMAIN_ACC_COL] if DOMAIN_ACC_COL else [])].drop_duplicates()
print(f"\nMGE domains removed: {mge_mask.sum():,} rows ({mge_mask.mean()*100:.2f}% of total)")
print(f"Unique MGE domain families removed: {len(removed_domains)}")
print(removed_domains.to_string())
 
df_filtered_pd = df_pd[~mge_mask].copy()
df_filtered = pl.from_pandas(df_filtered_pd)
 
n_after = len(df_filtered)
print(f"\nRows before MGE filter : {n_before:,}")
print(f"Rows after  MGE filter : {n_after:,}  (removed {n_before - n_after:,})")
 
# Save the MGE-removed domain list for auditability
output_path_no_mobile = Path(os.path.join(os.getcwd(), 'no_mobile_plasmid_networks'))
output_path_no_mobile.mkdir(exist_ok=True)
removed_domains.to_csv(output_path_no_mobile / 'removed_mge_domains.csv', index=False)
print(f"\nRemoved MGE domain list → {output_path_no_mobile / 'removed_mge_domains.csv'}")
 




# ── Replace df_merged with the filtered version for all downstream code ───────
# The variable name is intentionally kept the same so the graph-building
# section below is identical to the original.
df_merged = df_filtered



goi = list(set(list(df_merged['target_name'])))
 
tnps = [x for x in goi if 'Tnp' in x or 'tnp' in x]

goi = [x for x in goi if x not in tnps]

phages = [x for x in goi if 'phage' in x or 'Phage' in x or 'Phag' in x or 'phag' in x]

goi = [x for x in goi if x not in phages]



df_merged_no_TE = df_merged.filter(pl.col('target_name').is_in(goi))



all_plasmids = list(set(df_merged_no_TE['plasmid']))
 
domain_df = pd.read_csv('Pfam-A.clans.tsv', sep='\t', header=None)
domain_dict = dict(zip(domain_df[3].tolist(), domain_df[2].tolist()))
 
random.seed(42)

output_path = Path(os.path.join(os.getcwd(), 'no_TE_plasmid_batched_graphs'))
os.makedirs(output_path, exist_ok=True)


sys.stdout = open(os.devnull, 'w')

random.shuffle(all_plasmids) 
max_size = len(all_plasmids)
num_of_batches = 100 
batch_sizes = np.unique(np.geomspace(1, max_size, num=num_of_batches, dtype=int))
batch_num_to_plasmids = {}
for size in batch_sizes:
    batch_num_to_plasmids[size] = all_plasmids[:size]
for num, plasmids in batch_num_to_plasmids.items():
    df_filt = df_merged_no_TE.filter(pl.col('plasmid').is_in(plasmids))
    df = df_filt
    df = df.sort(['plasmid', 'start', 'ali_from'])
    #make ordered list of domain hits per gene/protein
    ordered = df.select(['plasmid','query_name','target_name','start','ali_from','strand'])
    df = ordered
    #make adj
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
    #make graph for adj
    df = adj_df
    domain_to_plasmids = defaultdict(set)
    for row in ordered.iter_rows(named=True):
        domain_to_plasmids[row['target_name']].add(row['plasmid'])
    #G = nx.DiGraph()
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
    nx.write_graphml(G, os.path.join(output_path, f'{num}_domain_architecture_network.graphml'), edge_id_from_attribute='id')
    #make graph for signed adj
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
    nx.write_graphml(F, os.path.join(output_path, f'{num}_domain_architecture_signed_network.graphml'))


sys.stdout.close()
sys.stdout = sys.__stdout__



bl_df_merged = df_merged.filter(pl.col('target_name').str.contains('lactamase'))





#get pfam data 
#get ordered domains (circular)
#if the beta-lactamase is within X distance of transposable element, or within them with X distance
#(or other method of prediction)
#score as being within MGE
#calculate percentage of BE prevalence wtihin TE vs overall (as per pfam)
#do so for other pfams domains and compare
#    
#
#could also do per gene name using only mapped instances:
#gene name -> PIDs 
#PIDs -> plasmids
#specfic PID -> specific plasmid specific instance as per pfam
#-> ordered domain list
#-> see if in MGE 
#-> divide by overall number of instances of that gene 
#-> rank genes
# df merged for beta-lactamases can be done with  df_merged.filter(pl.col('target_name').str.contains('lactamase'))



