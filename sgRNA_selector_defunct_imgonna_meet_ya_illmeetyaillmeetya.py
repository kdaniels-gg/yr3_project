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




FASTA_DIR        = Path('fastas')
PFAM_FASTA_DIR   = Path('pfam_fastas')
MERGED_FASTA_DIR = Path('merged_nonoverlapping_fastas')



data_dir = Path(os.path.join(os.getcwd(),'plasmid_motif_network/intermediate'))
files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))

df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()

df_merged = df_merged.with_columns(
    pl.col('strand').cast(pl.Int32).alias('strand')
)


PID_nuccore_pattern = re.compile(r'^(.+?)_\d+_\d+')
PID_nogene_pattern  = re.compile(r'^(.+?)_(\d+)_(\d+)$')
 

merged_kept_PIDs = ['.'.join(x.split('.')[:-1]) for x in os.listdir(MERGED_FASTA_DIR)]
kept_pfam_PIDs   = [p for p in merged_kept_PIDs if PID_nogene_pattern.match(p)]
kept_fasta_PIDs  = [p for p in merged_kept_PIDs if not PID_nogene_pattern.match(p)]
total_PIDs = len(merged_kept_PIDs)


#test = pd.read_csv('amrfindermapped_beta_lactamases.csv', low_memory=False)
#test = test.loc[test['query_id'].isin(merged_kept_PIDs)]


test_new = pd.read_csv('amrfindermapped_beta_lactamases_new.csv', low_memory=False)
prev_mapped_names = ['TEM-1', 'CTX-M-140', 'CMY-4', 'CTX-M-9', 'IMP-22', 'CTX-M-2', 'PAD', 'VEB', 'CTX-M-44', 'NDM-31', 'SFO-1', 'ROB-1', 'OXA-926', 'SHV', 'AFM-1', 'TEM-181', 'CTX-M-59', 'NDM-4', 'DHA-7', 'VIM', 'CTX-M-63', 'OXA-1041', 'CTX-M-125', 'CTX-M-123', 'OXA-1203', 'IMP-19', 'CTX-M-243', 'AFM-3', 'OXA-1204', 'bla-A2', 'LMB-1', 'GES-11', 'mecR1', 'OXA-1', 'BKC-1', 'FOX', 'CTX-M-136', 'OXA-58', 'NDM-1', 'CTX-M-32', 'TEM-171', 'KPC-6', 'KPC-49', 'CTX-M-30', 'CARB-16', 'KPC-93', 'L2', 'LAP-1', 'OXA-567', 'TEM-215', 'CMY-23', 'IMI', 'CMY-111', 'SHV-2A', 'RCP', 'OXA-19', 'OXA-436', 'VEB-18', 'OXA-237', 'NDM-19', 'CTX-M-17', 'IMI-16', 'CMY-6', 'CMY-172', 'VEB-5', 'KPC-109', 'FOX-5', 'IMP-100', 'OXA-655', 'PAU-1', 'TEM-21', 'OXA-96', 'VEB-16', 'VIM-19', 'IMP-56', 'OXA-2', 'SHV-30', 'CTX-M-25', 'SHV-28', 'IMP-45', 'IMP-26', 'CMY-148', 'OKP-B', 'VIM-2', 'CTX-M-40', 'KPC-204', 'OXA-932', 'TEM-4', 'mecA', 'OXA-420', 'KPC-121', 'NDM-5', 'NPS-1', 'KPC-3', 'TEM-12', 'ELC', 'KPC-113', 'MOX', 'OXA-164', 'HBL', 'PDC-16', 'CARB-2', 'OXA-653', 'PER-4', 'CTX-M-104', 'SHV-11', 'TEM-156', 'R39', 'PSV', 'GES-20', 'NDM-27', 'PEN-B', 'DIM-1', 'OXA-9', 'IMP-69', 'OXA-246', 'PER-1', 'VIM-7', 'OXA-198', 'CTX-M-173', 'TEM-61', 'OXA-101', 'TEM-34', 'CAE-1', 'MUN-1', 'NDM-29', 'TEM-3', 'MYO-1', 'SHV-7', 'OXA-97', 'VIM-84', 'OXA-438', 'CMY', 'VEB-2', 'KPC-33', 'GES-12', 'RAHN', 'NDM', 'IMP-14', 'VIM-11', 'IMP-63', 'CTX-M-226', 'VEB-8', 'CARB-8', 'CMY-97', 'ROB', 'CTX-M-53', 'KPC-40', 'OXA-244', 'SHV-5', 'mecB', 'ROB-11', 'KLUC-5', 'VIM-61', 'TEM-84', 'KPC-154', 'CMY-185', 'OXY-2-16', 'TEM-169', 'cfiA2', 'MCA', 'TEM-168', 'RSC1', 'IMP-23', 'CTX-M-62', 'OXA-732', 'CTX-M-195', 'NDM-9', 'CMY-166', 'VIM-85', 'ADC-30', 'NDM-7', 'TEM-116', 'TMB-1', 'VIM-86', 'CTX-M-174', 'bla-C', 'KPC-29', 'IMP-18', 'OXA-256', 'FRI-3', 'OXA-162', 'NPS', 'TEM-54', 'BIM-1', 'CTX-M-90', 'KPC-125', 'KPC-66', 'RAA-1', 'OXA-66', 'OXA-21', 'CMY-16', 'CTX-M-55', 'CMY-146', 'VEB-9', 'CTX-M-8', 'VHW', 'KPC-17', 'OXA-24', 'SHV-2', 'ADC-176', 'TEM-176', 'cdiA', 'CARB-4', 'KPC-4', 'KPC-14', 'GES-5', 'bla-A', 'pbp2m', 'OXA-232', 'VEB-3', 'CMY-36', 'TEM-20', 'CTX-M-39', 'VEB-25', 'CTX-M-65', 'IMP-11', 'ACC-1', 'OXA-181', 'OXY', 'mecI_of_mecA', 'NDM-17', 'SHV-31', 'KPC-78', 'CTX-M-5', 'IMP-38', 'CMY-44', 'CTX-M-134', 'LCR', 'GES-51', 'IMP-89', 'OXA-779', 'SHV-18', 'CMY-174', 'GIM-1', 'TER', 'GES-1', 'IMP-31', 'CMY-145', 'SHV-1', 'VIM-60', 'CTX-M-130', 'TEM-30', 'TEM-7', 'LAP-2', 'VIM-1', 'GES-44', 'CMY2-MIR-ACT-EC', 'LAP', 'CMY-2', 'RAHN-3', 'FRI-7', 'OXA-1391', 'OXA-82', 'FRI-4', 'SHV-12', 'bla-A_firm', 'CTX-M-64', 'OXA-209', 'OXA', 'DHA-15', 'BKC-2', 'IMI-23', 'TEM-24', 'bla-B1', 'R1', 'CTX-M-15', 'OXA-893', 'ADC', 'CMY-13', 'TEM-40', 'FRI-11', 'CTX-M-215', 'OXA-4', 'IMI-6', 'OXA-517', 'CMY-136', 'CTX-M-1', 'KPC-5', 'TEM-10', 'IMI-5', 'CTX-M-38', 'CTX-M-71', 'OXA-139', 'DHA-27', 'BRO', 'KPC-21', 'OXA-921', 'CMY-10', 'OXA-23', 'KHM-1', 'TEM-57', 'CTX-M-132', 'CTX-M-131', 'OXA-32', 'IMP-10', 'TEM-144', 'FRI-9', 'SCO-1', 'CAE', 'KPC-8', 'LCR-1', 'IMP-1', 'OXA-48', 'RTG', 'KPC-79', 'CMY-141', 'FRI-5', 'OXA-17', 'TEM-237', 'CTX-M-234', 'GES-14', 'NDM-13', 'VIM-27', 'CTX-M-27', 'NDM-36', 'KPC-112', 'KPC-111', 'NDM-14', 'GES-4', 'KPC-53', 'VEB-17', 'CARB-12', 'TEM-52', 'OXA-207', 'TEM-32', 'IMP-94', 'KPC-31', 'OXA-427', 'CTX-M-3', 'CTX-M', 'GES-6', 'SIM-2', 'OXA-520', 'OXA-897', 'bla1', 'VEB-1', 'PER-7', 'CTX-M-58', 'TEM-6', 'PSE', 'BES-1', 'CMY-178', 'NDM-6', 'BEL-1', 'Z', 'ADC-130', 'IMP-13', 'OXA-347', 'OXA-484', 'CTX-M-255', 'ACT-9', 'VIM-24', 'OXA-519', 'I', 'FRI-8', 'OXA-656', 'IMP', 'HER-3', 'PER-2', 'NDM-16b', 'CMY-5', 'OXA-29', 'IMP-64', 'IMP-6', 'TEM-256', 'VAM-1', 'CTX-M-236', 'PEN-bcc', 'ROB-2', 'KPC-84', 'TEM-37', 'bla-A_carba', 'CTX-M-102', 'ACC-4', 'VIM-66', 'FONA', 'CTX-M-14', 'KPC-18', 'OXA-1397', 'cfxA_fam', 'IMI-2', 'MOX-1', 'KPC-12', 'KPC-74', 'KPC-90', 'CTX-M-105', 'TEM-31', 'CTX-M-199', 'CTX-M-24', 'OXA-695', 'TEM-135', 'TEM-26', 'OXA-935', 'PER', 'TEM', 'IMP-96', 'IMP-8', 'NDM-23', 'KPC-67', 'PER-3', 'OXA-47', 'CTX-M-121', 'III', 'PSZ', 'KPC-189', 'CTX-M-98', 'CTX-M-101', 'OXA-1042', 'VCC-1', 'CMY-8', 'ACC', 'AFM-4', 'KPC-70', 'PEN-J', 'DIM', 'NDM-37', 'SHV-44', 'GES-24', 'PAU', 'VIM-23', 'IMI-22', 'OXA-392', 'KPC-110', 'KPC-2', 'CMY-31', 'TEM-33', 'MOX-18', 'VIM-4', 'OXA-10', 'KPC-71', 'KPC-44', 'GMA-1', 'OXA-235', 'CTX-M-37', 'CMY-42', 'OXA-204', 'DHA-1', 'FOX-7', 'VMB-1', 'TEM-210', 'OXA-796', 'PC1', 'OXA-900', 'GES-19', 'TEM-238', 'FRI-6', 'FLC-1', 'CTX-M-253', 'GES-7', 'SIM-1', 'TEM-206', 'IMP-4', 'KPC-87', 'FRI-2', 'OXA-72', 'KPC', 'SHV-102', 'OXA-1202', 'TLA-3', 'OXA-163', 'DHA', 'TEM-190', 'TEM-2', 'OXA-129', 'GES', 'VIM-6', 'KPC-24', 'PNC', 'AFM-2', 'KPC-55', 'PSZ-1', 'CTX-M-251', 'IMP-34']
new_mapped_names = ['NDM-3', 'NDM-11', 'NDM-20', 'NDM-21', 'VIM-5']
all_bl_mapped_names = prev_mapped_names + new_mapped_names 
test_new = test_new.loc[test_new['gene_name'].isin(all_bl_mapped_names)]
test_old = pd.read_csv('amrfindermapped_beta_lactamases_old.csv', low_memory=False)
test = pd.concat([test_old, test_new]).drop_duplicates(keep='first')

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



bl_df_merged = df_merged.filter(pl.col('target_name').str.contains('lactamase')|pl.col('target_name').str.contains('Lactamase')|pl.col('query_name').is_in(test['query_id'].tolist()))


pfam_beta_ids = list(set(list(bl_df_merged['query_name'])))

BETA_FASTA_PATH = Path('beta_lactam_fastas')

fasta_beta_ids = [x.split('.fa')[0] for x in os.listdir(BETA_FASTA_PATH)]




PID_pattern  = re.compile(r'^(.+?)_(\d+)_(\d+)')

all_beta_ids = list(set(pfam_beta_ids + fasta_beta_ids))

all_beta_ids_standardised = ['_'.join(x.split('_')[:-1]) if not PID_nogene_pattern.match(x) else x for x in all_beta_ids]

plasmid_to_takeouts = {}

for x in all_beta_ids_standardised:
    nuccore = PID_nogene_pattern.match(x)[1] if PID_nogene_pattern.match(x) else None
    if nuccore:
        plasmid_to_takeouts[nuccore] = []


for x in all_beta_ids_standardised:
    nuccore = PID_nogene_pattern.match(x)[1] if PID_nogene_pattern.match(x) else None
    start = PID_nogene_pattern.match(x)[2] if PID_nogene_pattern.match(x) else None
    stop = PID_nogene_pattern.match(x)[3] if PID_nogene_pattern.match(x) else None
    if nuccore and start and stop:
        plasmid_to_takeouts[nuccore].append((start, stop))


plasmid_no_beta_sequences_path = Path('plasmids_no_bl_seq')
os.makedirs(plasmid_no_beta_sequences_path, exist_ok=True)


def merge_intervals(intervals):
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for current in sorted_intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], current[1]))
        else:
            merged.append(current)
    return merged


import shutil

PLASMID_PATH = Path('plasmids')
non_bl_plasmids = [x.split('.fa')[0] for x in os.listdir(PLASMID_PATH) if x.split('.fa')[0] not in plasmid_to_takeouts.keys()]

for plasmid in non_bl_plasmids:
    plas_file_path = PLASMID_PATH / f'{plasmid}.fa'
    plas_out_path = plasmid_no_beta_sequences_path / f'{plasmid}.fa'
    shutil.copy(plas_file_path, plas_out_path)



for plasmid, sites in plasmid_to_takeouts.items():
    plasmid_fasta = PLASMID_PATH / f'{plasmid}.fa'
    output_fasta = plasmid_no_beta_sequences_path / f'{plasmid}.fa'
    if not plasmid_fasta.exists():
        print(f"Warning: {plasmid_fasta} not found. Skipping.")
        continue
    with open(plasmid_fasta, 'r') as f:
        header = f.readline().strip()
        seq = "".join(line.strip() for line in f)
    merged_sites = merge_intervals(sites)
    new_seq_parts = []
    current_idx = 0
    for start, stop in merged_sites:
        new_seq_parts.append(seq[int(current_idx):int(start)])
        current_idx = int(stop)
    new_seq_parts.append(seq[int(current_idx):])
    new_seq = "".join(new_seq_parts)
    with open(output_fasta, 'w') as f:
        f.write(f"{header}\n{new_seq}\n")



##now check for homology in this directory plasmids_no_bl_seq
all_species = list(set(list(nuc_spc.values())))


#
#███████╗ ██████╗ ██████╗     ██████╗ ██╗      █████╗ ███████╗███╗   ███╗██╗██████╗ ███████╗
#██╔════╝██╔═══██╗██╔══██╗    ██╔══██╗██║     ██╔══██╗██╔════╝████╗ ████║██║██╔══██╗██╔════╝
#█████╗  ██║   ██║██████╔╝    ██████╔╝██║     ███████║███████╗██╔████╔██║██║██║  ██║███████╗
#██╔══╝  ██║   ██║██╔══██╗    ██╔═══╝ ██║     ██╔══██║╚════██║██║╚██╔╝██║██║██║  ██║╚════██║
#██║     ╚██████╔╝██║  ██║    ██║     ███████╗██║  ██║███████║██║ ╚═╝ ██║██║██████╔╝███████║
#╚═╝      ╚═════╝ ╚═╝  ╚═╝    ╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═╝╚═════╝ ╚══════╝

##############################################################################################################
####################################
import re
import os
import sys
import signal
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


# =============================================================================
# PATHS & PARAMETERS
# =============================================================================

CRISPR_OUT_DIR        = Path('crispr_results')
PLASMID_DIR           = Path('plasmids_no_bl_seq')
OUT_DIR               = Path('homology_check_pam')
OUT_DIR.mkdir(exist_ok=True)

DB_PATH               = OUT_DIR / 'plasmid_db'
QUERY_FA              = OUT_DIR / 'guides.fa'
BLAST_OUT             = OUT_DIR / 'blast_hits_raw.tsv'

MAX_MISMATCHES        = 3      # hard ceiling; hits above this are dropped
MIN_MISMATCHES_FOR_OK = 2      # < this → HIGH risk  (0 or 1 mm = HIGH)
N_PROCESSES           = 4      # parallel BLAST workers


# =============================================================================
# GRACEFUL INTERRUPT
# Registers SIGINT handler in the parent; each worker subprocess is in its
# own process group so SIGTERM reaches it directly.
# =============================================================================

#_worker_pids = []   # populated when pool workers are submitted
_worker_pids = set()

def _sigint_handler(sig, frame):
    print('\n[interrupt] Terminating worker processes...', flush=True)
    for pid in _worker_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    sys.exit(1)

signal.signal(signal.SIGINT, _sigint_handler)


# =============================================================================
# PAM DEFINITIONS
# =============================================================================

IUPAC_RE = {
    'A':'A','C':'C','G':'G','T':'T',
    'N':'[ACGT]',
    'R':'[AG]','Y':'[CT]','S':'[GC]','W':'[AT]',
    'K':'[GT]','M':'[AC]',
    'B':'[CGT]','D':'[AGT]','H':'[ACT]','V':'[ACG]',
}

def pam_to_regex(pam_str):
    return re.compile(''.join(IUPAC_RE[c] for c in pam_str.upper()))

EDITOR_PAM = {
    'BE3':               'NGG',
    'Valdez_narrow_ABE': 'NGG',
    'CRISPR-cBEST':      'NGG',
    'CRISPR-aBEST':      'NGG',
    'BE4':               'NGG',
    'ABE8e':             'NGG',
    'VQR-BE3':           'NGAN',
    'EQR-BE3':           'NGAG',
    'VRER-BE3':          'NGCG',
    'SaBE3':             'NNGRRT',
    'SaKKH-BE3':         'NNNRRT',
}

EDITOR_PAM_RE  = {ed: pam_to_regex(pam) for ed, pam in EDITOR_PAM.items()}
EDITOR_PAM_LEN = {ed: len(pam)          for ed, pam in EDITOR_PAM.items()}


# =============================================================================
# VECTORISED PAM CHECK
# Works on a DataFrame slice; no Python-level row iteration.
# Returns two Series: pam_ok (bool), pam_seq (str).
# =============================================================================

def _extract_pam_series(plasmid_seqs, plasmid_ids, sstart_0, send_0,
                         sstrand, pam_len):
    """
    Extract PAM sequence strings as a pandas Series, operating on arrays.
    plasmid_ids, sstart_0, send_0, sstrand are all pandas Series of the same
    length; pam_len is a scalar (same PAM length for this editor batch).
    Returns a Series of extracted PAM strings (empty string where out-of-bounds).
    """
    pam_seqs = []
    for pid, ss, se, strand in zip(plasmid_ids, sstart_0, send_0, sstrand):
        seq = plasmid_seqs.get(pid, '')
        N   = len(seq)
        if not seq:
            pam_seqs.append('')
            continue
        if strand == 'plus':
            #ps = se + 1 # 0 or 1 based indexing?
            ps = se
            pe = ps + pam_len
            pam_seqs.append(seq[ps:pe] if pe <= N else '')
        else:
            #pe = ss           # exclusive on fwd strand
            pe = ss - 1  # 0 or 1 based indexing?
            ps = pe - pam_len
            if ps < 0:
                pam_seqs.append('')
            else:
                raw = seq[ps:pe]
                pam_seqs.append(str(Seq(raw).reverse_complement()))
    return pd.Series(pam_seqs, dtype='string')
    #return pd.array(pam_seqs, dtype='string')


def apply_pam_filter_vectorised(blast_df, plasmid_seqs, guide_to_editor):
    """
    Adds pam_ok (bool) and pam_seq (str) columns to blast_df.
    Processes one editor at a time to keep memory flat.
    Returns the filtered DataFrame (pam_ok == True rows only).
    """
    blast_df = blast_df.copy()
    blast_df['editor']  = blast_df['guide_id'].map(guide_to_editor)
    blast_df['pam_ok']  = False
    blast_df['pam_seq'] = ''

    for editor, grp_idx in blast_df.groupby('editor').groups.items():
        pam_len = EDITOR_PAM_LEN.get(editor)
        pam_re  = EDITOR_PAM_RE.get(editor)
        if pam_len is None or pam_re is None:
            continue

        grp = blast_df.loc[grp_idx]
        pam_seqs = _extract_pam_series(
            plasmid_seqs,
            grp['plasmid_id'].values,
            grp['sstart_0'].values,
            grp['send_0'].values,
            grp['sstrand'].values,
            pam_len,
        )
        # vectorised regex match — one call on the whole Series
        pam_ok = pam_seqs.str.fullmatch(
            ''.join(IUPAC_RE[c] for c in EDITOR_PAM[editor].upper())
        ).fillna(False)

        blast_df.loc[grp_idx, 'pam_ok']  = pam_ok.values
        blast_df.loc[grp_idx, 'pam_seq'] = pam_seqs.values

    return blast_df[blast_df['pam_ok']].copy()


# =============================================================================
# PARALLEL BLAST WORKER
# Each worker blasts one chunk of the query FASTA against the shared DB.
# Returns path to its output TSV.  Receives only paths (cheap to pickle).
# =============================================================================

def _blast_chunk(args):
    chunk_fa, db_path, out_tsv, threads = args
    cmd = [
        'blastn',
        '-task',          'blastn-short',
        '-query',         str(chunk_fa),
        '-db',            str(db_path),
        '-out',           str(out_tsv),
        '-outfmt',        '6 qseqid sseqid pident length mismatch qlen sstart send sstrand',
        '-evalue',        '0.01',
        '-perc_identity', '80',
        '-word_size',     '9',
        '-dust',          'no',
        '-num_threads',   str(threads),
        '-strand',        'both',
    ]
    proc = subprocess.Popen(cmd)
    return proc.pid, proc, out_tsv


BLAST_THREADS = 16

def run_blast_parallel(query_fa, db_path, out_tsv, n_processes):
    """
    Split query_fa into n_processes chunks, blast each in parallel,
    concatenate results into out_tsv.
    """
    # read all records
    records = list(SeqIO.parse(query_fa, 'fasta'))
    if not records:
        out_tsv.write_text('')
        return

    chunk_size = max(1, len(records) // n_processes + 1)
    chunks     = [records[i:i+chunk_size]
                  for i in range(0, len(records), chunk_size)]

    tmp_dir    = Path(tempfile.mkdtemp(dir=out_tsv.parent))
    chunk_fas  = []
    chunk_outs = []

    for i, chunk in enumerate(chunks):
        cfa = tmp_dir / f'chunk_{i}.fa'
        SeqIO.write(chunk, cfa, 'fasta')
        chunk_fas.append(cfa)
        chunk_outs.append(tmp_dir / f'chunk_{i}.tsv')

    # spawn workers; track PIDs for interrupt handler
    procs = []
    for cfa, cout in zip(chunk_fas, chunk_outs):
        # threads per worker: spread BLAST_THREADS evenly
        #t = max(1, 4 // n_processes)
        t = max(1, BLAST_THREADS // n_processes)
        cmd = [
            'blastn', '-task', 'blastn-short',
            '-query', str(cfa), '-db', str(db_path),
            '-out',   str(cout),
            '-outfmt', '6 qseqid sseqid pident length mismatch qlen sstart send sstrand',
            '-evalue', '0.01', '-perc_identity', '80',
            '-word_size', '9', '-dust', 'no',
            '-num_threads', str(t), '-strand', 'both',
        ]
        p = subprocess.Popen(cmd)
        #_worker_pids.append(p.pid)
        _worker_pids.add(p.pid)
        procs.append((p, cout))

    print(f'  {len(procs)} BLAST workers running (PIDs: {_worker_pids})...')

    # wait for all and collect
    for p, cout in procs:
        rc = p.wait()
        if rc not in (0, -15):   # -15 = SIGTERM on interrupt
            raise RuntimeError(f'blastn (pid {p.pid}) exited with code {rc}')
        _worker_pids.discard(p.pid) if hasattr(_worker_pids, 'discard') else None

    # concatenate chunk outputs
    with open(out_tsv, 'wb') as out:
        for _, cout in procs:
            if cout.exists():
                out.write(cout.read_bytes())

    # cleanup temp dir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f'  BLAST complete → {out_tsv}')


# =============================================================================
# STEP 1 – LOAD PLASMID SEQUENCES  (for PAM checking only — stays in parent)
# =============================================================================

print('Loading plasmid sequences for PAM checking...')
plasmid_seqs = {}
for fa in sorted(PLASMID_DIR.glob('*.fa')):
    for rec in SeqIO.parse(fa, 'fasta'):
        if len(rec.seq) >= 100:
            plasmid_seqs[rec.id] = str(rec.seq).upper()
print(f'Loaded {len(plasmid_seqs):,} plasmid sequences')


# =============================================================================
# STEP 2 – BUILD BLAST DATABASE
# =============================================================================

if not Path(str(DB_PATH) + '.nhr').exists():
    print('Building BLAST database...')
    combined = OUT_DIR / 'plasmids_combined.fa'
    with open(combined, 'w') as out:
        for pid, seq in plasmid_seqs.items():
            SeqIO.write(SeqRecord(Seq(seq), id=pid, description=''), out, 'fasta')
    subprocess.run([
        'makeblastdb', '-in', str(combined),
        '-dbtype', 'nucl', '-out', str(DB_PATH),
    ], check=True)
    print('  DB built.')
else:
    print(f'BLAST DB exists: {DB_PATH}')


# =============================================================================
# STEP 3 – LOAD CANDIDATES & WRITE QUERY FASTA
# =============================================================================

safe = pd.read_csv(CRISPR_OUT_DIR / 'candidates_safe.csv')

guide_cols    = ['editor', 'editor_type', 'protospacer', 'strand']
unique_guides = safe[guide_cols].drop_duplicates().reset_index(drop=True).copy()
unique_guides['guide_id'] = [f'guide_{i:06d}' for i in range(len(unique_guides))]

records = [
    SeqRecord(
        Seq(row['protospacer'].upper().replace('U', 'T')),
        id=row['guide_id'], description=''
    )
    for _, row in unique_guides.iterrows()
]
SeqIO.write(records, QUERY_FA, 'fasta')
print(f'Wrote {len(records):,} query guides → {QUERY_FA}')


# =============================================================================
# STEP 4 – RUN BLASTN (parallel chunks)
# =============================================================================

if not BLAST_OUT.exists():
    print(f'Running BLAST ({N_PROCESSES} parallel workers)...')
    run_blast_parallel(QUERY_FA, DB_PATH, BLAST_OUT, N_PROCESSES)
else:
    print(f'Using cached BLAST output: {BLAST_OUT}')


# =============================================================================
# STEP 5 – PARSE, MISMATCH FILTER, VECTORISED PAM CHECK
# =============================================================================

blast_raw = pd.read_csv(
    BLAST_OUT, sep='\t', header=None,
    names=['guide_id', 'plasmid_id', 'pident', 'length',
           'mismatch', 'qlen', 'sstart', 'send', 'sstrand'],
    dtype={'guide_id': 'string', 'plasmid_id': 'string',
           'sstrand': 'string'},
)

blast_raw = blast_raw[
    (blast_raw['length'] >= blast_raw['qlen'] - 1) &
    (blast_raw['mismatch'] <= MAX_MISMATCHES)
].copy()
print(f'Hits after mismatch filter (≤{MAX_MISMATCHES} mm): {len(blast_raw):,}')


coords_min = blast_raw[['sstart','send']].min(axis=1).astype(int)
coords_max = blast_raw[['sstart','send']].max(axis=1).astype(int)

#think the coords were as is for python?

blast_raw['sstart_0'] = coords_min
blast_raw['send_0']   = coords_max

guide_to_editor = unique_guides.set_index('guide_id')['editor'].to_dict()

print('Applying per-editor PAM filter (vectorised)...')
blast = apply_pam_filter_vectorised(blast_raw, plasmid_seqs, guide_to_editor)
print(f'Hits after PAM filter: {len(blast):,}')

blast['risk_tier'] = (blast['mismatch'] < MIN_MISMATCHES_FOR_OK).map(
    {True: 'HIGH', False: 'MEDIUM'}
)

# free raw table — no longer needed
del blast_raw


# =============================================================================
# STEP 6 – SUMMARISE PER GUIDE
# =============================================================================

def summarise(df):
    best = df.loc[df['mismatch'].idxmin()]
    return pd.Series({
        'n_offtarget_hits':   len(df),
        'n_high_risk_hits':   (df['risk_tier'] == 'HIGH').sum(),
        'n_medium_risk_hits': (df['risk_tier'] == 'MEDIUM').sum(),
        'n_plasmids_hit':     df['plasmid_id'].nunique(),
        'min_mismatch':       int(df['mismatch'].min()),
        'max_pident':         df['pident'].max(),
        'worst_hit_plasmid':  best['plasmid_id'],
        'worst_hit_sstart_0': int(best['sstart_0']),
        'worst_hit_send_0':   int(best['send_0']),
        'worst_hit_strand':   best['sstrand'],
        'worst_hit_pam':      best['pam_seq'],
    })

per_guide = (blast.groupby('guide_id')
                  .apply(summarise, include_groups=False)
                  .reset_index())


# =============================================================================
# STEP 7 – MERGE BACK
# =============================================================================

unique_guides_hom = unique_guides.merge(per_guide, on='guide_id', how='left')

fill_int = ['n_offtarget_hits', 'n_high_risk_hits', 'n_medium_risk_hits', 'n_plasmids_hit']
unique_guides_hom[fill_int] = unique_guides_hom[fill_int].fillna(0).astype(int)
unique_guides_hom['min_mismatch'] = unique_guides_hom['min_mismatch'].fillna(999)
unique_guides_hom['max_pident']   = unique_guides_hom['max_pident'].fillna(0)
#!!!MAY HAVE TO REDEFINE
unique_guides_hom['has_high_risk_hit'] = unique_guides_hom['n_high_risk_hits'] > 0
#unique_guides_hom['has_high_risk_hit'] = (
#    (unique_guides_hom['n_high_risk_hits'] > 0) &
#    (unique_guides_hom['n_plasmids_hit'] > 1)
#)

print(f'Guides HIGH-risk: {unique_guides_hom["has_high_risk_hit"].sum():,}  '
      f'clean: {(~unique_guides_hom["has_high_risk_hit"]).sum():,}')

merge_key = ['editor', 'protospacer', 'strand']

safe_hom = safe.merge(
    unique_guides_hom[merge_key + [
        'guide_id', 'n_offtarget_hits', 'n_high_risk_hits',
        'n_medium_risk_hits', 'n_plasmids_hit',
        'min_mismatch', 'max_pident', 'has_high_risk_hit',
        'worst_hit_plasmid', 'worst_hit_sstart_0', 'worst_hit_send_0',
        'worst_hit_strand', 'worst_hit_pam',
    ]],
    on=merge_key, how='left'
)

safe_hom['homology_clean'] = ~safe_hom['has_high_risk_hit'].fillna(False)
safe_hom['rank_score']     = (
    safe_hom['efficiency_score'] * (1 - safe_hom['pct_early']) *
    safe_hom['pct_conserved_dn'].fillna(0.5)
)
safe_hom        = safe_hom.sort_values('rank_score', ascending=False)
safe_hom_strict = safe_hom[safe_hom['homology_clean']].copy()

print(f'Safe candidates (original):  {len(safe):,}')
print(f'Safe + homology-clean:       {len(safe_hom_strict):,}')


# =============================================================================
# STEP 8 – TARGET LOSS REPORT
# =============================================================================

all_input_pids  = set(safe['query_id'].unique())
pids_with_clean = set(safe_hom_strict['query_id'].unique())
pids_fully_lost = all_input_pids - pids_with_clean

lost_guide_info = (
    safe_hom[
        safe_hom['query_id'].isin(pids_fully_lost) &
        ~safe_hom['homology_clean']
    ][[
        'query_id', 'gene_name', 'family', 'editor', 'protospacer',
        'n_high_risk_hits', 'n_plasmids_hit', 'min_mismatch',
        'worst_hit_plasmid', 'worst_hit_sstart_0', 'worst_hit_send_0',
        'worst_hit_strand', 'worst_hit_pam',
    ]]
    .drop_duplicates()
)

target_loss = (
    lost_guide_info
    .sort_values('min_mismatch')
    .groupby('query_id', sort=False)
    .first()
    .reset_index()
)

print(f'\n{"="*60}')
print('TARGET LOSS REPORT')
print(f'{"="*60}')
print(f'Total input targets:                    {len(all_input_pids):,}')
print(f'Targets with ≥1 homology-clean guide:   {len(pids_with_clean):,}')
print(f'Targets LOST (no clean guide remains):  {len(pids_fully_lost):,}  '
      f'({len(pids_fully_lost)/max(len(all_input_pids),1)*100:.1f}%)')

if len(pids_fully_lost):
    print('\nLoss by family:')
    for fam, cnt in target_loss['family'].value_counts().items():
        print(f'  {str(fam):<25s} {cnt:>5,}')
    print('\nLoss by editor:')
    for ed, cnt in target_loss['editor'].value_counts().items():
        print(f'  {str(ed):<25s} {cnt:>5,}')
    print('\nMismatch distribution of worst hits:')
    print(target_loss['min_mismatch'].value_counts().sort_index().to_string())

print(f'{"="*60}')


# =============================================================================
# STEP 9 – GREEDY MINIMAL sgRNA SET
# =============================================================================

def greedy_minimal_set(candidates_df):
    guide_cov = (
        candidates_df
        .groupby(['editor', 'protospacer', 'strand'])
        .agg(
            covered            = ('query_id',          lambda x: frozenset(x)),
            n_pids             = ('query_id',          'nunique'),
            mean_eff           = ('efficiency_score',  'mean'),
            mean_early         = ('pct_early',         'mean'),
            mean_cons          = ('pct_conserved_dn',  'mean'),
            family             = ('family',            'first'),
            editor_type        = ('editor_type',       'first'),
            n_offtarget_hits   = ('n_offtarget_hits',  'first'),
            n_high_risk_hits   = ('n_high_risk_hits',  'first'),
            n_plasmids_hit     = ('n_plasmids_hit',    'first'),
            min_mismatch       = ('min_mismatch',      'first'),
            worst_hit_plasmid  = ('worst_hit_plasmid', 'first'),
            worst_hit_sstart_0 = ('worst_hit_sstart_0','first'),
            worst_hit_send_0   = ('worst_hit_send_0',  'first'),
        )
        .reset_index()
        .sort_values(['n_pids', 'mean_eff', 'mean_early'],
                     ascending=[False, False, True])
    )
    uncovered = set(candidates_df['query_id'].unique())
    selected  = []
    while uncovered:
        best_new, best_row = frozenset(), None
        for _, r in guide_cov.iterrows():
            new = r['covered'] & uncovered
            if len(new) > len(best_new):
                best_new, best_row = new, r
        if best_row is None:
            break
        selected.append({
            'editor':              best_row['editor'],
            'editor_type':         best_row['editor_type'],
            'protospacer':         best_row['protospacer'],
            'strand':              best_row['strand'],
            'family':              best_row['family'],
            'n_pids_covered':      len(best_new),
            'mean_efficiency':     round(best_row['mean_eff'],   3),
            'mean_pct_early':      round(best_row['mean_early'], 3),
            'mean_pct_cons_dn':    round(best_row['mean_cons'],  3) if pd.notna(best_row['mean_cons']) else None,
            'n_offtarget_hits':    best_row['n_offtarget_hits'],
            'n_high_risk_hits':    best_row['n_high_risk_hits'],
            'n_plasmids_hit':      best_row['n_plasmids_hit'],
            'min_mismatch':        best_row['min_mismatch'],
            'worst_hit_plasmid':   best_row['worst_hit_plasmid'],
            'worst_hit_sstart_0':  best_row['worst_hit_sstart_0'],
            'worst_hit_send_0':    best_row['worst_hit_send_0'],
            'covers':              sorted(best_new),
        })
        uncovered -= best_new
    return pd.DataFrame(selected)


sgrna_set_hom = greedy_minimal_set(safe_hom_strict)

print(f'\nMinimal sgRNA set (homology-filtered): {len(sgrna_set_hom)} guide(s)')
if not sgrna_set_hom.empty:
    print(sgrna_set_hom[[
        'editor', 'protospacer', 'strand', 'n_pids_covered',
        'mean_efficiency', 'n_high_risk_hits', 'n_plasmids_hit', 'min_mismatch'
    ]].to_string(index=False))


# =============================================================================
# STEP 10 – SAVE
# =============================================================================

safe_hom.to_csv(         OUT_DIR / 'candidates_safe_homology_annotated.csv', index=False)
safe_hom_strict.to_csv(  OUT_DIR / 'candidates_safe_homology_clean.csv',      index=False)
sgrna_set_hom.to_csv(    OUT_DIR / 'sgrna_minimal_set_homology_filtered.csv', index=False)
unique_guides_hom.to_csv(OUT_DIR / 'unique_guides_homology_summary.csv',      index=False)
blast.to_csv(            OUT_DIR / 'blast_hits_pam_filtered.csv',             index=False)
target_loss.to_csv(      OUT_DIR / 'target_loss_report.csv',                  index=False)

print(f'\nAll outputs → {OUT_DIR}/')




greg = blast.loc[blast['risk_tier'] == 'HIGH'].copy()
print(f'HIGH-risk hits: {len(greg):,}')
print(f'Unique guides with HIGH-risk hits: {greg["guide_id"].nunique():,}')
print(f'Unique plasmids hit (HIGH-risk): {greg["plasmid_id"].nunique():,}')

original_targets  = set(safe['query_id'])
remaining_targets = set(safe_hom_strict['query_id'])

lost_targets = original_targets - remaining_targets

print(f"Targets total:     {len(original_targets):,}")
print(f"Targets remaining: {len(remaining_targets):,}")
print(f"Targets lost:      {len(lost_targets):,}")
print(f"Fraction lost:     {len(lost_targets)/len(original_targets):.2%}")
#
#Targets total:     44,429
#Targets remaining: 44,031
#Targets lost:      398
#Fraction lost:     0.90%
#>>>

##########################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################


#███████╗ ██████╗ ██████╗     ██╗  ██╗ ██████╗ ███████╗████████╗███████╗
#██╔════╝██╔═══██╗██╔══██╗    ██║  ██║██╔═══██╗██╔════╝╚══██╔══╝██╔════╝
#█████╗  ██║   ██║██████╔╝    ███████║██║   ██║███████╗   ██║   ███████╗
#██╔══╝  ██║   ██║██╔══██╗    ██╔══██║██║   ██║╚════██║   ██║   ╚════██║
#██║     ╚██████╔╝██║  ██║    ██║  ██║╚██████╔╝███████║   ██║   ███████║
#╚═╝      ╚═════╝ ╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝


species_list_i_think = [
    'Streptococcus_pasteurianus', 'Mycetohabitans_endofungorum', 'Vibrio_fluvialis', 'Priestia_flexa', 
    'Streptomyces_decoyicus', 'Empedobacter_falsenii', 'Citrobacter_tructae', 'Loigolactobacillus_backii', 
    'Pusillibacter_faecalis', 'Piscirickettsia_salmonis', 'Klebsiella_pasteurii', 'Staphylococcus_petrasii', 
    'Lachnospira_eligens', 'Fusobacterium_vincentii', 'Lactococcus_lactis', 'Citrobacter_youngae', 
    'Bacillus_intermedius', 'Rubrobacter_marinus', 'Campylobacter_coli', 'Pseudomonas_paraeruginosa', 
    'Streptomyces_globosus', 'Providencia_hangzhouensis', 'Bordetella_bronchiseptica', 'Chryseobacterium_gallinarum', 
    'Neorhizobium_petrolearium', 'Streptomyces_californicus', 'Acinetobacter_johnsonii', 'Caballeronia_insecticola', 
    'Nonomuraea_gerenzanensis', 'Mannheimia_pernigra', 'Rhodococcus_gordoniae', 'Aureibacter_tunicatorum', 
    'Massilia_forsythiae', 'Pseudoalteromonas_piscicida', 'Aerococcus_viridans', 'Azospirillum_thermophilum', 
    'Rahnella_aceris', 'Alkalihalophilus_pseudofirmus', 'Novosphingobium_aromaticivorans', 'Aeromonas_veronii', 
    'Gemmobacter_aquarius', 'Stutzerimonas_zhaodongensis', 'Thauera_aromatica', 'Streptococcus_salivarius', 
    'Rahnella_sikkimica', 'Acinetobacter_venetianus', 'Citrobacter_amalonaticus', 'Staphylococcus_cohnii', 
    'Pseudomonas_monteilii', 'Burkholderia_gladioli', 'Vibrio_tubiashii', 'Streptococcus_infantis', 
    'Pseudomonas_koreensis', 'Brevibacillus_laterosporus', 'Cupriavidus_gilardii', 'Sphingobacterium_multivorum', 
    'Acinetobacter_gerneri', 'Xanthomonas_vesicatoria', 'Ideonella_dechloratans', 'Pseudomonas_aeruginosa', 
    'Limosilactobacillus_gastricus', 'Pseudomonas_putida', 'Mammaliicoccus_sciuri', 'Micrococcus_luteus', 
    'Rhodococcus_pyridinivorans', 'Desulforapulum_autotrophicum', 'Thermus_thermophilus', 'Bacillus_thuringiensis', 
    'Bacillus_albus', 'Streptococcus_agalactiae', 'Pseudomonas_kurunegalensis', 'Shewanella_putrefaciens', 
    'Staphylococcus_haemolyticus', 'Macrococcus_armenti', 'Cellulosimicrobium_cellulans', 'Lacticaseibacillus_paracasei', 
    'Lactococcus_cremoris', 'Shewanella_aestuarii', 'Haloferax_mediterranei', 'Methylobacterium_bullatum', 
    'Cronobacter_turicensis', 'Staphylococcus_lugdunensis', 'Bacillus_toyonensis', 'Legionella_anisa', 
    'Mesorhizobium_amorphae', 'Corynebacterium_striatum', 'Enterobacter_cloacae', 'Acinetobacter_colistiniresistens', 
    'Rubrobacter_tropicus', 'Vibrio_anguillarum', 'Staphylococcus_argenteus', 'Pantoea_anthophila', 
    'Acetobacter_pasteurianus', 'Clostridium_estertheticum', 'Rhizobium_esperanzae', 'Pantoea_stewartii', 
    'Paraburkholderia_terrae', 'Halobacterium_salinarum', 'Klebsiella_oxytoca', 'Nitratidesulfovibrio_vulgaris', 
    'Xanthomonas_arboricola', 'Acinetobacter_piscicola', 'Phyllobacterium_zundukense', 'Acinetobacter_towneri', 
    'Leptospira_interrogans', 'Croceicoccus_marinus', 'Bacteroides_uniformis', 'Escherichia_ruysiae', 
    'Serratia_ureilytica', 'Paracoccus_pantotrophus', 'Legionella_longbeachae', 'Brochothrix_thermosphacta', 
    'Pandoraea_pnomenusa', 'Enterococcus_faecium', 'Phocaeicola_massiliensis', 'Pseudomonas_fragi', 
    'Sphingomonas_naphthae', 'Hafnia_paralvei', 'Enterobacter_chengduensis', 'Shinella_zoogloeoides', 
    'Enterococcus_hirae', 'Yersinia_enterocolitica', 'Variovorax_paradoxus', 'Agrobacterium_larrymoorei', 
    'Shewanella_xiamenensis', 'Agrobacterium_leguminum', 'Streptomyces_cellulosae', 'Bacteroides_xylanisolvens', 
    'Deinococcus_aetherius', 'Xanthomonas_euvesicatoria', 'Shigella_sonnei', 'Clostridioides_difficile', 
    'Paenibacillus_amylolyticus', 'Winkia_neuii', 'Acinetobacter_bereziniae', 'Aliarcobacter_cryaerophilus', 
    'Enterobacter_huaxiensis', 'Citrobacter_arsenatis', 'Staphylococcus_delphini', 'Comamonas_testosteroni', 
    'Raoultella_planticola', 'Halogeometricum_borinquense', 'Azotobacter_vinelandii', 'Corynebacterium_crudilactis', 
    'Brevundimonas_nasdae', 'Vibrio_fortis', 'Limosilactobacillus_mucosae', 'Trabulsiella_odontotermitis', 
    'Brevibacterium_spongiae', 'Flagellimonas_marinaquae', 'Hafnia_alvei', 'Pseudomonas_silvicola', 
    'Lactococcus_raffinolactis', 'Bradyrhizobium_japonicum', 'Acinetobacter_radioresistens', 'Stutzerimonas_stutzeri', 
    'Achromobacter_ruhlandii', 'Corynebacterium_jeikeium', 'Bacillus_anthracis', 'Lactobacillus_amylovorus', 
    'Staphylococcus_xylosus', 'Lelliottia_amnigena', 'Citrobacter_europaeus', 'Pseudoalteromonas_shioyasakiensis', 
    'Staphylococcus_aureus', 'Actinobacillus_pleuropneumoniae', 'Cutibacterium_granulosum', 'Achromobacter_xylosoxidans', 
    'Aminobacter_niigataensis', 'Cupriavidus_necator', 'Sphingobium_xenophagum', 'Aeromonas_allosaccharophila', 
    'Natrinema_versiforme', 'Halobaculum_halophilum', 'Streptococcus_iniae', 'Atlantibacter_subterranea', 
    'Acinetobacter_wuhouensis', 'Enterobacter_kobei', 'Clostridium_beijerinckii', 'Citrobacter_werkmanii', 
    'Ligilactobacillus_salivarius', 'Paracoccus_versutus', 'Cytobacillus_spongiae', 'Listeria_welshimeri', 
    'Priestia_megaterium', 'Clostridium_felsineum', 'Bacillus_paranthracis', 'Streptomyces_fungicidicus', 
    'Citrobacter_portucalensis', 'Plesiomonas_shigelloides', 'Alcaligenes_faecalis', 'Raoultella_ornithinolytica', 
    'Leptospira_weilii', 'Novosphingobium_resinovorum', 'Clostridium_argentinense', 'Staphylococcus_warneri', 
    'Microvirga_terrae', 'Acinetobacter_ursingii', 'Metabacillus_dongyingensis', 'Corynebacterium_faecale', 
    'Bordetella_pertussis', 'Secundilactobacillus_malefermentans', 'Vagococcus_lutrae', 'Sinorhizobium_kummerowiae', 
    'Lactococcus_garvieae', 'Phaeobacter_piscinae', 'Oligella_ureolytica', 'Empedobacter_brevis', 
    'Paenibacillus_cellulosilyticus', 'Lactococcus_formosensis', 'Halococcus_dombrowskii', 'Pluralibacter_gergoviae', 
    'Enterococcus_gilvus', 'Sagittula_stellata', 'Serratia_myotis', 'Dietzia_kunjamensis', 'Ralstonia_solanacearum', 
    'Streptomyces_chartreusis', 'Riemerella_anatipestifer', 'Rhizobium_rhizoryzae', 'Neisseria_gonorrhoeae', 
    'Sinorhizobium_chiapasense', 'Vibrio_penaeicida', 'Citrobacter_braakii', 'Butyricimonas_faecalis', 
    'Streptomyces_albidoflavus', 'Bacteroides_fragilis', 'Dolichospermum_flos-aquae', 'Paraclostridium_sordellii', 
    'Methylocystis_rosea', 'Streptomyces_nigrescens', 'Paenalcaligenes_faecalis', 'Empedobacter_stercoris', 
    'Moraxella_osloensis', 'Corynebacterium_glutamicum', 'Shigella_dysenteriae', 'Chondrinema_litorale', 
    'Halorussus_salilacus', 'Ruminococcus_albus', 'Lactiplantibacillus_pentosus', 'Rhizobium_phaseoli', 
    'Paracidovorax_avenae', 'Aeromonas_dhakensis', 'Sulfitobacter_dubius', 'Lysinibacillus_fusiformis', 
    'Pantoea_piersonii', 'Shinella_oryzae', 'Vibrio_splendidus', 'Croceicoccus_naphthovorans', 'Brucella_anthropi', 
    'Haemophilus_influenzae', 'Latilactobacillus_curvatus', 'Streptomyces_violaceusniger', 'Myroides_albus', 
    'Nitrincola_iocasae', 'Stutzerimonas_decontaminans', 'Cupriavidus_basilensis', 'Bifidobacterium_choerinum', 
    'Ciceribacter_thiooxidans', 'Novosphingobium_humi', 'Shewanella_algae', 'Pantoea_ananatis', 
    'Paraclostridium_bifermentans', 'Citrobacter_freundii', 'Geobacillus_stearothermophilus', 'Gluconacetobacter_diazotrophicus', 
    'Caballeronia_grimmiae', 'Aeromonas_bestiarum', 'Vibrio_diabolicus', 'Acinetobacter_lwoffii', 'Vibrio_harveyi', 
    'Sphingobacterium_faecium', 'Pediococcus_acidilactici', 'Acinetobacter_soli', 'Haloarcula_halophila', 
    'Staphylococcus_arlettae', 'Edwardsiella_tarda', 'Streptococcus_equinus', 'Paenibacillus_larvae', 
    'Glaesserella_parasuis', 'Prescottella_equi', 'Pantoea_soli', 'Priestia_aryabhattai', 'Azotobacter_salinestris', 
    'Aliivibrio_wodanis', 'Staphylococcus_hyicus', 'Burkholderia_anthina', 'Bacillus_luti', 'Natronosalvus_halobius', 
    'Streptomyces_clavuligerus', 'Vescimonas_fastidiosa', 'Apirhabdus_apintestini', 'Acinetobacter_calcoaceticus', 
    'Staphylococcus_pettenkoferi', 'Leclercia_adecarboxylata', 'Enterobacter_asburiae', 'Rhizobium_daejeonense', 
    'Bacteroides_faecis', 'Sphingomonas_parapaucimobilis', 'Pseudomonas_syringae', 'Vibrio_nigripulchritudo', 
    'Edwardsiella_anguillarum', 'Streptomyces_camelliae', 'Clostridium_perfringens', 'Aeromonas_sobria', 
    'Pantoea_agglomerans', 'Mammaliicoccus_lentus', 'Pseudomonas_shirazica', 'Pantoea_vagans', 'Listeria_monocytogenes', 
    'Vitreoscilla_filiformis', 'Sulfuricurvum_kujiense', 'Enterococcus_dispar', 'Haloferax_alexandrinus', 
    'Pseudomonas_juntendi', 'Ralstonia_wenshanensis', 'Leifsonia_shinshuensis', 'Persicobacter_psychrovividus', 
    'Vibrio_alfacsensis', 'Enterococcus_gallinarum', 'Pseudomonas_mandelii', 'Ensifer_adhaerens', 
    'Ralstonia_pseudosolanacearum', 'Streptococcus_mutans', 'Aliarcobacter_cibarius', 'Enterococcus_casseliflavus', 
    'Nitrosomonas_eutropha', 'Mycobacteroides_chelonae', 'Pediococcus_inopinatus', 'Rhizobium_gallicum', 
    'Yersinia_massiliensis', 'Achromobacter_insolitus', 'Acinetobacter_defluvii', 'Methylorubrum_extorquens', 
    'Burkholderia_cepacia', 'Klebsiella_quasipneumoniae', 'Klebsiella_aerogenes', 'Clostridium_baratii', 
    'Staphylococcus_pasteuri', 'Paracoccus_denitrificans', 'Dermacoccus_abyssi', 'Sporosarcina_ureae', 
    'Phytobacter_ursingii', 'Serratia_marcescens', 'Agrobacterium_fabrum', 'Niallia_taxi', 'Klebsiella_quasivariicola', 
    'Vagococcus_fluvialis', 'Natrinema_thermotolerans', 'Vagococcus_carniphilus', 'Enterobacter_ludwigii', 
    'Exiguobacterium_aurantiacum', 'Rhodococcus_opacus', 'Agrobacterium_tumefaciens', 'Vibrio_alginolyticus', 
    'Pseudochrobactrum_algeriensis', 'Bacillus_bombysepticus', 'Bacillus_subtilis', 'Paraburkholderia_caledonica', 
    'Lysinibacillus_capsici', 'Kosakonia_cowanii', 'Comamonas_antarctica', 'Morganella_morganii', 
    'Deinococcus_wulumuqiensis', 'Proteus_terrae', 'Pseudomonas_psychrotolerans', 'Limosilactobacillus_oris', 
    'Duffyella_gerundensis', 'Cupriavidus_pauculus', 'Serratia_proteamaculans', 'Phaeobacter_inhibens', 
    'Staphylococcus_schleiferi', 'Heyndrickxia_oleronia', 'Xanthomonas_sacchari', 'Aeromonas_salmonicida', 
    'Latilactobacillus_sakei', 'Cutibacterium_modestum', 'Bacillus_arachidis', 'Streptantibioticus_cattleyicolor', 
    'Azospirillum_ramasamyi', 'Mixta_calida', 'Pantoea_deleyi', 'Proteus_penneri', 'Rhodococcus_erythropolis', 
    'Citrobacter_gillenii', 'Klebsiella_pneumoniae', 'Providencia_alcalifaciens', 'Companilactobacillus_farciminis', 
    'Cronobacter_dublinensis', 'Klebsiella_grimontii', 'Burkholderia_arboris', 'Escherichia_coli', 
    'Anoxybacillus_amylolyticus', 'Haemophilus_parainfluenzae', 'Kosakonia_radicincitans', 'Actinobacillus_porcitonsillarum', 
    'Bhargavaea_cecembensis', 'Mycolicibacterium_aubagnense', 'Thioclava_nitratireducens', 'Acinetobacter_variabilis', 
    'Staphylococcus_rostri', 'Edwardsiella_ictaluri', 'Deinococcus_aquaticus', 'Staphylococcus_hsinchuensis', 
    'Cutibacterium_acnes', 'Legionella_sainthelensi', 'Spirosoma_oryzicola', 'Acinetobacter_indicus', 
    'Pseudomonas_asiatica', 'Zymomonas_mobilis', 'Avibacterium_paragallinarum', 'Burkholderia_pseudomallei', 
    'Rhizobium_rhizogenes', 'Rhodococcus_oxybenzonivorans', 'Exiguobacterium_arabatum', 'Burkholderia_cenocepacia', 
    'Proteus_mirabilis', 'Klebsiella_electrica', 'Bacteroides_thetaiotaomicron', 'Azospirillum_argentinense', 
    'Lactiplantibacillus_plantarum', 'Leptolyngbya_boryana', 'Pseudescherichia_vulneris', 'Pseudocitrobacter_faecalis', 
    'Bacillus_velezensis', 'Clostridium_sporogenes', 'Novosphingobium_pentaromativorans', 'Tsukamurella_tyrosinosolvens', 
    'Paraburkholderia_graminis', 'Aeromonas_hydrophila', 'Shinella_sumterensis', 'Rhodococcus_qingshengii', 
    'Deinococcus_radiodurans', 'Cupriavidus_campinensis', 'Cereibacter_sphaeroides', 'Microvirga_lotononidis', 
    'Streptomyces_goshikiensis', 'Enterobacter_chuandaensis', 'Niveispirillum_cyanobacteriorum', 'Enterococcus_saigonensis', 
    'Clostridium_novyi', 'Paenibacillus_urinalis', 'Sphingobium_herbicidovorans', 'Raoultella_terrigena', 
    'Stenotrophomonas_rhizophila', 'Pseudomonas_rhodesiae', 'Sinorhizobium_mexicanum', 'Macrococcus_equipercicus', 
    'Shewanella_bicestrii', 'Enterobacter_mori', 'Acinetobacter_schindleri', 'Blautia_hydrogenotrophica', 
    'Streptomyces_buecherae', 'Agrobacterium_salinitolerans', 'Pseudomonas_mosselii', 'Vibrio_cholerae', 
    'Citrobacter_pasteurii', 'Vibrio_cyclitrophicus', 'Enterococcus_avium', 'Paenarthrobacter_nicotinovorans', 
    'Streptococcus_suis', 'Paraburkholderia_largidicola', 'Paenibacillus_thiaminolyticus', 'Campylobacter_iguaniorum', 
    'Cronobacter_malonaticus', 'Psychromicrobium_lacuslunae', 'Vibrio_aquimaris', 'Nitrosomonas_europaea', 
    'Rhizobium_laguerreae', 'Pseudomonas_mendocina', 'Streptococcus_gallolyticus', 'Bradyrhizobium_barranii', 
    'Pseudonocardia_autotrophica', 'Acinetobacter_proteolyticus', 'Comamonas_aquatica', 'Enterococcus_durans', 
    'Bacillus_paramobilis', 'Mannheimia_varigena', 'Agrobacterium_vitis', 'Monoglobus_pectinilyticus', 
    'Sulfitobacter_faviae', 'Serratia_fonticola', 'Azotobacter_chroococcum', 'Ralstonia_nicotianae', 
    'Burkholderia_multivorans', 'Cupriavidus_pinatubonensis', 'Streptomyces_griseorubiginosus', 'Enterococcus_raffinosus', 
    'Xanthomonas_hortorum', 'Bradyrhizobium_septentrionale', 'Deinococcus_radiophilus', 'Proteus_vulgaris', 
    'Kluyvera_cryocrescens', 'Ralstonia_insidiosa', 'Staphylococcus_nepalensis', 'Staphylococcus_caprae', 
    'Yersinia_hibernica', 'Sulfitobacter_pontiacus', 'Rhizobium_lusitanum', 'Clostridium_tetani', 'Streptomyces_virginiae', 
    'Moellerella_wisconsensis', 'Pseudomonas_paralcaligenes', 'Geobacillus_subterraneus', 'Listeria_grayi', 
    'Klebsiella_michiganensis', 'Pantoea_dispersa', 'Photobacterium_damselae', 'Shigella_flexneri', 'Shigella_boydii', 
    'Citrobacter_telavivensis', 'Citrobacter_farmeri', 'Halostagnicola_larsenii', 'Desulfobaculum_bizertense', 
    'Azospirillum_brasilense', 'Leptotrichia_wadei', 'Limosilactobacillus_fermentum', 'Exiguobacterium_acetylicum', 
    'Sphingobium_yanoikuyae', 'Pediococcus_damnosus', 'Haladaptatus_caseinilyticus', 'Macrococcoides_bohemicum', 
    'Paraburkholderia_caribensis', 'Rhizobium_jaguaris', 'Staphylococcus_condimenti', 'Bacillus_licheniformis', 
    'Stutzerimonas_frequens', 'Trichlorobacter_lovleyi', 'Desemzia_incerta', 'Legionella_adelaidensis', 
    'Staphylococcus_pseudintermedius', 'Alicyclobacillus_fastidiosus', 'Acinetobacter_pseudolwoffii', 
    'Pseudomonas_hunanensis', 'Natribaculum_longum', 'Atlantibacter_hermannii', 'Cupriavidus_metallidurans', 
    'Lactobacillus_gasseri', 'Mycolicibacterium_poriferae', 'Sinorhizobium_alkalisoli', 'Sinorhizobium_fredii', 
    'Segatella_copri', 'Methylorubrum_populi', 'Delftia_acidovorans', 'Paracoccus_alcaliphilus', 'Nocardia_farcinica', 
    'Edwardsiella_piscicida', 'Wohlfahrtiimonas_chitiniclastica', 'Rhizobium_ruizarguesonis', 'Halapricum_desulfuricans', 
    'Enterobacter_bugandensis', 'Pseudomonas_fulva', 'Acetoanaerobium_noterae', 'Ralstonia_mannitolilytica', 
    'Rossellomorea_marisflavi', 'Bacillus_mycoides', 'Vagococcus_xieshaowenii', 'Lactococcus_piscium', 
    'Pseudomonas_migulae', 'Pseudoalteromonas_donghaensis', 'Aeromonas_taiwanensis', 'Corynebacterium_diphtheriae', 
    'Rhodococcus_rhodochrous', 'Carnobacterium_maltaromaticum', 'Halorussus_limi', 'Klebsiella_variicola', 
    'Staphylococcus_succinus', 'Parabacteroides_distasonis', 'Paraburkholderia_hospita', 'Clostridium_botulinum', 
    'Delftia_tsuruhatensis', 'Actinobacillus_indolicus', 'Listeria_seeligeri', 'Pantoea_jilinensis', 
    'Pseudomonas_amygdali', 'Citrobacter_koseri', 'Klebsiella_africana', 'Laribacter_hongkongensis', 
    'Kluyvera_intermedia', 'Streptococcus_pyogenes', 'Deinococcus_actinosclerus', 'Paroceanicella_profunda', 
    'Kurthia_gibsonii', 'Sinorhizobium_medicae', 'Aerococcus_urinaeequi', 'Lacticaseibacillus_rhamnosus', 
    'Enterobacter_hormaechei', 'Enterococcus_faecalis', 'Staphylococcus_saprophyticus', 'Bacillus_cereus', 
    'Marinobacter_salarius', 'Natribaculum_breve', 'Acinetobacter_chinensis', 'Hymenobacter_psoromatis', 
    'Indioceanicola_profundi', 'Acinetobacter_cumulans', 'Rahnella_aquatilis', 'Agrobacterium_pusense', 
    'Phocaeicola_dorei', 'Paenarthrobacter_ureafaciens', 'Acaryochloris_marina', 'Vibrio_pelagius', 'Aeromonas_media', 
    'Vibrio_gangliei', 'Acinetobacter_haemolyticus', 'Pasteurella_multocida', 'Sinorhizobium_terangae', 
    'Pseudomonas_fluorescens', 'Streptococcus_dysgalactiae', 'Staphylococcus_chromogenes', 'Vibrio_campbellii', 
    'Natrinema_zhouii', 'Glutamicibacter_nicotianae', 'Serratia_liquefaciens', 'Chitinibacter_bivalviorum', 
    'Roseomonas_mucosa', 'Paracoccus_aminophilus', 'Bacillus_wiedmannii', 'Bacillus_pseudomycoides', 
    'Bacteroides_ovatus', 'Salmonella_enterica', 'Pseudosulfitobacter_pseudonitzschiae', 'Providencia_vermicola', 
    'Macrococcoides_caseolyticum', 'Microvirga_ossetica', 'Aliarcobacter_butzleri', 'Streptomyces_coeruleorubidus', 
    'Burkholderia_stagnalis', 'Xanthomonas_perforans', 'Citrobacter_cronae', 'Sinorhizobium_americanum', 
    'Glutamicibacter_protophormiae', 'Streptococcus_parasuis', 'Comamonas_odontotermitis', 'Priestia_filamentosa', 
    'Martelella_mediterranea', 'Psychrobacter_raelei', 'Staphylococcus_simulans', 'Jiella_pelagia', 
    'Providencia_heimbachae', 'Sinorhizobium_meliloti', 'Streptomyces_anthocyanicus', 'Aulosira_laxa', 
    'Providencia_rettgeri', 'Simplicispira_suum', 'Skermanella_rosea', 'Enterobacter_cancerogenus', 
    'Pseudomonas_hygromyciniae', 'Streptococcus_parauberis', 'Alicycliphilus_denitrificans', 'Cetobacterium_somerae', 
    'Kluyvera_ascorbata', 'Desulfovibrio_desulfuricans', 'Staphylococcus_capitis', 'Acinetobacter_baumannii', 
    'Azospirillum_oryzae', 'Erwinia_amylovora', 'Pseudomonas_luteola', 'Bacteroides_finegoldii', 
    'Paenibacillus_polymyxa', 'Yokenella_regensburgei', 'Staphylococcus_equorum', 'Limosilactobacillus_portuensis', 
    'Bacillus_safensis', 'Paraburkholderia_phymatum', 'Francisella_tularensis', 'Providencia_stuartii', 
    'Escherichia_fergusonii', 'Achromobacter_denitrificans', 'Trueperella_pyogenes', 'Pantoea_alfalfae', 
    'Listeria_innocua', 'Paucilactobacillus_hokkaidonensis', 'Pseudomonas_veronii', 'Yersinia_pseudotuberculosis', 
    'Gluconobacter_albidus', 'Peptoclostridium_acidaminophilum', 'Providencia_zhijiangensis', 'Agrobacterium_rubi', 
    'Ensifer_canadensis', 'Azospirillum_lipoferum', 'Allorhizobium_pseudoryzae', 'Rhizobium_rosettiformans', 
    'Hymenobacter_aerilatus', 'Aliivibrio_salmonicida', 'Vibrio_vulnificus', 'Vibrio_furnissii', 'Aeromonas_caviae', 
    'Acinetobacter_baylyi', 'Pseudomonas_alcaligenes', 'Oceanobacillus_oncorhynchi', 'Vibrio_parahaemolyticus', 
    'Rhizobium_grahamii', 'Phytobacter_diazotrophicus', 'Pantoea_eucrina', 'Ralstonia_syzygii', 
    'Exiguobacterium_mexicanum', 'Pararhizobium_qamdonense', 'Campylobacter_jejuni', 'Cronobacter_sakazakii', 
    'Staphylococcus_hominis', 'Streptomyces_althioticus', 'Enterococcus_lactis', 'Mammaliicoccus_vitulinus', 
    'Levilactobacillus_brevis', 'Paracoccus_ferrooxidans', 'Nostoc_edaphicum', 'Legionella_pneumophila', 
    'Myroides_odoratimimus', 'Serratia_nevei', 'Pantoea_eucalypti', 'Yersinia_pestis', 'Rhodococcus_ruber', 
    'Ralstonia_pickettii', 'Ewingella_americana', 'Vibrio_mediterranei', 'Rahnella_victoriana', 
    'Enterobacter_roggenkampii', 'Yersinia_ruckeri', 'Comamonas_endophytica', 'Paenibacillus_cellulositrophicus', 
    'Streptococcus_pneumoniae', 'Escherichia_albertii', 'Corynebacterium_resistens', 'Mannheimia_haemolytica', 
    'Rhizobium_tropici', 'Burkholderia_vietnamiensis', 'Bacillus_tropicus', 'Staphylococcus_shinii', 'Kocuria_rosea', 
    'Lactobacillus_paragasseri', 'Methylobacterium_aquaticum', 'Mycobacterium_intracellulare', 'Acinetobacter_pittii', 
    'Brucella_intermedia', 'Rhodobacter_xanthinilyticus', 'Psychrobacter_maritimus', 'Shewanella_baltica', 
    'Corynebacterium_marinum', 'Rhodovastum_atsumiense', 'Pseudomonas_bubulae', 'Comamonas_thiooxydans', 
    'Cupriavidus_oxalaticus', 'Vibrio_owensii', 'Providencia_huaxiensis', 'Haloarcula_marismortui', 'Fulvitalea_axinellae', 
    'Bacillus_paramycoides', 'Xanthomonas_citri', 'Subdoligranulum_variabile', 'Bradyrhizobium_quebecense', 
    'Mycobacteroides_abscessus', 'Macrococcoides_canis', 'Limosilactobacillus_reuteri', 'Acinetobacter_junii', 
    'Shewanella_decolorationis', 'Acinetobacter_nosocomialis', 'Rouxiella_badensis', 'Borreliella_tanukii', 
    'Citrobacter_sedlakii', 'Mycetohabitans_rhizoxinica', 'Acinetobacter_portensis', 'Pseudomonas_cerasi', 
    'Escherichia_marmotae', 'Staphylococcus_epidermidis', 'Mycobacterium_avium', 'Bradyrhizobium_elkanii', 
    'Bibersteinia_trehalosi', 'Klebsiella_huaxiensis'
]


species_goin_out = [x.replace('_', ' ') for x in species_list_i_think]
species_text_block = '\n'.join(species_goin_out)

with open('species_list_for_ncbi.txt', 'w') as f:
    f.write(species_text_block)


f.close()

# go from list of species to assemblies of genomes in a not disgusting memory way? 
# same exact process as before, except that now i have no idea what the off-target hits actually implies, unless 
# the genomes are annotated.


##BASH 

#mkdir -p downloaded_genomes
#
#while read -r species; do
#    [ -z "$species" ] && continue
#    echo "Downloading: $species..."
#    datasets download genome taxon "$species" --reference \
#      --filename "downloaded_genomes/${species}.zip" \
#      --no-progressbar || echo " ---> WARNING: No genome available or download failed for $species"
#
#done < species_list_for_ncbi.txt


#mkdir -p all_host_fastas

#for zipfile in downloaded_genomes/*.zip; do
#    unzip -j "$zipfile" "*.fna" -d all_host_fastas/ >/dev/null 2>&1
#done


#abricate --db ncbi all_host_fastas/*.fna > host_genome_amr_annotations.tsv


import re
import os
import sys
import signal
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


# =============================================================================
# PATHS & PARAMETERS
# =============================================================================

CRISPR_OUT_DIR        = Path('crispr_results')
GENOME_DIR            = Path('all_host_fastas')       # <-- Updated for genomes
OUT_DIR               = Path('homology_check_host')   # <-- New output dir to avoid overwriting
AMR_TSV               = Path('host_genome_amr_annotations.tsv')   # <-- Abricate output
OUT_DIR.mkdir(exist_ok=True)

DB_PATH               = OUT_DIR / 'genome_db'
QUERY_FA              = OUT_DIR / 'guides.fa'
BLAST_OUT             = OUT_DIR / 'blast_hits_raw.tsv'

MAX_MISMATCHES        = 3      # hard ceiling; hits above this are dropped
MIN_MISMATCHES_FOR_OK = 2      # < this → HIGH risk  (0 or 1 mm = HIGH)
N_PROCESSES           = 4      # parallel BLAST workers


# =============================================================================
# GRACEFUL INTERRUPT
# =============================================================================

_worker_pids = set()

def _sigint_handler(sig, frame):
    print('\n[interrupt] Terminating worker processes...', flush=True)
    for pid in _worker_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    sys.exit(1)

signal.signal(signal.SIGINT, _sigint_handler)


# =============================================================================
# PAM DEFINITIONS
# =============================================================================

IUPAC_RE = {
    'A':'A','C':'C','G':'G','T':'T',
    'N':'[ACGT]',
    'R':'[AG]','Y':'[CT]','S':'[GC]','W':'[AT]',
    'K':'[GT]','M':'[AC]',
    'B':'[CGT]','D':'[AGT]','H':'[ACT]','V':'[ACG]',
}

def pam_to_regex(pam_str):
    return re.compile(''.join(IUPAC_RE[c] for c in pam_str.upper()))

EDITOR_PAM = {
    'BE3':               'NGG',
    'Valdez_narrow_ABE': 'NGG',
    'CRISPR-cBEST':      'NGG',
    'CRISPR-aBEST':      'NGG',
    'BE4':               'NGG',
    'ABE8e':             'NGG',
    'VQR-BE3':           'NGAN',
    'EQR-BE3':           'NGAG',
    'VRER-BE3':          'NGCG',
    'SaBE3':             'NNGRRT',
    'SaKKH-BE3':         'NNNRRT',
}

EDITOR_PAM_RE  = {ed: pam_to_regex(pam) for ed, pam in EDITOR_PAM.items()}
EDITOR_PAM_LEN = {ed: len(pam)          for ed, pam in EDITOR_PAM.items()}


# =============================================================================
# VECTORISED PAM CHECK
# =============================================================================

def _extract_pam_series(plasmid_seqs, plasmid_ids, sstart_0, send_0,
                         sstrand, pam_len):
    pam_seqs = []
    for pid, ss, se, strand in zip(plasmid_ids, sstart_0, send_0, sstrand):
        seq = plasmid_seqs.get(pid, '')
        N   = len(seq)
        if not seq:
            pam_seqs.append('')
            continue
        if strand == 'plus':
            ps = se
            pe = ps + pam_len
            pam_seqs.append(seq[ps:pe] if pe <= N else '')
        else:
            pe = ss - 1  
            ps = pe - pam_len
            if ps < 0:
                pam_seqs.append('')
            else:
                raw = seq[ps:pe]
                pam_seqs.append(str(Seq(raw).reverse_complement()))
    return pd.Series(pam_seqs, dtype='string')


def apply_pam_filter_vectorised(blast_df, plasmid_seqs, guide_to_editor):
    blast_df = blast_df.copy()
    blast_df['editor']  = blast_df['guide_id'].map(guide_to_editor)
    blast_df['pam_ok']  = False
    blast_df['pam_seq'] = ''

    for editor, grp_idx in blast_df.groupby('editor').groups.items():
        pam_len = EDITOR_PAM_LEN.get(editor)
        pam_re  = EDITOR_PAM_RE.get(editor)
        if pam_len is None or pam_re is None:
            continue

        grp = blast_df.loc[grp_idx]
        pam_seqs = _extract_pam_series(
            plasmid_seqs,
            grp['plasmid_id'].values,
            grp['sstart_0'].values,
            grp['send_0'].values,
            grp['sstrand'].values,
            pam_len,
        )
        
        pam_ok = pam_seqs.str.fullmatch(
            ''.join(IUPAC_RE[c] for c in EDITOR_PAM[editor].upper())
        ).fillna(False)

        blast_df.loc[grp_idx, 'pam_ok']  = pam_ok.values
        blast_df.loc[grp_idx, 'pam_seq'] = pam_seqs.values

    return blast_df[blast_df['pam_ok']].copy()


# =============================================================================
# PARALLEL BLAST WORKER
# =============================================================================

def _blast_chunk(args):
    chunk_fa, db_path, out_tsv, threads = args
    cmd = [
        'blastn',
        '-task',          'blastn-short',
        '-query',         str(chunk_fa),
        '-db',            str(db_path),
        '-out',           str(out_tsv),
        '-outfmt',        '6 qseqid sseqid pident length mismatch qlen sstart send sstrand',
        '-evalue',        '0.01',
        '-perc_identity', '80',
        '-word_size',     '9',
        '-dust',          'no',
        '-num_threads',   str(threads),
        '-strand',        'both',
    ]
    proc = subprocess.Popen(cmd)
    return proc.pid, proc, out_tsv

BLAST_THREADS = 16

def run_blast_parallel(query_fa, db_path, out_tsv, n_processes):
    records = list(SeqIO.parse(query_fa, 'fasta'))
    if not records:
        out_tsv.write_text('')
        return

    chunk_size = max(1, len(records) // n_processes + 1)
    chunks     = [records[i:i+chunk_size]
                  for i in range(0, len(records), chunk_size)]

    tmp_dir    = Path(tempfile.mkdtemp(dir=out_tsv.parent))
    chunk_fas  = []
    chunk_outs = []

    for i, chunk in enumerate(chunks):
        cfa = tmp_dir / f'chunk_{i}.fa'
        SeqIO.write(chunk, cfa, 'fasta')
        chunk_fas.append(cfa)
        chunk_outs.append(tmp_dir / f'chunk_{i}.tsv')

    procs = []
    for cfa, cout in zip(chunk_fas, chunk_outs):
        t = max(1, BLAST_THREADS // n_processes)
        cmd = [
            'blastn', '-task', 'blastn-short',
            '-query', str(cfa), '-db', str(db_path),
            '-out',   str(cout),
            '-outfmt', '6 qseqid sseqid pident length mismatch qlen sstart send sstrand',
            '-evalue', '0.01', '-perc_identity', '80',
            '-word_size', '9', '-dust', 'no',
            '-num_threads', str(t), '-strand', 'both',
        ]
        p = subprocess.Popen(cmd)
        _worker_pids.add(p.pid)
        procs.append((p, cout))

    print(f'  {len(procs)} BLAST workers running (PIDs: {_worker_pids})...')

    for p, cout in procs:
        rc = p.wait()
        if rc not in (0, -15):
            raise RuntimeError(f'blastn (pid {p.pid}) exited with code {rc}')
        _worker_pids.discard(p.pid) if hasattr(_worker_pids, 'discard') else None

    with open(out_tsv, 'wb') as out:
        for _, cout in procs:
            if cout.exists():
                out.write(cout.read_bytes())

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f'  BLAST complete → {out_tsv}')


# =============================================================================
# STEP 1 – LOAD GENOME SEQUENCES
# =============================================================================

print('Loading host genome sequences for PAM checking...')
plasmid_seqs = {}
# Grab both .fna and .fa in case of mixed formats
for ext in ['*.fna', '*.fa']:
    for fa in sorted(GENOME_DIR.glob(ext)):
        for rec in SeqIO.parse(fa, 'fasta'):
            if len(rec.seq) >= 100:
                plasmid_seqs[rec.id] = str(rec.seq).upper()
print(f'Loaded {len(plasmid_seqs):,} sequence contigs/chromosomes')


# =============================================================================
# STEP 2 – BUILD BLAST DATABASE
# =============================================================================

if not Path(str(DB_PATH) + '.nhr').exists():
    print('Building BLAST database...')
    combined = OUT_DIR / 'genomes_combined.fa'
    with open(combined, 'w') as out:
        for pid, seq in plasmid_seqs.items():
            SeqIO.write(SeqRecord(Seq(seq), id=pid, description=''), out, 'fasta')
    subprocess.run([
        'makeblastdb', '-in', str(combined),
        '-dbtype', 'nucl', '-out', str(DB_PATH),
    ], check=True)
    print('  DB built.')
else:
    print(f'BLAST DB exists: {DB_PATH}')


# =============================================================================
# STEP 3 – LOAD CANDIDATES & WRITE QUERY FASTA
# =============================================================================

safe = pd.read_csv(CRISPR_OUT_DIR / 'candidates_safe.csv')

guide_cols    = ['editor', 'editor_type', 'protospacer', 'strand']
unique_guides = safe[guide_cols].drop_duplicates().reset_index(drop=True).copy()
unique_guides['guide_id'] = [f'guide_{i:06d}' for i in range(len(unique_guides))]

records = [
    SeqRecord(
        Seq(row['protospacer'].upper().replace('U', 'T')),
        id=row['guide_id'], description=''
    )
    for _, row in unique_guides.iterrows()
]
SeqIO.write(records, QUERY_FA, 'fasta')
print(f'Wrote {len(records):,} query guides → {QUERY_FA}')


# =============================================================================
# STEP 4 – RUN BLASTN (parallel chunks)
# =============================================================================

if not BLAST_OUT.exists() or os.path.getsize(BLAST_OUT) == 0:
    print(f'Running BLAST ({N_PROCESSES} parallel workers)...')
    run_blast_parallel(QUERY_FA, DB_PATH, BLAST_OUT, N_PROCESSES)
else:
    print(f'Using cached BLAST output: {BLAST_OUT}')


# =============================================================================
# STEP 5 – PARSE, AMR FILTER, MISMATCH FILTER, VECTORISED PAM CHECK
# =============================================================================

print('Parsing BLAST hits...')
if os.path.getsize(BLAST_OUT) == 0:
    print('No hits found in BLAST. Skipping filters.')
    blast_raw = pd.DataFrame(columns=[
        'guide_id', 'plasmid_id', 'pident', 'length',
        'mismatch', 'qlen', 'sstart', 'send', 'sstrand'
    ])
    blast_raw = blast_raw.astype({'guide_id': 'string', 'plasmid_id': 'string', 'sstrand': 'string'})
else:
    blast_raw = pd.read_csv(
        BLAST_OUT, sep='\t', header=None,
        names=['guide_id', 'plasmid_id', 'pident', 'length',
               'mismatch', 'qlen', 'sstart', 'send', 'sstrand'],
        dtype={'guide_id': 'string', 'plasmid_id': 'string', 'sstrand': 'string'},
    )

blast_raw = blast_raw[
    (blast_raw['length'] >= blast_raw['qlen'] - 1) &
    (blast_raw['mismatch'] <= MAX_MISMATCHES)
].copy()
print(f'Hits after mismatch filter (≤{MAX_MISMATCHES} mm): {len(blast_raw):,}')

coords_min = blast_raw[['sstart','send']].min(axis=1).astype(int)
coords_max = blast_raw[['sstart','send']].max(axis=1).astype(int)
blast_raw['sstart_0'] = coords_min
blast_raw['send_0']   = coords_max

# --- AMR SELF-HIT FILTERING ---
if AMR_TSV.exists() and len(blast_raw) > 0:
    print('Loading AMR annotations to filter self-hits...')
    amr_df = pd.read_csv(AMR_TSV, sep='\t')
    
    if 'SEQUENCE' in amr_df.columns:
        amr_df = amr_df[['SEQUENCE', 'START', 'END', 'GENE']]
        amr_df.rename(columns={'SEQUENCE': 'plasmid_id'}, inplace=True)
        
        def is_self_hit(row, amr_data):
            contig_amrs = amr_data[amr_data['plasmid_id'] == row['plasmid_id']]
            if contig_amrs.empty:
                return False
            
            buffer = 50 # 50bp allowance on either side of the AMR gene 
            for _, amr in contig_amrs.iterrows():
                if (row['send_0'] >= amr['START'] - buffer) and (row['sstart_0'] <= amr['END'] + buffer):
                    return True
            return False

        print("Removing overlapping AMR self-hits from BLAST results...")
        blast_raw['is_amr_self_hit'] = blast_raw.apply(is_self_hit, amr_data=amr_df, axis=1)
        
        hits_before = len(blast_raw)
        blast_raw = blast_raw[~blast_raw['is_amr_self_hit']].copy()
        print(f"Removed {hits_before - len(blast_raw):,} self-hits. {len(blast_raw):,} off-targets remain.")
    else:
        print("Warning: AMR TSV format unrecognized. Skipping self-hit filter.")
else:
    print("No AMR annotations provided/needed. Skipping self-hit filter.")
# ------------------------------

guide_to_editor = unique_guides.set_index('guide_id')['editor'].to_dict()

if len(blast_raw) > 0:
    print('Applying per-editor PAM filter (vectorised)...')
    blast = apply_pam_filter_vectorised(blast_raw, plasmid_seqs, guide_to_editor)
    print(f'Hits after PAM filter: {len(blast):,}')
    
    blast['risk_tier'] = (blast['mismatch'] < MIN_MISMATCHES_FOR_OK).map(
        {True: 'HIGH', False: 'MEDIUM'}
    )
else:
    blast = blast_raw.copy()
    blast['risk_tier'] = pd.Series(dtype='string')

del blast_raw


# =============================================================================
# STEP 6 – SUMMARISE PER GUIDE
# =============================================================================

def summarise(df):
    best = df.loc[df['mismatch'].idxmin()]
    return pd.Series({
        'n_offtarget_hits':   len(df),
        'n_high_risk_hits':   (df['risk_tier'] == 'HIGH').sum(),
        'n_medium_risk_hits': (df['risk_tier'] == 'MEDIUM').sum(),
        'n_plasmids_hit':     df['plasmid_id'].nunique(),
        'min_mismatch':       int(df['mismatch'].min()),
        'max_pident':         df['pident'].max(),
        'worst_hit_plasmid':  best['plasmid_id'],
        'worst_hit_sstart_0': int(best['sstart_0']),
        'worst_hit_send_0':   int(best['send_0']),
        'worst_hit_strand':   best['sstrand'],
        'worst_hit_pam':      best['pam_seq'] if 'pam_seq' in best else '',
    })

if len(blast) > 0:
    per_guide = (blast.groupby('guide_id')
                      .apply(summarise, include_groups=False)
                      .reset_index())
else:
    # Empty DataFrame fallback to prevent merge crash
    per_guide = pd.DataFrame(columns=[
        'guide_id', 'n_offtarget_hits', 'n_high_risk_hits', 'n_medium_risk_hits',
        'n_plasmids_hit', 'min_mismatch', 'max_pident', 'worst_hit_plasmid',
        'worst_hit_sstart_0', 'worst_hit_send_0', 'worst_hit_strand', 'worst_hit_pam'
    ])


# =============================================================================
# STEP 7 – MERGE BACK
# =============================================================================

unique_guides_hom = unique_guides.merge(per_guide, on='guide_id', how='left')

fill_int = ['n_offtarget_hits', 'n_high_risk_hits', 'n_medium_risk_hits', 'n_plasmids_hit']
unique_guides_hom[fill_int] = unique_guides_hom[fill_int].fillna(0).astype(int)
unique_guides_hom['min_mismatch'] = unique_guides_hom['min_mismatch'].fillna(999)
unique_guides_hom['max_pident']   = unique_guides_hom['max_pident'].fillna(0)

unique_guides_hom['has_high_risk_hit'] = unique_guides_hom['n_high_risk_hits'] > 0

print(f'Guides HIGH-risk: {unique_guides_hom["has_high_risk_hit"].sum():,}  '
      f'clean: {(~unique_guides_hom["has_high_risk_hit"]).sum():,}')

merge_key = ['editor', 'protospacer', 'strand']

safe_hom = safe.merge(
    unique_guides_hom[merge_key + [
        'guide_id', 'n_offtarget_hits', 'n_high_risk_hits',
        'n_medium_risk_hits', 'n_plasmids_hit',
        'min_mismatch', 'max_pident', 'has_high_risk_hit',
        'worst_hit_plasmid', 'worst_hit_sstart_0', 'worst_hit_send_0',
        'worst_hit_strand', 'worst_hit_pam',
    ]],
    on=merge_key, how='left'
)

safe_hom['homology_clean'] = ~safe_hom['has_high_risk_hit'].fillna(False)
safe_hom['rank_score']     = (
    safe_hom['efficiency_score'] * (1 - safe_hom['pct_early']) *
    safe_hom['pct_conserved_dn'].fillna(0.5)
)
safe_hom        = safe_hom.sort_values('rank_score', ascending=False)
safe_hom_strict = safe_hom[safe_hom['homology_clean']].copy()

print(f'Safe candidates (original):  {len(safe):,}')
print(f'Safe + homology-clean:       {len(safe_hom_strict):,}')


# =============================================================================
# STEP 8 – TARGET LOSS REPORT
# =============================================================================

all_input_pids  = set(safe['query_id'].unique())
pids_with_clean = set(safe_hom_strict['query_id'].unique())
pids_fully_lost = all_input_pids - pids_with_clean

lost_guide_info = (
    safe_hom[
        safe_hom['query_id'].isin(pids_fully_lost) &
        ~safe_hom['homology_clean']
    ][[
        'query_id', 'gene_name', 'family', 'editor', 'protospacer',
        'n_high_risk_hits', 'n_plasmids_hit', 'min_mismatch',
        'worst_hit_plasmid', 'worst_hit_sstart_0', 'worst_hit_send_0',
        'worst_hit_strand', 'worst_hit_pam',
    ]]
    .drop_duplicates()
)

if len(lost_guide_info) > 0:
    target_loss = (
        lost_guide_info
        .sort_values('min_mismatch')
        .groupby('query_id', sort=False)
        .first()
        .reset_index()
    )
else:
    target_loss = pd.DataFrame()

print(f'\n{"="*60}')
print('TARGET LOSS REPORT')
print(f'{"="*60}')
print(f'Total input targets:                    {len(all_input_pids):,}')
print(f'Targets with ≥1 homology-clean guide:   {len(pids_with_clean):,}')
print(f'Targets LOST (no clean guide remains):  {len(pids_fully_lost):,}  '
      f'({len(pids_fully_lost)/max(len(all_input_pids),1)*100:.1f}%)')

if len(pids_fully_lost) > 0 and not target_loss.empty:
    print('\nLoss by family:')
    for fam, cnt in target_loss['family'].value_counts().items():
        print(f'  {str(fam):<25s} {cnt:>5,}')
    print('\nLoss by editor:')
    for ed, cnt in target_loss['editor'].value_counts().items():
        print(f'  {str(ed):<25s} {cnt:>5,}')
    print('\nMismatch distribution of worst hits:')
    print(target_loss['min_mismatch'].value_counts().sort_index().to_string())

print(f'{"="*60}')


# =============================================================================
# STEP 9 – GREEDY MINIMAL sgRNA SET
# =============================================================================

def greedy_minimal_set(candidates_df):
    if candidates_df.empty:
        return pd.DataFrame()
        
    guide_cov = (
        candidates_df
        .groupby(['editor', 'protospacer', 'strand'])
        .agg(
            covered            = ('query_id',          lambda x: frozenset(x)),
            n_pids             = ('query_id',          'nunique'),
            mean_eff           = ('efficiency_score',  'mean'),
            mean_early         = ('pct_early',         'mean'),
            mean_cons          = ('pct_conserved_dn',  'mean'),
            family             = ('family',            'first'),
            editor_type        = ('editor_type',       'first'),
            n_offtarget_hits   = ('n_offtarget_hits',  'first'),
            n_high_risk_hits   = ('n_high_risk_hits',  'first'),
            n_plasmids_hit     = ('n_plasmids_hit',    'first'),
            min_mismatch       = ('min_mismatch',      'first'),
            worst_hit_plasmid  = ('worst_hit_plasmid', 'first'),
            worst_hit_sstart_0 = ('worst_hit_sstart_0','first'),
            worst_hit_send_0   = ('worst_hit_send_0',  'first'),
        )
        .reset_index()
        .sort_values(['n_pids', 'mean_eff', 'mean_early'],
                     ascending=[False, False, True])
    )
    uncovered = set(candidates_df['query_id'].unique())
    selected  = []
    while uncovered:
        best_new, best_row = frozenset(), None
        for _, r in guide_cov.iterrows():
            new = r['covered'] & uncovered
            if len(new) > len(best_new):
                best_new, best_row = new, r
        if best_row is None:
            break
        selected.append({
            'editor':              best_row['editor'],
            'editor_type':         best_row['editor_type'],
            'protospacer':         best_row['protospacer'],
            'strand':              best_row['strand'],
            'family':              best_row['family'],
            'n_pids_covered':      len(best_new),
            'mean_efficiency':     round(best_row['mean_eff'],   3),
            'mean_pct_early':      round(best_row['mean_early'], 3),
            'mean_pct_cons_dn':    round(best_row['mean_cons'],  3) if pd.notna(best_row['mean_cons']) else None,
            'n_offtarget_hits':    best_row['n_offtarget_hits'],
            'n_high_risk_hits':    best_row['n_high_risk_hits'],
            'n_plasmids_hit':      best_row['n_plasmids_hit'],
            'min_mismatch':        best_row['min_mismatch'],
            'worst_hit_plasmid':   best_row['worst_hit_plasmid'],
            'worst_hit_sstart_0':  best_row['worst_hit_sstart_0'],
            'worst_hit_send_0':    best_row['worst_hit_send_0'],
            'covers':              sorted(best_new),
        })
        uncovered -= best_new
    return pd.DataFrame(selected)


sgrna_set_hom = greedy_minimal_set(safe_hom_strict)

print(f'\nMinimal sgRNA set (homology-filtered): {len(sgrna_set_hom)} guide(s)')
if not sgrna_set_hom.empty:
    print(sgrna_set_hom[[
        'editor', 'protospacer', 'strand', 'n_pids_covered',
        'mean_efficiency', 'n_high_risk_hits', 'n_plasmids_hit', 'min_mismatch'
    ]].to_string(index=False))


# =============================================================================
# STEP 10 – SAVE
# =============================================================================

safe_hom.to_csv(         OUT_DIR / 'candidates_safe_homology_annotated.csv', index=False)
safe_hom_strict.to_csv(  OUT_DIR / 'candidates_safe_homology_clean.csv',      index=False)
sgrna_set_hom.to_csv(    OUT_DIR / 'sgrna_minimal_set_homology_filtered.csv', index=False)
unique_guides_hom.to_csv(OUT_DIR / 'unique_guides_homology_summary.csv',      index=False)
blast.to_csv(            OUT_DIR / 'blast_hits_pam_filtered.csv',             index=False)
if not target_loss.empty:
    target_loss.to_csv(  OUT_DIR / 'target_loss_report.csv',                  index=False)

print(f'\nAll outputs → {OUT_DIR}/')




pesto = pd.read_csv(Path('homology_check_pam') / 'target_loss_report.csv')
combined_target_loss = pd.concat([target_loss, pesto])
combined_target_loss = combined_target_loss.drop_duplicates(subset=['query_id'])


ohno = list(set(combined_target_loss['query_id'].tolist()))
ohyeah = list(set(safe['query_id']))
ohmaybe = [x for x in ohyeah if x not in ohno]
#len(ohmaybe)/len(ohyeah)*100
#99.1041887055752

#100 - len(ohmaybe)/len(ohyeah)*100
#0.8958112944248029 % of targets lost












##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

#███████╗██╗   ██╗ █████╗ ██╗     ██╗   ██╗ █████╗ ████████╗███████╗    ██╗  ██╗██╗████████╗███████╗
#██╔════╝██║   ██║██╔══██╗██║     ██║   ██║██╔══██╗╚══██╔══╝██╔════╝    ██║  ██║██║╚══██╔══╝██╔════╝
#█████╗  ██║   ██║███████║██║     ██║   ██║███████║   ██║   █████╗      ███████║██║   ██║   ███████╗
#██╔══╝  ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══██║   ██║   ██╔══╝      ██╔══██║██║   ██║   ╚════██║
#███████╗ ╚████╔╝ ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ███████╗    ██║  ██║██║   ██║   ███████║
#╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝    ╚═╝  ╚═╝╚═╝   ╚═╝   ╚══════╝


import polars as pl 
from collections import Counter

data_dir = Path(os.path.join(os.getcwd(),'plasmid_motif_network/intermediate'))
files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))

df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
df_merged = df_merged.with_columns(
    pl.col('strand').cast(pl.Int32).alias('strand')
)

PID_nuccore_pattern = re.compile(r'^(.+?)_\d+_\d+')
PID_nogene_pattern  = re.compile(r'^(.+?)_(\d+)_(\d+)$')
 
FASTA_DIR = Path('fastas')

import os
import re
import polars as pl

# =============================================================================
# 1. PARSE THE 'fastas' DIRECTORY INTO A POLARS DATAFRAME
# =============================================================================
print("Parsing fasta directory...")
fasta_files = [f for f in os.listdir('fastas') if f.endswith('.fa')]

# Regex to safely extract pieces, handling underscores in plasmid or gene names
# Format: <plasmid_id>_<start>_<stop>_<gene_name>.fa
pattern = re.compile(r'^(.*)_(\d+)_(\d+)_(.*)\.fa$')

genes_data = []
for f in fasta_files:
    match = pattern.match(f)
    if match:
        plasmid, g_start, g_stop, gene = match.groups()
        genes_data.append((plasmid, int(g_start), int(g_stop), gene))

# Create Gene DataFrame and calculate min/max coordinates
df_genes = pl.DataFrame(genes_data, schema=["plasmid_id", "g_start", "g_stop", "gene_name"])
df_genes = df_genes.with_columns([
    pl.min_horizontal("g_start", "g_stop").alias("gene_min"),
    pl.max_horizontal("g_start", "g_stop").alias("gene_max")
])


# =============================================================================
# 2. PREPARE BLAST AND PFAM DATAFRAMES
# =============================================================================
print("Preparing DataFrames...")

# Isolate HIGH risk hits and convert to Polars

#NEED TO COMBINE THE BLAST DATAFRAMES


blast_high = blast[blast['risk_tier'] == 'HIGH'].copy()
df_blast = pl.from_pandas(blast_high)

# Prep Pfam DataFrame (handle start > stop strand flips)
df_pfam = df_merged.select([
    pl.col("plasmid").alias("plasmid_id"),
    pl.col("target_name"),
    pl.min_horizontal("start", "stop").alias("pfam_min"),
    pl.max_horizontal("start", "stop").alias("pfam_max")
])


# =============================================================================
# 3. PERFORM OVERLAP JOINS (THE FAST PART)
# =============================================================================
print("Calculating overlaps...")

# --- A. Find Overlapping Pfam Domains ---
# 1. Join on plasmid_id
# 2. Filter for coordinate overlap: (blast_start <= pfam_max) AND (blast_end >= pfam_min)
# 3. Group by the BLAST hit and gather a list of overlapping domains
overlap_pfam = (
    df_blast.join(df_pfam, on="plasmid_id", how="inner")
    .filter(
        (pl.col("sstart_0") <= pl.col("pfam_max")) &
        (pl.col("send_0") >= pl.col("pfam_min"))
    )
    .group_by(["guide_id", "plasmid_id", "sstart_0", "send_0"])
    .agg(pl.col("target_name").unique().alias("overlapping_pfams"))
)

# --- B. Find Overlapping Genes (from FASTA filenames) ---
overlap_genes = (
    df_blast.join(df_genes, on="plasmid_id", how="inner")
    .filter(
        (pl.col("sstart_0") <= pl.col("gene_max")) &
        (pl.col("send_0") >= pl.col("gene_min"))
    )
    .group_by(["guide_id", "plasmid_id", "sstart_0", "send_0"])
    .agg(pl.col("gene_name").unique().alias("overlapping_genes"))
)


# =============================================================================
# 4. MERGE BACK TO THE MAIN BLAST DATAFRAME
# =============================================================================
print("Annotating hits...")

# Left join the aggregated overlaps back onto the original HIGH risk BLAST hits
df_blast_annotated = (
    df_blast
    .join(overlap_pfam, on=["guide_id", "plasmid_id", "sstart_0", "send_0"], how="left")
    .join(overlap_genes, on=["guide_id", "plasmid_id", "sstart_0", "send_0"], how="left")
)

# Clean up lists into readable strings (e.g. "TraL; HTH_1") and fill nulls with blanks
df_blast_annotated = df_blast_annotated.with_columns([
    pl.col("overlapping_pfams").list.join("; ").fill_null(""),
    pl.col("overlapping_genes").list.join("; ").fill_null("")
])

# Convert back to pandas so you can view/save it normally
blast_high_annotated = df_blast_annotated.to_pandas()

print("Done!")


gib = df_blast_annotated.to_pandas()

glib = gib[(gib['overlapping_pfams'] != '') | (gib['overlapping_genes'] != '')]


Counter(glib['overlapping_pfams'].tolist())
Counter(glib['overlapping_genes'].tolist())


# Counter(glib['overlapping_pfams'].tolist())
#Counter({'DDE_Tnp_IS240': 2027, 'Beta-lactamase2': 117, 'Cpn60_TCP1': 115, '': 62, 'rve_3': 38, 'Peptidase_M56; Transpeptidase': 35, 'Beta-lactamase': 29, 'DDE_Tnp_IS240; TetR_C_1; TetR_N': 12, 'TetR_C_1; TetR_N': 11, 'DDE_Tnp_IS240; Y2_Tnp': 6, 'rve; HTH_38; DDE_Tnp_IS240': 6, 'AAA_11; DDE_Tnp_IS240': 6, 'DDE_Tnp_IS240; TelA': 6, 'DUF6685; DDE_Tnp_IS240': 4, 'EAL': 2, 'DUF6685': 2, 'Transposase_20; DEDD_Tnp_IS110': 1, 'Y2_Tnp': 1, 'rve; HTH_38': 1, 'DDE_Tnp_Tn3; DUF4158': 1, 'TelA': 1, 'HTH_7; Resolvase; DDE_Tnp_IS240': 1, 'MCPsignal; PAS_3; HAMP': 1, 'AAA_11': 1, 'DDE_Tnp_IS66; DDE_Tnp_IS66_C; zf-IS66; LZ_Tnp_IS66': 1})
#>>> Counter(glib['overlapping_genes'].tolist())
#Counter({'': 2415, 'TEM-206': 26, 'BcIII': 10, 'SHV-105': 9, 'CARB-3': 7, "AAC6'-Ib7": 7, 'aadA3': 7, "AAC6'-Ib9": 6})
#>>>

#Cpn60_TCP1: This is GroEL, a highly conserved, absolutely essential Class I chaperonin. It forms a physical barrel that misfolded proteins go inside to be re-folded correctly. It is heavily upregulated during heat shock or stress.

#MCPsignal; PAS_3; HAMP: These domains almost always appear together in Methyl-accepting Chemotaxis Proteins (MCPs). They are trans-membrane sensor receptors. The bacteria use them to "smell" their environment (e.g., swimming toward food or away from toxins).

#EAL: This domain breaks down cyclic-di-GMP (c-di-GMP), a massive secondary messenger in bacteria that controls the switch between swimming (motile) and settling down to form a biofilm.

#The rest are mobile related, AA1 and HTH probably involved to some extent also

