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



#
#FASTA_DIR        = Path('fastas')
#PFAM_FASTA_DIR   = Path('pfam_fastas')
#MERGED_FASTA_DIR = Path('merged_nonoverlapping_fastas')
#
#
#
#data_dir = Path(os.path.join(os.getcwd(),'plasmid_motif_network/intermediate'))
#files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))
#
#df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
#
#df_merged = df_merged.with_columns(
#    pl.col('strand').cast(pl.Int32).alias('strand')
#)
#
#
#PID_nuccore_pattern = re.compile(r'^(.+?)_\d+_\d+')
#PID_nogene_pattern  = re.compile(r'^(.+?)_(\d+)_(\d+)$')
# 
#
#merged_kept_PIDs = ['.'.join(x.split('.')[:-1]) for x in os.listdir(MERGED_FASTA_DIR)]
#kept_pfam_PIDs   = [p for p in merged_kept_PIDs if PID_nogene_pattern.match(p)]
#kept_fasta_PIDs  = [p for p in merged_kept_PIDs if not PID_nogene_pattern.match(p)]
#total_PIDs = len(merged_kept_PIDs)
#
#
##test = pd.read_csv('amrfindermapped_beta_lactamases.csv', low_memory=False)
##test = test.loc[test['query_id'].isin(merged_kept_PIDs)]
#
#
#test_new = pd.read_csv('amrfindermapped_beta_lactamases_new.csv', low_memory=False)
#prev_mapped_names = ['TEM-1', 'CTX-M-140', 'CMY-4', 'CTX-M-9', 'IMP-22', 'CTX-M-2', 'PAD', 'VEB', 'CTX-M-44', 'NDM-31', 'SFO-1', 'ROB-1', 'OXA-926', 'SHV', 'AFM-1', 'TEM-181', 'CTX-M-59', 'NDM-4', 'DHA-7', 'VIM', 'CTX-M-63', 'OXA-1041', 'CTX-M-125', 'CTX-M-123', 'OXA-1203', 'IMP-19', 'CTX-M-243', 'AFM-3', 'OXA-1204', 'bla-A2', 'LMB-1', 'GES-11', 'mecR1', 'OXA-1', 'BKC-1', 'FOX', 'CTX-M-136', 'OXA-58', 'NDM-1', 'CTX-M-32', 'TEM-171', 'KPC-6', 'KPC-49', 'CTX-M-30', 'CARB-16', 'KPC-93', 'L2', 'LAP-1', 'OXA-567', 'TEM-215', 'CMY-23', 'IMI', 'CMY-111', 'SHV-2A', 'RCP', 'OXA-19', 'OXA-436', 'VEB-18', 'OXA-237', 'NDM-19', 'CTX-M-17', 'IMI-16', 'CMY-6', 'CMY-172', 'VEB-5', 'KPC-109', 'FOX-5', 'IMP-100', 'OXA-655', 'PAU-1', 'TEM-21', 'OXA-96', 'VEB-16', 'VIM-19', 'IMP-56', 'OXA-2', 'SHV-30', 'CTX-M-25', 'SHV-28', 'IMP-45', 'IMP-26', 'CMY-148', 'OKP-B', 'VIM-2', 'CTX-M-40', 'KPC-204', 'OXA-932', 'TEM-4', 'mecA', 'OXA-420', 'KPC-121', 'NDM-5', 'NPS-1', 'KPC-3', 'TEM-12', 'ELC', 'KPC-113', 'MOX', 'OXA-164', 'HBL', 'PDC-16', 'CARB-2', 'OXA-653', 'PER-4', 'CTX-M-104', 'SHV-11', 'TEM-156', 'R39', 'PSV', 'GES-20', 'NDM-27', 'PEN-B', 'DIM-1', 'OXA-9', 'IMP-69', 'OXA-246', 'PER-1', 'VIM-7', 'OXA-198', 'CTX-M-173', 'TEM-61', 'OXA-101', 'TEM-34', 'CAE-1', 'MUN-1', 'NDM-29', 'TEM-3', 'MYO-1', 'SHV-7', 'OXA-97', 'VIM-84', 'OXA-438', 'CMY', 'VEB-2', 'KPC-33', 'GES-12', 'RAHN', 'NDM', 'IMP-14', 'VIM-11', 'IMP-63', 'CTX-M-226', 'VEB-8', 'CARB-8', 'CMY-97', 'ROB', 'CTX-M-53', 'KPC-40', 'OXA-244', 'SHV-5', 'mecB', 'ROB-11', 'KLUC-5', 'VIM-61', 'TEM-84', 'KPC-154', 'CMY-185', 'OXY-2-16', 'TEM-169', 'cfiA2', 'MCA', 'TEM-168', 'RSC1', 'IMP-23', 'CTX-M-62', 'OXA-732', 'CTX-M-195', 'NDM-9', 'CMY-166', 'VIM-85', 'ADC-30', 'NDM-7', 'TEM-116', 'TMB-1', 'VIM-86', 'CTX-M-174', 'bla-C', 'KPC-29', 'IMP-18', 'OXA-256', 'FRI-3', 'OXA-162', 'NPS', 'TEM-54', 'BIM-1', 'CTX-M-90', 'KPC-125', 'KPC-66', 'RAA-1', 'OXA-66', 'OXA-21', 'CMY-16', 'CTX-M-55', 'CMY-146', 'VEB-9', 'CTX-M-8', 'VHW', 'KPC-17', 'OXA-24', 'SHV-2', 'ADC-176', 'TEM-176', 'cdiA', 'CARB-4', 'KPC-4', 'KPC-14', 'GES-5', 'bla-A', 'pbp2m', 'OXA-232', 'VEB-3', 'CMY-36', 'TEM-20', 'CTX-M-39', 'VEB-25', 'CTX-M-65', 'IMP-11', 'ACC-1', 'OXA-181', 'OXY', 'mecI_of_mecA', 'NDM-17', 'SHV-31', 'KPC-78', 'CTX-M-5', 'IMP-38', 'CMY-44', 'CTX-M-134', 'LCR', 'GES-51', 'IMP-89', 'OXA-779', 'SHV-18', 'CMY-174', 'GIM-1', 'TER', 'GES-1', 'IMP-31', 'CMY-145', 'SHV-1', 'VIM-60', 'CTX-M-130', 'TEM-30', 'TEM-7', 'LAP-2', 'VIM-1', 'GES-44', 'CMY2-MIR-ACT-EC', 'LAP', 'CMY-2', 'RAHN-3', 'FRI-7', 'OXA-1391', 'OXA-82', 'FRI-4', 'SHV-12', 'bla-A_firm', 'CTX-M-64', 'OXA-209', 'OXA', 'DHA-15', 'BKC-2', 'IMI-23', 'TEM-24', 'bla-B1', 'R1', 'CTX-M-15', 'OXA-893', 'ADC', 'CMY-13', 'TEM-40', 'FRI-11', 'CTX-M-215', 'OXA-4', 'IMI-6', 'OXA-517', 'CMY-136', 'CTX-M-1', 'KPC-5', 'TEM-10', 'IMI-5', 'CTX-M-38', 'CTX-M-71', 'OXA-139', 'DHA-27', 'BRO', 'KPC-21', 'OXA-921', 'CMY-10', 'OXA-23', 'KHM-1', 'TEM-57', 'CTX-M-132', 'CTX-M-131', 'OXA-32', 'IMP-10', 'TEM-144', 'FRI-9', 'SCO-1', 'CAE', 'KPC-8', 'LCR-1', 'IMP-1', 'OXA-48', 'RTG', 'KPC-79', 'CMY-141', 'FRI-5', 'OXA-17', 'TEM-237', 'CTX-M-234', 'GES-14', 'NDM-13', 'VIM-27', 'CTX-M-27', 'NDM-36', 'KPC-112', 'KPC-111', 'NDM-14', 'GES-4', 'KPC-53', 'VEB-17', 'CARB-12', 'TEM-52', 'OXA-207', 'TEM-32', 'IMP-94', 'KPC-31', 'OXA-427', 'CTX-M-3', 'CTX-M', 'GES-6', 'SIM-2', 'OXA-520', 'OXA-897', 'bla1', 'VEB-1', 'PER-7', 'CTX-M-58', 'TEM-6', 'PSE', 'BES-1', 'CMY-178', 'NDM-6', 'BEL-1', 'Z', 'ADC-130', 'IMP-13', 'OXA-347', 'OXA-484', 'CTX-M-255', 'ACT-9', 'VIM-24', 'OXA-519', 'I', 'FRI-8', 'OXA-656', 'IMP', 'HER-3', 'PER-2', 'NDM-16b', 'CMY-5', 'OXA-29', 'IMP-64', 'IMP-6', 'TEM-256', 'VAM-1', 'CTX-M-236', 'PEN-bcc', 'ROB-2', 'KPC-84', 'TEM-37', 'bla-A_carba', 'CTX-M-102', 'ACC-4', 'VIM-66', 'FONA', 'CTX-M-14', 'KPC-18', 'OXA-1397', 'cfxA_fam', 'IMI-2', 'MOX-1', 'KPC-12', 'KPC-74', 'KPC-90', 'CTX-M-105', 'TEM-31', 'CTX-M-199', 'CTX-M-24', 'OXA-695', 'TEM-135', 'TEM-26', 'OXA-935', 'PER', 'TEM', 'IMP-96', 'IMP-8', 'NDM-23', 'KPC-67', 'PER-3', 'OXA-47', 'CTX-M-121', 'III', 'PSZ', 'KPC-189', 'CTX-M-98', 'CTX-M-101', 'OXA-1042', 'VCC-1', 'CMY-8', 'ACC', 'AFM-4', 'KPC-70', 'PEN-J', 'DIM', 'NDM-37', 'SHV-44', 'GES-24', 'PAU', 'VIM-23', 'IMI-22', 'OXA-392', 'KPC-110', 'KPC-2', 'CMY-31', 'TEM-33', 'MOX-18', 'VIM-4', 'OXA-10', 'KPC-71', 'KPC-44', 'GMA-1', 'OXA-235', 'CTX-M-37', 'CMY-42', 'OXA-204', 'DHA-1', 'FOX-7', 'VMB-1', 'TEM-210', 'OXA-796', 'PC1', 'OXA-900', 'GES-19', 'TEM-238', 'FRI-6', 'FLC-1', 'CTX-M-253', 'GES-7', 'SIM-1', 'TEM-206', 'IMP-4', 'KPC-87', 'FRI-2', 'OXA-72', 'KPC', 'SHV-102', 'OXA-1202', 'TLA-3', 'OXA-163', 'DHA', 'TEM-190', 'TEM-2', 'OXA-129', 'GES', 'VIM-6', 'KPC-24', 'PNC', 'AFM-2', 'KPC-55', 'PSZ-1', 'CTX-M-251', 'IMP-34']
#new_mapped_names = ['NDM-3', 'NDM-11', 'NDM-20', 'NDM-21', 'VIM-5']
#all_bl_mapped_names = prev_mapped_names + new_mapped_names 
#test_new = test_new.loc[test_new['gene_name'].isin(all_bl_mapped_names)]
#test_old = pd.read_csv('amrfindermapped_beta_lactamases_old.csv', low_memory=False)
#test = pd.concat([test_old, test_new]).drop_duplicates(keep='first')
#
#all_betas = [x for x in test['gene_name'].unique() if isinstance(x, str)]
#
#betas_to_PIDs       = {}
#betas_to_plas       = {}
#pfam_betas_to_PIDs  = {}
#fasta_betas_to_PIDs = {}
#
#for gene in all_betas:
#    queries = list(set(test.loc[test['gene_name'] == gene, 'query_id']))
#    betas_to_PIDs[gene]       = queries
#    pfam_betas_to_PIDs[gene]  = [x for x in queries if PID_nogene_pattern.match(x)]
#    fasta_betas_to_PIDs[gene] = [x for x in queries if not PID_nogene_pattern.match(x)]
#    betas_to_plas[gene]       = list(set(
#        m.group(1) for x in queries if (m := PID_nuccore_pattern.match(x))
#    ))
#
#
#
#plsdb_meta_path = Path('plsdb_meta')
#
#nuc_df  = pd.read_csv(plsdb_meta_path / 'nuccore_only.csv')
#typ_df  = pd.read_csv(plsdb_meta_path / 'typing_only.csv')
#bio_df  = pd.read_csv(plsdb_meta_path / 'biosample.csv', low_memory=False)
#tax_df  = pd.read_csv(plsdb_meta_path / 'taxonomy.csv')
#
#nuc_tax = dict(zip(nuc_df['NUCCORE_ACC'], nuc_df['TAXONOMY_UID']))
#nuc_bio = dict(zip(nuc_df['NUCCORE_ACC'], nuc_df['BIOSAMPLE_UID']))
#nuc_mob = dict(zip(typ_df['NUCCORE_ACC'], typ_df['predicted_mobility']))
#tax_spc = dict(zip(tax_df['TAXONOMY_UID'], tax_df['TAXONOMY_species']))
#nuc_spc = {k: tax_spc.get(v) for k, v in nuc_tax.items()}
#
#
#
#bl_df_merged = df_merged.filter(pl.col('target_name').str.contains('lactamase')|pl.col('target_name').str.contains('Lactamase')|pl.col('query_name').is_in(test['query_id'].tolist()))
#
#
#pfam_beta_ids = list(set(list(bl_df_merged['query_name'])))
#
#BETA_FASTA_PATH = Path('beta_lactam_fastas')
#
#fasta_beta_ids = [x.split('.fa')[0] for x in os.listdir(BETA_FASTA_PATH)]
#
#
#
#
#PID_pattern  = re.compile(r'^(.+?)_(\d+)_(\d+)')
#
#all_beta_ids = list(set(pfam_beta_ids + fasta_beta_ids))
#
#all_beta_ids_standardised = ['_'.join(x.split('_')[:-1]) if not PID_nogene_pattern.match(x) else x for x in all_beta_ids]
#
#plasmid_to_takeouts = {}
#
#for x in all_beta_ids_standardised:
#    nuccore = PID_nogene_pattern.match(x)[1] if PID_nogene_pattern.match(x) else None
#    if nuccore:
#        plasmid_to_takeouts[nuccore] = []
#
#
#for x in all_beta_ids_standardised:
#    nuccore = PID_nogene_pattern.match(x)[1] if PID_nogene_pattern.match(x) else None
#    start = PID_nogene_pattern.match(x)[2] if PID_nogene_pattern.match(x) else None
#    stop = PID_nogene_pattern.match(x)[3] if PID_nogene_pattern.match(x) else None
#    if nuccore and start and stop:
#        plasmid_to_takeouts[nuccore].append((start, stop))
#
#
#plasmid_no_beta_sequences_path = Path('plasmids_no_bl_seq')
#os.makedirs(plasmid_no_beta_sequences_path, exist_ok=True)
#
#
#def merge_intervals(intervals):
#    if not intervals:
#        return []
#    sorted_intervals = sorted(intervals, key=lambda x: x[0])
#    merged = [sorted_intervals[0]]
#    for current in sorted_intervals[1:]:
#        prev = merged[-1]
#        if current[0] <= prev[1]:
#            merged[-1] = (prev[0], max(prev[1], current[1]))
#        else:
#            merged.append(current)
#    return merged
#
#
#import shutil
#
#PLASMID_PATH = Path('plasmids')
#non_bl_plasmids = [x.split('.fa')[0] for x in os.listdir(PLASMID_PATH) if x.split('.fa')[0] not in plasmid_to_takeouts.keys()]
#
#for plasmid in non_bl_plasmids:
#    plas_file_path = PLASMID_PATH / f'{plasmid}.fa'
#    plas_out_path = plasmid_no_beta_sequences_path / f'{plasmid}.fa'
#    shutil.copy(plas_file_path, plas_out_path)
#
#
#
#for plasmid, sites in plasmid_to_takeouts.items():
#    plasmid_fasta = PLASMID_PATH / f'{plasmid}.fa'
#    output_fasta = plasmid_no_beta_sequences_path / f'{plasmid}.fa'
#    if not plasmid_fasta.exists():
#        print(f"Warning: {plasmid_fasta} not found. Skipping.")
#        continue
#    with open(plasmid_fasta, 'r') as f:
#        header = f.readline().strip()
#        seq = "".join(line.strip() for line in f)
#    merged_sites = merge_intervals(sites)
#    new_seq_parts = []
#    current_idx = 0
#    for start, stop in merged_sites:
#        new_seq_parts.append(seq[int(current_idx):int(start)])
#        current_idx = int(stop)
#    new_seq_parts.append(seq[int(current_idx):])
#    new_seq = "".join(new_seq_parts)
#    with open(output_fasta, 'w') as f:
#        f.write(f"{header}\n{new_seq}\n")
#
#
#
###now check for homology in this directory plasmids_no_bl_seq
#all_species = list(set(list(nuc_spc.values())))
#
#
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

CRISPR_OUT_DIR        = Path('crispr_results_twentytwo')
PLASMID_DIR           = Path('plasmids_no_bl_seq')
OUT_DIR               = Path('homology_check_pam_twentytwo')
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

#with open('species_list_for_ncbi.txt', 'w') as f:
#    f.write(species_text_block)
#f.close()



##BASH 
#
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

CRISPR_OUT_DIR        = Path('crispr_results_twentytwo')
GENOME_DIR            = Path('all_host_fastas')       # <-- Updated for genomes
OUT_DIR               = Path('homology_check_host_twentytwo')   # <-- New output dir to avoid overwriting
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




pesto = pd.read_csv(Path('homology_check_pam_twentytwo') / 'target_loss_report.csv')
combined_target_loss = pd.concat([target_loss, pesto])
combined_target_loss = combined_target_loss.drop_duplicates(subset=['query_id'])


ohno = list(set(combined_target_loss['query_id'].tolist()))
ohyeah = list(set(safe['query_id']))
ohmaybe = [x for x in ohyeah if x not in ohno]
#
#>>> len(ohmaybe)/len(ohyeah)*100
#92.11130907726933
#>>> 100 - len(ohmaybe)/len(ohyeah)*100
#7.8886909227306745
#>>> 
#










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



big_blast = pd.concat([pd.read_csv(Path('homology_check_pam_twentytwo') / 'blast_hits_pam_filtered.csv'), pd.read_csv(Path('homology_check_host_twentytwo') / 'blast_hits_pam_filtered.csv')])

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


blast_high = big_blast[big_blast['risk_tier'] == 'HIGH'].copy()
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


blast_high_annotated.to_csv('homology_investigated_twentytwo.csv')

gib = df_blast_annotated.to_pandas()

glib = gib[(gib['overlapping_pfams'] != '') | (gib['overlapping_genes'] != '')]


Counter(glib['overlapping_pfams'].tolist())
Counter(glib['overlapping_genes'].tolist())


glibo = glib.loc[~(glib['overlapping_pfams'].str.contains('lactamase')|glib['overlapping_genes'].str.contains('bla'))]

glibdown = glibo.loc[~glibo['overlapping_pfams'].str.contains('Tnp')]
glibdown = glibdown.loc[~glibdown['overlapping_pfams'].str.contains('tnp')]
glibdown = glibdown.loc[~glibdown['overlapping_pfams'].str.contains('phage')]
glibdown = glibdown.loc[~glibdown['overlapping_pfams'].str.contains('Phage')]

dont_care = [
    # Phage structural / packaging / lifecycle (non-recombination)
    "Terminase_3",
    "Terminase_3C",
    "Terminase_6N",
    "GP3_package",
    "Cag12",
    "T4BSS_DotH_IcmK",
        # AMR genes
    "APH",
    "APH_6_hur",
    "AadA_C",
    "AadA_C; NTP_transf_2",
    "CAT",
    "Arr-ms",
    "Multi_Drug_Res",
    "Multi_Drug_Res; HypA",
    # IS-associated but non-recombinase structural/accessory proteins
    "IstB_IS21"
]

glibdown = glibdown.loc[~glibdown['overlapping_pfams'].isin(dont_care)]


test = pd.read_csv('amrfindermapped_beta_lactamases.csv', low_memory=False)

test_families = list(set(test['gene_family']))

test_genes = list(set(test['gene_name']))


glibdown = glibdown.loc[~glibdown['overlapping_genes'].isin(test_genes)]



gene_pattern = '|'.join(map(re.escape, test_genes))
glibdown = glibdown[~glibdown['overlapping_genes'].str.contains(gene_pattern, na=False)]


fams = ['mecB', 'PER', 'CARB', 'ampC', 'TER', 'PAD', 'pbp2m', 'BKC', 'PSV', 'PC1', 'VMB', 'CTX-M', 'RCP', 'KLUC', 'TLA', 'FONA', 'DIM', 'cdiA', 'KPC', 'mecA', 'ADC', 'TEM', 'PSZ', 'SIM', 'VHW', 'HER', 'BcII', 'GMA', 'SFO', 'RTG', 'OXA', 'BRO', 'LAP', 'NDM', 'GIM', 'PAU', 'IMP', 'mecR1', 'OKP-B', 'ACT', 'MUN', 'MOX', 'RAA', 'RSC', 'PDC', 'TMB', 'bla1', 'LMB', 'cfxA', 'VAM', 'CAE', 'NPS', 'PNC', 'IMI', 'LCR', 'HBL', 'VIM', 'ROB', 'PEN', 'KHM', 'MCA', 'MYO', 'FRI', 'BEL', 'CMY', 'GES', 'BIM', 'BES', 'SCO', 'ELC', 'FLC', 'RAHN', 'SHV', 'ACC', 'FOX', 'PSE', 'VEB', 'VCC', 'cfiA', 'AFM', 'OXY', 'DHA', 'OXA-PR']

fam_pattern = '|'.join(map(re.escape, fams))
glibdown_filtered = glibdown[~glibdown['overlapping_genes'].str.contains(fam_pattern, na=False)]


import ast
import pandas as pd
from pathlib import Path

# 1. Load the guide mapping keys
guide_map_pam = pd.read_csv(Path('homology_check_pam_twentytwo') / 'unique_guides_homology_summary.csv')
guide_map_host = pd.read_csv(Path('homology_check_host_twentytwo') / 'unique_guides_homology_summary.csv')

# Combine them to ensure we have the mapping for every guide_id
guide_map = pd.concat([guide_map_pam, guide_map_host]).drop_duplicates(subset=['guide_id'])

# 2. Extract the "bad" guide_ids from your filtered glibdown dataframe
bad_guide_ids = glibdown_filtered['guide_id'].unique()

# Filter the guide_map to only the bad guides, and extract their exact signatures
bad_guides_df = guide_map[guide_map['guide_id'].isin(bad_guide_ids)][['editor', 'protospacer', 'strand']].drop_duplicates()

# Convert to a set of tuples for lightning-fast, highly accurate lookups
bad_signatures = set(tuple(x) for x in bad_guides_df.to_numpy())

# 3. Load your minimal set (bib)
CRISPR_OUT_DIR = Path('crispr_results_twentytwo')
bib = pd.read_csv(CRISPR_OUT_DIR / 'sgrna_minimal_set.csv')

# Create the identical signature tuple for each row in the minimal set
bib['signature'] = list(zip(bib['editor'], bib['protospacer'], bib['strand']))

# Flag which guides exactly match the bad signatures
bib['is_bad_homology'] = bib['signature'].isin(bad_signatures)

# Split into clean and bad minimal guides
bib_clean = bib[~bib['is_bad_homology']].copy()
bib_bad = bib[bib['is_bad_homology']].copy()

print(f"Total sgRNAs in minimal set: {len(bib)}")
print(f"sgRNAs knocked out by strict homology: {len(bib_bad)}")
print(f"sgRNAs remaining (clean): {len(bib_clean)}\n")

# 4. Calculate Target Loss
all_targets = set()
for cov in bib['covers']:
    # Safely evaluate the string representation of the list
    if isinstance(cov, str):
        all_targets.update(ast.literal_eval(cov))
    else:
        all_targets.update(cov)
    
clean_targets = set()
for cov in bib_clean['covers']:
    if isinstance(cov, str):
        clean_targets.update(ast.literal_eval(cov))
    else:
        clean_targets.update(cov)

lost_targets = all_targets - clean_targets

print(f'{"="*40}')
print('MINIMAL SET TARGET LOSS REPORT')
print(f'{"="*40}')
print(f"Total targets originally covered: {len(all_targets):,}")
print(f"Targets covered by clean guides:  {len(clean_targets):,}")
if len(all_targets) > 0:
    print(f"Targets LOST:                     {len(lost_targets):,} ({(len(lost_targets)/len(all_targets))*100:.2f}%)")
else:
    print("Targets LOST:                     0 (0.00%)")
print(f'{"="*40}')




safe = pd.read_csv(CRISPR_OUT_DIR / 'candidates_safe.csv')

# 2. Apply the exact same signature mapping to the FULL pool
safe['signature'] = list(zip(safe['editor'], safe['protospacer'], safe['strand']))

# 3. Filter out the bad homology guides from the entire pool
safe_clean = safe[~safe['signature'].isin(bad_signatures)].copy()

print(f"Total candidates originally: {len(safe):,}")
print(f"Clean candidates remaining:  {len(safe_clean):,}")



def greedy_minimal_set(safe_df):
    df = safe_df.copy()
    # --- scoring (same as your script) ---
    df['rank_score'] = (
        df['efficiency_score'] *
        (1 - df['pct_early']) *
        df['pct_conserved_dn'].fillna(0.5)
    )
    df = df.sort_values('rank_score', ascending=False)
    # --- collapse to guide-level coverage ---
    guide_cov = (
        df.groupby(['editor', 'protospacer', 'strand'])
        .agg(
            covered=('query_id', lambda x: frozenset(x)),
            n_pids=('query_id', 'nunique'),
            mean_eff=('efficiency_score', 'mean'),
            mean_early=('pct_early', 'mean'),
            mean_cons=('pct_conserved_dn', 'mean'),
            family=('family', 'first'),
            editor_type=('editor_type', 'first')
        )
        .reset_index()
        .sort_values(['n_pids', 'mean_eff', 'mean_early'],
                     ascending=[False, False, True])
    )
    # --- greedy selection ---
    uncovered = set(df['query_id'].unique())
    sgrna_set = []
    while uncovered:
        best_new = set()
        best_row = None
        for _, r in guide_cov.iterrows():
            new = r['covered'] & uncovered
            if len(new) > len(best_new):
                best_new = new
                best_row = r

        if best_row is None:
            break
        sgrna_set.append({
            'editor':           best_row['editor'],
            'editor_type':      best_row['editor_type'],
            'protospacer':      best_row['protospacer'],
            'strand':           best_row['strand'],
            'family':           best_row['family'],
            'n_pids_covered':   len(best_new),
            'mean_efficiency':  round(best_row['mean_eff'], 3),
            'mean_pct_early':   round(best_row['mean_early'], 3),
            'mean_pct_cons_dn': round(best_row['mean_cons'], 3)
                                if pd.notna(best_row['mean_cons']) else None,
            'covers':           sorted(best_new),
        })
        uncovered -= best_new
    return pd.DataFrame(sgrna_set)

# 4. RE-RUN THE GREEDY ALGORITHM on the clean pool
# (This forces the algorithm to find backup guides for the ones we just deleted)
print("\nRe-running greedy set cover on clean candidates...")
new_bib = greedy_minimal_set(safe_clean)



new_bib.to_csv(CRISPR_OUT_DIR / 'sgrna_minimal_set_refined.csv')

import ast

def extract_targets(df):
    targets = set()
    for cov in df['covers']:
        if isinstance(cov, str):
            targets.update(ast.literal_eval(cov))
        else:
            targets.update(cov)
    return targets


# Targets covered by original minimal set
old_targets = extract_targets(bib)

# Targets covered by new minimal set
new_targets = extract_targets(new_bib)

# Lost targets
lost_targets = old_targets - new_targets

print(f"Targets originally covered: {len(old_targets):,}")
print(f"Targets still covered:      {len(new_targets):,}")
print(f"Targets LOST:               {len(lost_targets):,}")


#Targets originally covered: 42,656
#Targets still covered:      39,359
#Targets LOST:               3,297
#>>> 


#187 vs 193 sgRNAs.




######################################################################################################

"""
all_betalactamase_evo2_bystander.py
====================================
Enumerate CBE bystander mutations for ALL beta-lactamase families
(TEM, SHV, KPC, CTX-M, NDM, AmpC, BlaC) and score them via the
Evo2-40B API.

Run interactively (cell-by-cell in Jupyter / ipython) or as a plain
Python script.  No def main() wrapper — every section executes at
module level, left-to-right.

Fixes vs scale_up_API.py:
  1. NVIDIA_API_KEY moved to top-level config (no bare key buried ~line 714).
  2. 'requests', 'time' imported at the top (were missing until line 704).
  3. Taxonomy enrichment block made robust (KeyError on nuc_taxuid lookup
     was a silent crash risk; now wrapped in .get()).
  4. Prefix cache is now bounded (LRU-style dict eviction) to prevent
     unbounded RAM growth over large runs.
  5. Evo2 FASTA header now uses query_id only (no pipe-split confusion).
  6. Cross-scope summary extended to all families.
  7. All-families scope replaces the old TEM-only scope A and TEM-wide
     scope B; three new scopes: per-family, all-families, and
     priority (GOF-overlap-only) rows from the all-families enumeration.
  8. Mutation count printed before the API loop begins.
  9. Output CSV append-mode write buffered with flush-per-row (unchanged)
     but also prints a running ETA.
 10. Removed hard-coded output path duplication (was Path('bystander_evo2
     _results/A_TEM1/output.csv') repeated in two places).
"""

# =============================================================================
# SECTION 1 — IMPORTS  (ALL AT TOP — fixes missing 'requests'/'time')
# =============================================================================

import re
import os
import ast
import csv
import sys
import time
import itertools
import requests
from collections import defaultdict, OrderedDict
from pathlib import Path

import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

print('Imports OK.')


# =============================================================================
# SECTION 2 — PATHS & GLOBAL CONFIG
# =============================================================================

AMR_CSV     = Path('amrfindermapped_beta_lactamases.csv')
NUC_FA      = Path('card_gof_reference/all_query_sequences.fa')
SGRNA_CSV   = Path('crispr_results_twentytwo/sgrna_minimal_set_refined.csv')
GOF_NUC_CSV = Path('gof_mapping_results_twentytwo/gof_positions_per_pid.csv')
OUT_BASE    = Path('bystander_evo2_results_max_twentytwo')
OUT_BASE.mkdir(exist_ok=True)

PLSDB_META_PATH = Path('plsdb_meta')   # optional; taxonomy enrichment skipped if absent

# ── API config ────────────────────────────────────────────────────────────────
# Set NVIDIA_API_KEY in your environment:  export NVIDIA_API_KEY="nvapi-..."
# Or replace the fallback string below (not recommended for shared repos).
NVIDIA_API_KEY = os.getenv(
    'NVIDIA_API_KEY',
    'not leaking my key'
)
EVO2_URL = os.getenv(
    'EVO2_URL',
    'https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate'
)

# ── Enumeration settings ──────────────────────────────────────────────────────
MAX_RUN_LEN         = 4     # cap on contiguous-C run length before skipping run combos
INCLUDE_RUN_COMBOS  = True  # Tier 2: contiguous-C run subsets
INCLUDE_CODON_PAIRS = True  # Tier 3: same-codon C pairs

# ── Memory safety ─────────────────────────────────────────────────────────────
# Prefix-cache: how many unique prefixes to hold in RAM simultaneously.
# Each prefix can be up to ~1 kb; 5 000 entries ≈ 5 MB of strings + logit arrays.
# At ~512 floats per logit vector, 5 000 * 512 * 4 bytes ≈ 10 MB — very safe.
PREFIX_CACHE_MAX = 5_000

print('Paths & config set.')
print(f'  sgRNA set : {SGRNA_CSV}')
print(f'  Output    : {OUT_BASE}')
print(f'  Evo2 URL  : {EVO2_URL}')
print(f'  Run combos (Tier 2): {INCLUDE_RUN_COMBOS}, max run len: {MAX_RUN_LEN}')
print(f'  Codon pairs (Tier 3): {INCLUDE_CODON_PAIRS}')
print(f'  Prefix cache max size: {PREFIX_CACHE_MAX:,}')


# =============================================================================
# SECTION 3 — CONSTANTS
# =============================================================================


FAMILY_ALIASES = {
    'bla-ampc': 'AmpC', 'ampc': 'AmpC', 'blaampc': 'AmpC',
    'blac': 'BlaC',
    'tem': 'TEM', 'shv': 'SHV', 'ndm': 'NDM',
    'ctx-m': 'CTX-M', 'ctxm': 'CTX-M',
    'kpc': 'KPC',
}

STOP_CODONS = {'TAA', 'TAG', 'TGA'}

IUPAC = {
    'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'N': '[ACGT]',
    'R': '[AG]', 'Y': '[CT]', 'S': '[GC]', 'W': '[AT]',
    'K': '[GT]', 'M': '[AC]', 'B': '[CGT]', 'D': '[AGT]',
    'H': '[ACT]', 'V': '[ACG]',
}

BASE_EDITORS = {
    'BE3': {
        'type': 'CBE', 'cas_variant': 'SpCas9', 'PAM': 'NGG',
        'protospacer_len': 23, 'activity_window': (4, 8), 'buffer_bp': 2,
        'context_preference': {'T': 'best', 'C': 'good', 'A': 'good', 'G': 'poor'},
    },
    'SaBE3': {
        'type': 'CBE', 'cas_variant': 'SaCas9', 'PAM': 'NNGRRT',
        'protospacer_len': 21, 'activity_window': (4, 8), 'buffer_bp': 2,
        'context_preference': {'T': 'best', 'C': 'good', 'A': 'good', 'G': 'poor'},
    },
    'VQR-BE3': {
        'type': 'CBE', 'cas_variant': 'VQR-SpCas9', 'PAM': 'NGAN',
        'protospacer_len': 23, 'activity_window': (4, 5), 'buffer_bp': 1,
        'context_preference': {'T': 'best', 'C': 'good', 'A': 'good', 'G': 'poor'},
    },
    'EQR-BE3': {
        'type': 'CBE', 'cas_variant': 'EQR-SpCas9', 'PAM': 'NGAG',
        'protospacer_len': 23, 'activity_window': (4, 8), 'buffer_bp': 2,
        'context_preference': {'T': 'best', 'C': 'good', 'A': 'good', 'G': 'poor'},
    },
    'VRER-BE3': {
        'type': 'CBE', 'cas_variant': 'VRER-SpCas9', 'PAM': 'NGCG',
        'protospacer_len': 23, 'activity_window': (4, 8), 'buffer_bp': 2,
        'context_preference': {'T': 'best', 'C': 'good', 'A': 'good', 'G': 'poor'},
    },
    'SaKKH-BE3': {
        'type': 'CBE', 'cas_variant': 'SaKKH-Cas9', 'PAM': 'NNNRRT',
        'protospacer_len': 21, 'activity_window': (4, 8), 'buffer_bp': 2,
        'context_preference': {'T': 'best', 'C': 'good', 'A': 'good', 'G': 'poor'},
    },
    'CRISPR-cBEST': {
        'type': 'CBE', 'cas_variant': 'Streptomyces-optimised', 'PAM': 'NGG',
        'protospacer_len': 23, 'activity_window': (4, 10), 'buffer_bp': 2,
        'context_preference': {'T': 'best', 'A': 'good', 'G': 'reduced', 'C': 'poor'},
    },
    'BE4': {
        'type': 'CBE', 'cas_variant': 'SpCas9', 'PAM': 'NGG',
        'protospacer_len': 23, 'activity_window': (4, 8), 'buffer_bp': 2,
        'context_preference': {'T': 'best', 'C': 'good', 'A': 'good', 'G': 'reduced'},
        'context_window_modifiers': {
            'T': (3, 9), 'C': (4, 8), 'A': (4, 7), 'G': (5, 7),
        },
    },
    # ABEs — present for PAM scanning completeness, skipped in enumeration
    'ABE8e': {
        'type': 'ABE', 'cas_variant': 'SpCas9', 'PAM': 'NGG',
        'protospacer_len': 23, 'activity_window': (4, 8), 'buffer_bp': 2,
        'context_preference': {'T': 'best', 'A': 'good', 'G': 'good', 'C': 'moderate'},
        'context_window_modifiers': {
            'T': (3, 11), 'A': (4, 8), 'G': (4, 8), 'C': (4, 8),
        },
    },
    'Valdez_narrow_ABE': {
        'type': 'ABE', 'cas_variant': 'SpCas9', 'PAM': 'NGG',
        'protospacer_len': 23, 'activity_window': (4, 7), 'buffer_bp': 1,
        'context_preference': {'T': 'best', 'G': 'poor', 'A': 'poor', 'C': 'good'},
    },
    'CRISPR-aBEST': {
        'type': 'ABE', 'cas_variant': 'Streptomyces-optimised', 'PAM': 'NGG',
        'protospacer_len': 23, 'activity_window': (1, 6), 'buffer_bp': 2,
        'context_preference': {'T': 'best', 'G': 'good', 'A': 'moderate', 'C': 'poor'},
    },
}

CBE_EDITORS = {k: v for k, v in BASE_EDITORS.items() if v['type'] == 'CBE'}
_PAM_RE     = {
    name: re.compile(''.join(IUPAC[c] for c in ed['PAM'].upper()))
    for name, ed in BASE_EDITORS.items()
}

print(f'CBE editors: {sorted(CBE_EDITORS.keys())}')


# =============================================================================
# SECTION 4 — HELPERS
# =============================================================================

def eff_window(editor_name, preceding_base):
    """Context-adjusted activity window — 1-indexed protospacer positions."""
    ed = BASE_EDITORS[editor_name]
    w  = ed['activity_window']
    if preceding_base and 'context_window_modifiers' in ed:
        return ed['context_window_modifiers'].get(preceding_base.upper(), w)
    return w


def scan_pam_cbe(nt_seq, editor_name):
    """Yield PAM hit dicts (strand, protospacer, proto_start_cds, win_s, win_e)."""
    ed     = BASE_EDITORS[editor_name]
    plen   = ed['protospacer_len']
    pam_re = _PAM_RE[editor_name]
    w_s, w_e = ed['activity_window']
    seq    = nt_seq.upper()
    rc     = str(Seq(seq).reverse_complement())
    L      = len(seq)
    for strand, s in (('+', seq), ('-', rc)):
        for m in pam_re.finditer(s):
            proto_s = m.start() - plen
            if proto_s < 0:
                continue
            proto_seq_strand = s[proto_s: proto_s + plen]
            if strand == '+':
                proto_start_cds = proto_s
                protospacer     = proto_seq_strand
            else:
                proto_start_cds = L - proto_s - plen
                protospacer     = str(Seq(proto_seq_strand).reverse_complement())
            ws_cds = proto_start_cds + (w_s - 1)
            we_cds = proto_start_cds + (w_e - 1)
            if we_cds < 0 or ws_cds >= L:
                continue
            yield {
                'strand':          strand,
                'protospacer':     protospacer,
                'proto_start_cds': proto_start_cds,
                'win_s':           max(0, ws_cds),
                'win_e':           min(L - 1, we_cds),
            }


def apply_c_to_t(nt_seq, positions):
    """Return new CDS string with C→T at each position in `positions`."""
    seq = list(nt_seq)
    for p in positions:
        if seq[p] == 'C':
            seq[p] = 'T'
    return ''.join(seq)


def aa_changes(wt_nt, edited_nt):
    """
    Return list of 'WTposMUT' strings for every codon that changes.
    Synonymous changes noted as 'R104R(syn)'.
    """
    changes = []
    n = min(len(wt_nt), len(edited_nt)) // 3
    for i in range(n):
        wc = wt_nt[i*3: i*3+3]
        mc = edited_nt[i*3: i*3+3]
        if wc == mc:
            continue
        wa = str(Seq(wc).translate())
        ma = str(Seq(mc).translate())
        label = f'{wa}{i+1}{ma}' if wa != ma else f'{wa}{i+1}{ma}(syn)'
        changes.append(label)
    return changes


def contiguous_runs(positions):
    """
    Given a sorted list of integer positions, return a list of runs
    of consecutive integers.
    E.g. [3,4,7,10,11,12] -> [[3,4],[7],[10,11,12]]
    """
    if not positions:
        return []
    runs, cur = [], [positions[0]]
    for p in positions[1:]:
        if p == cur[-1] + 1:
            cur.append(p)
        else:
            runs.append(cur)
            cur = [p]
    runs.append(cur)
    return runs


def gof_overlaps(combo_positions, gof_wins):
    """True if any edited position's codon overlaps a GOF nucleotide window."""
    for p in combo_positions:
        codon_s = (p // 3) * 3
        codon_e = codon_s + 2
        if any(gs <= codon_e and codon_s <= ge for gs, ge in gof_wins):
            return True
    return False


# ── Evo2 editor type helpers ──────────────────────────────────────────────────

EDITOR_TYPE_MAP = {
    'BE3':               'CBE',
    'BE4':               'CBE',
    'SABE3':             'CBE',
    'SAKKH-BE3':         'CBE',
    'VQR-BE3':           'CBE',
    'EQR-BE3':           'CBE',
    'VRER-BE3':          'CBE',
    'CRISPR-CBEST':      'CBE',
    'ABE8E':             'ABE',
    'VALDEZ_NARROW_ABE': 'ABE',
    'CRISPR-ABEST':      'ABE',
    'CBE':               'CBE',
    'ABE':               'ABE',
}

EDITOR_EDITS = {
    'CBE': {'C': 'T', 'G': 'A'},
    'ABE': {'A': 'G', 'T': 'C'},
}

EDITOR_REVERSE = {
    cls: {mut: wt for wt, mut in edits.items()}
    for cls, edits in EDITOR_EDITS.items()
}


def resolve_editor_class(editor: str) -> str:
    return EDITOR_TYPE_MAP.get(editor.upper().strip(), 'CBE')


def get_wt_nucleotide(mut_nuc: str, editor: str) -> str:
    cls = resolve_editor_class(editor)
    return EDITOR_REVERSE.get(cls, {}).get(mut_nuc.upper(), mut_nuc.upper())


print('Helpers ready.')


# =============================================================================
# SECTION 5 — LOAD SHARED DATA
# =============================================================================

amr = pd.read_csv(AMR_CSV, low_memory=False)
amr = amr[amr['gene_name'].apply(lambda x: isinstance(x, str))]
amr = amr[amr['gene_family'].apply(lambda x: isinstance(x, str))]
amr['gene_family'] = amr['gene_family'].apply(
    lambda f: FAMILY_ALIASES.get(str(f).lower(), str(f)))
print(f'AMR table: {len(amr):,} rows, {amr["query_id"].nunique():,} PIDs')



FAMILIES_OF_INTEREST = sorted(amr['gene_family'].dropna().unique())


# Nucleotide sequences — trimmed to codon boundary
nuc_lookup = {}
for rec in SeqIO.parse(str(NUC_FA), 'fasta'):
    s = str(rec.seq).upper()
    nuc_lookup[rec.id] = s[:len(s) - len(s) % 3]
print(f'Nucleotide sequences: {len(nuc_lookup):,}')

# Selected CBE sgRNAs — load and parse 'covers' column
sgrna_all = pd.read_csv(SGRNA_CSV)
sgrna_cbe = sgrna_all[sgrna_all['editor_type'] == 'CBE'].copy()
print(f'sgRNAs total: {len(sgrna_all):,}   CBE: {len(sgrna_cbe):,}')


def _parse_covers(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


# guide_pid_map: (editor, protospacer, strand) -> [pid, ...]
guide_pid_map = {}
for _, row in sgrna_cbe.iterrows():
    key  = (row['editor'], row['protospacer'], row['strand'])
    pids = _parse_covers(row.get('covers', '[]'))
    guide_pid_map[key] = pids
print(f'CBE guide→PID map: {len(guide_pid_map):,} guides, '
      f'{sum(len(v) for v in guide_pid_map.values()):,} (guide,PID) pairs total')

# GOF nucleotide windows per PID
pid_gof_windows = defaultdict(set)
if GOF_NUC_CSV.exists():
    gof_nuc = pd.read_csv(GOF_NUC_CSV)
    for r in gof_nuc.itertuples(index=False):
        if pd.notna(r.query_nuc_start) and pd.notna(r.query_nuc_end):
            pid_gof_windows[r.query_id].add(
                (int(r.query_nuc_start), int(r.query_nuc_end)))
    print(f'GOF windows: {len(pid_gof_windows):,} PIDs')
else:
    print('WARNING: GOF nuc CSV not found — gof_overlap column will be empty')


# =============================================================================
# SECTION 6 — CORE STREAMING ENUMERATOR
# =============================================================================

CSV_COLS = [
    'query_id', 'gene_name', 'gene_family',
    'editor', 'protospacer', 'strand',
    'win_s_cds', 'win_e_cds',
    'n_c_in_window', 'combo_tier', 'combo_size',
    'edited_nt_positions',   # comma-sep 0-indexed CDS positions
    'edited_aa_pos_1',       # comma-sep 1-indexed AA positions (deduplicated)
    'aa_changes',            # e.g. "A42V; G238*"  or "synonymous"
    'has_premature_stop',
    'gof_codon_overlap',
    'edited_cds',
]


def run_scope(scope_pids, pid_to_gene, pid_to_family, out_dir, label):
    """
    Stream enumeration for one scope.  Writes CSV and FASTA incrementally.
    Returns (n_combo_rows, n_evo2_rows).
    """
    out_dir.mkdir(exist_ok=True)
    csv_path  = out_dir / 'bystander_mutations.csv'
    evo2_path = out_dir / 'bystander_EVO2_input.csv'
    fa_path   = out_dir / 'nuc_seqs_for_evo2.fa'

    # Seen set for Evo2 dedup: (pid, hash(edited_cds))
    seen_evo2 = set()

    n_combos = 0
    n_evo2   = 0
    n_pairs  = sum(
        1 for gkey, pids in guide_pid_map.items()
        for pid in pids if pid in scope_pids
    )
    processed_pairs = 0

    with (open(csv_path,  'w', newline='') as csv_fh,
          open(evo2_path, 'w', newline='') as evo2_fh,
          open(fa_path,   'w')             as fa_fh):

        csv_writer  = csv.DictWriter(csv_fh,  fieldnames=CSV_COLS)
        evo2_writer = csv.DictWriter(evo2_fh, fieldnames=CSV_COLS)
        csv_writer.writeheader()
        evo2_writer.writeheader()

        for gkey, pids in guide_pid_map.items():
            ed_name, protospacer, strand = gkey

            for pid in pids:
                if pid not in scope_pids:
                    continue

                processed_pairs += 1
                if processed_pairs % 1000 == 0:
                    print(f'  [{label}] {processed_pairs:,}/{n_pairs:,} pairs  '
                          f'| combos: {n_combos:,}  evo2: {n_evo2:,}')

                nt = nuc_lookup.get(pid)
                if nt is None:
                    continue

                L = len(nt)

                # ── Find PAM hit matching this guide on this PID's sequence ──
                hit_found = None
                for hit in scan_pam_cbe(nt, ed_name):
                    if hit['strand'] == strand and hit['protospacer'] == protospacer:
                        hit_found = hit
                        break
                if hit_found is None:
                    continue

                proto_start = hit_found['proto_start_cds']
                win_s       = hit_found['win_s']
                win_e       = hit_found['win_e']

                # ── Collect C positions passing context-adjusted window ──────
                c_positions = []
                for pos in range(win_s, min(win_e + 1, L)):
                    if nt[pos] != 'C':
                        continue
                    preceding    = nt[pos - 1] if pos > 0 else 'N'
                    pos_in_proto = (pos - proto_start) + 1
                    ctx_s, ctx_e = eff_window(ed_name, preceding)
                    if ctx_s <= pos_in_proto <= ctx_e:
                        c_positions.append(pos)

                if not c_positions:
                    continue

                k        = len(c_positions)
                gof_wins = pid_gof_windows.get(pid, set())
                gene     = pid_to_gene.get(pid, '')
                family   = pid_to_family.get(pid, '')

                # ── Build combo → tier map ──────────────────────────────────
                combo_tier_map = {}

                # Tier 1: singles
                for p in c_positions:
                    combo_tier_map[frozenset([p])] = 'single'

                # Tier 2: contiguous runs
                if INCLUDE_RUN_COMBOS:
                    for run in contiguous_runs(sorted(c_positions)):
                        if len(run) <= MAX_RUN_LEN:
                            for r in range(2, len(run) + 1):
                                for sub in itertools.combinations(run, r):
                                    fs = frozenset(sub)
                                    combo_tier_map.setdefault(fs, f'run{len(sub)}')

                # Tier 3: same-codon pairs/triples
                if INCLUDE_CODON_PAIRS:
                    by_codon = defaultdict(list)
                    for p in c_positions:
                        by_codon[p // 3].append(p)
                    for cps in by_codon.values():
                        if len(cps) >= 2:
                            for pair in itertools.combinations(cps, 2):
                                fs = frozenset(pair)
                                combo_tier_map.setdefault(fs, 'codon_pair')
                            if len(cps) >= 3:
                                fs = frozenset(cps)
                                combo_tier_map.setdefault(fs, 'codon_triple')

                # ── Emit one row per combo ──────────────────────────────────
                for combo_fs, tier in combo_tier_map.items():
                    combo      = tuple(sorted(combo_fs))
                    edited_nt  = apply_c_to_t(nt, combo)
                    changes    = aa_changes(nt, edited_nt)
                    has_stop   = any('*' in c for c in changes)
                    gof_hit    = gof_overlaps(combo, gof_wins)
                    aa_pos_dedup = ','.join(
                        dict.fromkeys(str(p // 3 + 1) for p in combo))

                    row = {
                        'query_id':            pid,
                        'gene_name':           gene,
                        'gene_family':         family,
                        'editor':              ed_name,
                        'protospacer':         protospacer,
                        'strand':              strand,
                        'win_s_cds':           win_s,
                        'win_e_cds':           win_e,
                        'n_c_in_window':       k,
                        'combo_tier':          tier,
                        'combo_size':          len(combo),
                        'edited_nt_positions': ','.join(str(p) for p in combo),
                        'edited_aa_pos_1':     aa_pos_dedup,
                        'aa_changes':          '; '.join(changes) if changes else 'synonymous',
                        'has_premature_stop':  has_stop,
                        'gof_codon_overlap':   gof_hit,
                        'edited_cds':          edited_nt,
                    }

                    csv_writer.writerow(row)
                    n_combos += 1

                    # Evo2 dedup: one call per unique (pid, edited sequence)
                    evo2_key = (pid, hash(edited_nt))
                    if evo2_key not in seen_evo2:
                        seen_evo2.add(evo2_key)
                        evo2_writer.writerow(row)
                        uid = (f"{pid}|{ed_name}|"
                               f"nt{'_'.join(str(p) for p in combo)}")
                        fa_fh.write(f'>{uid}\n{edited_nt}\n')
                        n_evo2 += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    df_s = pd.read_csv(csv_path, usecols=[
        'query_id', 'editor', 'protospacer', 'strand',
        'combo_tier', 'combo_size', 'has_premature_stop', 'gof_codon_overlap',
    ])
    n_pids_scope   = df_s['query_id'].nunique()
    n_guides_scope = df_s[['editor','protospacer','strand']].drop_duplicates().shape[0]

    lines = [
        f'Scope : {label}',
        f'PIDs with mutations enumerated : {n_pids_scope:>8,}',
        f'Unique CBE guides applied      : {n_guides_scope:>8,}',
        f'Total combo rows               : {n_combos:>8,}',
        f'Unique Evo2 calls (deduped)    : {n_evo2:>8,}',
        f'  of which GOF overlap         : {int(df_s["gof_codon_overlap"].sum()):>8,}',
        f'  of which premature stop      : {int(df_s["has_premature_stop"].sum()):>8,}',
        '',
        'By tier:',
    ]
    for tier, cnt in df_s['combo_tier'].value_counts().sort_index().items():
        lines.append(f'  {tier:<15s} : {cnt:,}')
    lines += ['', 'By combo size:']
    for sz, cnt in df_s['combo_size'].value_counts().sort_index().items():
        lines.append(f'  size {sz} : {cnt:,}')
    lines += ['', 'By editor:']
    for ed, cnt in df_s['editor'].value_counts().items():
        lines.append(f'  {ed:<20s} : {cnt:,}')

    summary = '\n'.join(lines) + '\n'
    (out_dir / 'summary.txt').write_text(summary)
    print(f'\n--- {label} ---')
    print(summary)
    return n_combos, n_evo2


print('Enumerator ready.')


# =============================================================================
# SECTION 7 — SCOPE: PER-FAMILY (one output dir per family)
# =============================================================================

scope_results = {}   # label -> (n_combos, n_evo2, csv_path)

for family in FAMILIES_OF_INTEREST:
    print('\n' + '='*60)
    print(f'SCOPE: {family}')
    print('='*60)

    fam_rows = amr[amr['gene_family'] == family].copy()
    fam_pids = set(fam_rows['query_id'].unique())

    if not fam_pids:
        print(f'  No PIDs for {family} — skipping.')
        continue

    pid_to_gene_fam   = fam_rows.set_index('query_id')['gene_name'].to_dict()
    pid_to_family_fam = fam_rows.set_index('query_id')['gene_family'].to_dict()

    label    = family.replace('-', '_')
    out_dir  = OUT_BASE / f'family_{label}'
    print(f'  PIDs: {len(fam_pids):,}')
    n_c, n_e = run_scope(fam_pids, pid_to_gene_fam, pid_to_family_fam, out_dir, family)
    scope_results[family] = (n_c, n_e, out_dir / 'bystander_EVO2_input.csv')


# =============================================================================
# SECTION 8 — SCOPE: ALL FAMILIES COMBINED
# =============================================================================

print('\n' + '='*60)
print('SCOPE: ALL families combined')
print('='*60)

# Only include families of interest (exclude any unmapped leftovers)
all_rows = amr[amr['gene_family'].isin(FAMILIES_OF_INTEREST)].copy()
all_pids = set(all_rows['query_id'].unique())

pid_to_gene_all   = all_rows.set_index('query_id')['gene_name'].to_dict()
pid_to_family_all = all_rows.set_index('query_id')['gene_family'].to_dict()

print(f'All-family PIDs: {len(all_pids):,}')
n_all, evo2_all = run_scope(
    all_pids, pid_to_gene_all, pid_to_family_all,
    OUT_BASE / 'all_families', 'all-families'
)
scope_results['ALL'] = (n_all, evo2_all, OUT_BASE / 'all_families' / 'bystander_EVO2_input.csv')


# =============================================================================
# SECTION 9 — CROSS-SCOPE SUMMARY
# =============================================================================

print('\n' + '='*60)
print('CROSS-SCOPE FEASIBILITY SUMMARY')
print('='*60)

cross_rows = []
for label, (n_c, n_e, _) in scope_results.items():
    cross_rows.append({'scope': label, 'total_combos': n_c, 'evo2_calls': n_e})

cross = pd.DataFrame(cross_rows)
print(cross.to_string(index=False))
cross.to_csv(OUT_BASE / 'cross_scope_summary.csv', index=False)

print(f'\nAll outputs under {OUT_BASE}/')
for p in sorted(OUT_BASE.rglob('*')):
    if p.is_file():
        kb = p.stat().st_size / 1024
        print(f'  {str(p.relative_to(OUT_BASE)):<65s}  {kb:>8.1f} KB')


# =============================================================================
# SECTION 10 — OPTIONAL TAXONOMY ENRICHMENT
# =============================================================================
# Adds plasmid taxonomy strings to the all-families Evo2 input CSV.
# Skipped gracefully if PLSDB metadata files are absent.

PID_nuccore_pattern = re.compile(r'^(.+?)_\d+_\d+')

all_fam_evo2_csv    = OUT_BASE / 'all_families' / 'bystander_EVO2_input.csv'
all_fam_evo2_tax    = OUT_BASE / 'all_families' / 'bystander_EVO2_input_with_tax_info.csv'

nuc_csv  = PLSDB_META_PATH / 'nuccore_only.csv'
tax_csv  = PLSDB_META_PATH / 'taxonomy.csv'

if nuc_csv.exists() and tax_csv.exists():
    print('\nLoading PLSDB taxonomy metadata...')
    nuc_df = pd.read_csv(nuc_csv)
    tax_df = pd.read_csv(tax_csv)

    nuc_taxuid = dict(zip(nuc_df['NUCCORE_ACC'], nuc_df['TAXONOMY_UID']))

    taxuid_kingdom = dict(zip(tax_df['TAXONOMY_UID'], tax_df['TAXONOMY_superkingdom']))
    taxuid_phylum  = dict(zip(tax_df['TAXONOMY_UID'], tax_df['TAXONOMY_phylum']))
    taxuid_classs  = dict(zip(tax_df['TAXONOMY_UID'], tax_df['TAXONOMY_class']))
    taxuid_order   = dict(zip(tax_df['TAXONOMY_UID'], tax_df['TAXONOMY_order']))
    taxuid_genus   = dict(zip(tax_df['TAXONOMY_UID'], tax_df['TAXONOMY_genus']))
    taxuid_species = dict(zip(tax_df['TAXONOMY_UID'], tax_df['TAXONOMY_species']))

    idf = pd.read_csv(all_fam_evo2_csv)
    all_plas = list(set(
        PID_nuccore_pattern.match(x).group(1)
        if PID_nuccore_pattern.match(x) else None
        for x in idf['query_id'].tolist()
    ))
    all_plas = [p for p in all_plas if p is not None]

    plas_to_taxstring = {}
    for plas in all_plas:
        tax_uid = nuc_taxuid.get(plas)   # FIX: use .get() not direct key access
        if tax_uid is None:
            continue
        kingdom = taxuid_kingdom.get(tax_uid)
        phylum  = taxuid_phylum.get(tax_uid)
        classs  = taxuid_classs.get(tax_uid)
        order   = taxuid_order.get(tax_uid)
        genus   = taxuid_genus.get(tax_uid)
        species = taxuid_species.get(tax_uid)
        if kingdom and phylum and order and genus and species:
            tax_string = (f'k__[{kingdom}];p__[{phylum}];c__[{classs}];'
                          f'o__[{order}];g__[{genus}];s__[{species}]|')
            plas_to_taxstring[plas] = tax_string

    idf['plasmid'] = [
        PID_nuccore_pattern.match(x).group(1)
        if PID_nuccore_pattern.match(x) else None
        for x in idf['query_id'].tolist()
    ]
    idf['taxonomy_string'] = idf['plasmid'].map(plas_to_taxstring)
    idf.to_csv(all_fam_evo2_tax, index=False)
    print(f'Taxonomy-enriched CSV → {all_fam_evo2_tax}')
    MUTATIONS_CSV = all_fam_evo2_tax
else:
    print('\nPLSDB metadata not found — skipping taxonomy enrichment.')
    print(f'  Expected: {nuc_csv}  and  {tax_csv}')
    MUTATIONS_CSV = all_fam_evo2_csv

####!!!!! here is new code -pasted in fetch_evo2_logits from previous
# =============================================================================
# SECTION 11 — EVO2 API SCORING (TWO-PASS OPTIMIZED)
# =============================================================================
import json
 
OUTPUT_CSV = OUT_BASE / 'all_families' / 'evo2_scored_output.csv'
FASTA_FILE = OUT_BASE / 'all_families' / 'nuc_seqs_for_evo2.fa'
LOGITS_CACHE_FILE = OUT_BASE / 'all_families' / 'prefix_logits_cache.json'
 
headers_api = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Content-Type":  "application/json",
}
 
print(f'\nLoading FASTA for Evo2 scoring: {FASTA_FILE}')
edited_sequences = {}
for record in SeqIO.parse(str(FASTA_FILE), 'fasta'):
    qid_raw = record.id
    edited_sequences[qid_raw] = str(record.seq).upper()
 
print(f'Loading mutation candidates CSV: {MUTATIONS_CSV}')
df_mut = pd.read_csv(MUTATIONS_CSV)
if 'Unnamed: 0' in df_mut.columns:
    df_mut = df_mut.drop(columns=['Unnamed: 0'])
 
def make_fasta_uid(query_id, editor, edited_nt_positions):
    pos_str = '_'.join(str(p) for p in str(edited_nt_positions).split(','))
    return f"{query_id}|{editor}|nt{pos_str}"
 
df_mut['_fasta_uid'] = df_mut.apply(
    lambda r: make_fasta_uid(r['query_id'], r['editor'], r['edited_nt_positions']),
    axis=1
)
 
# -----------------------------------------------------------------------------
# STEP 1: GATHER ALL UNIQUE PREFIXES
# -----------------------------------------------------------------------------
print("\nScanning dataset to find unique sequence prefixes...")
unique_prefixes = set()
valid_rows = []
for index, row in df_mut.iterrows():
    query_id = str(row['query_id'])
    nt_pos_str = str(row['edited_nt_positions'])
    fasta_uid = row['_fasta_uid']
    edited_seq = edited_sequences.get(fasta_uid) or edited_sequences.get(query_id)
    if not edited_seq:
        continue
    try:
        positions_0based = [int(p) for p in nt_pos_str.split(',')]
        score_pos = max(positions_0based)
    except ValueError:
        continue
    if score_pos >= len(edited_seq):
        continue
    editor = str(row.get('editor', 'CBE'))
    mut_nuc = edited_seq[score_pos]
    wt_nuc = get_wt_nucleotide(mut_nuc, editor)
    if mut_nuc not in 'ACGT' or wt_nuc not in 'ACGT':
        continue
    prefix_seq = edited_seq[:score_pos]
    unique_prefixes.add(prefix_seq)
    # Save parsed data so we don't have to recalculate it in Step 3
    valid_rows.append({
        'original_row': row,
        'prefix_seq': prefix_seq,
        'score_pos': score_pos,
        'wt_nuc': wt_nuc,
        'mut_nuc': mut_nuc
    })
 
 
print(f"Reduced {len(df_mut):,} mutations to {len(unique_prefixes):,} unique API calls!")
 
 
# -----------------------------------------------------------------------------
# STEP 2: FETCH AND CACHE LOGITS
# -----------------------------------------------------------------------------
# Load existing cache if we crashed halfway through
 
# ── Evo2 API call ─────────────────────────────────────────────────────────────
def fetch_evo2_logits(sequence: str, max_retries: int = 5):
    payload = {"sequence": sequence, "num_tokens": 1, "enable_logits": True}
    for attempt in range(max_retries):
        try:
            r = requests.post(url=EVO2_URL, headers=headers_api,
                              json=payload, timeout=60)
            if 400 <= r.status_code < 500 and r.status_code != 429:
                print(f'    [!] Permanent client error {r.status_code}: {r.text[:300]}')
                return None
            if r.status_code == 429 or r.status_code >= 500:
                r.raise_for_status()
            response_data = r.json()
            if 'logits' not in response_data or not response_data['logits']:
                return None
            logits = response_data['logits'][0]
            if len(logits) < 85:   # need at least ASCII index 84 (T)
                return None
            return logits
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f'    [!] Network error (attempt {attempt+1}/{max_retries}): {e}. '
                  f'Retrying in {wait}s...')
            time.sleep(wait)
        except (KeyError, ValueError, IndexError) as e:
            wait = 2 ** attempt
            print(f'    [!] Parse error (attempt {attempt+1}/{max_retries}): {e}. '
                  f'Retrying in {wait}s...')
            time.sleep(wait)
    return None
 
 
prefix_to_logits = {}
if LOGITS_CACHE_FILE.exists():
    with open(LOGITS_CACHE_FILE, 'r') as f:
        prefix_to_logits = json.load(f)
    print(f"Loaded {len(prefix_to_logits)} previously fetched prefixes from disk.")
prefixes_to_fetch = [p for p in unique_prefixes if p not in prefix_to_logits]
print(f"Fetching {len(prefixes_to_fetch)} new prefixes from Evo2 API...")
for i, prefix_seq in enumerate(prefixes_to_fetch):
    logits = fetch_evo2_logits(prefix_seq)
    if logits:
        prefix_to_logits[prefix_seq] = logits
    print(f"  [API Call {i+1}/{len(prefixes_to_fetch)}] fetched.")
    time.sleep(0.5) # Rate limiting
    # Save to disk every 100 calls to prevent data loss on crash
    if i > 0 and i % 100 == 0:
        with open(LOGITS_CACHE_FILE, 'w') as f:
            json.dump(prefix_to_logits, f)
 
# Final save
with open(LOGITS_CACHE_FILE, 'w') as f:
    json.dump(prefix_to_logits, f)
 

import csv

print("\nApplying precomputed logits to calculate scores...")
 
input_columns = [c for c in df_mut.columns if c != '_fasta_uid']
evo2_columns  = ['scored_nt_pos_0based', 'wt_nt', 'mut_nt', 
                 'wt_score', 'mut_score', 'delta_ll', 'prediction']

# Added newline='' which is required for the csv module to prevent double-spacing
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Write the header safely
    writer.writerow(input_columns + evo2_columns)
    
    for item in valid_rows:
        prefix = item['prefix_seq']
        logits = prefix_to_logits.get(prefix)
        if not logits:
            continue # Skip if API permanently failed for this prefix
            
        wt_nuc = item['wt_nuc']
        mut_nuc = item['mut_nuc']
        wt_score = logits[ord(wt_nuc)]
        mut_score = logits[ord(mut_nuc)]
        delta_ll = mut_score - wt_score
        prediction = 'Gain of Function / Stable' if delta_ll > 0 else 'Damaging / LoF'
        
        out_row = [str(item['original_row'][col]) if pd.notna(item['original_row'][col]) else '' 
                   for col in input_columns]
                   
        out_row.extend([
            str(item['score_pos']), wt_nuc, mut_nuc,
            f'{wt_score:.5f}', f'{mut_score:.5f}', f'{delta_ll:.5f}', prediction
        ])
        
        # Write the data row safely
        writer.writerow(out_row)
 
 
print(f"\nDone! Scored results written to: {OUTPUT_CSV}")
 
 
 
 
 
prediction_res_df = pd.read_csv(OUTPUT_CSV)
 
uh_oh = prediction_res_df.loc[prediction_res_df['prediction']=='Gain of Function / Stable']
 
 
 
 
 
 
# =============================================================================
# SECTION 12 — FILTER new_bib sgRNAs BASED ON Evo2 GOF PREDICTIONS,
#              THEN RE-RUN GREEDY SET COVER ON THE SURVIVING CANDIDATE POOL
# =============================================================================
 
print('\n' + '='*60)
print('FILTERING sgRNAs BASED ON Evo2 GOF PREDICTIONS')
print('='*60)
 
# -----------------------------------------------------------------------------
# 1. IDENTIFY GUIDES WITH ≥1 PREDICTED GOF BYSTANDER MUTATION
#    A guide is removed when ANY of its enumerated edits is predicted to be
#    "Gain of Function / Stable".  Damaging / LoF edits are fine — we want
#    to knock out the gene, so those are the intended outcome.
# -----------------------------------------------------------------------------
gof_preds = prediction_res_df[
    prediction_res_df['prediction'] == 'Gain of Function / Stable'
].copy()
 
print(f'GOF-predicted mutation rows  : {len(gof_preds):,}')
 
# Key = (query_id, editor, protospacer, strand)
# A guide is "GOF-tainted" if it can introduce a stabilising bystander edit
# in any of the PIDs it targets.
gof_guide_keys = set(
    zip(
        gof_preds['query_id'],
        gof_preds['editor'],
        gof_preds['protospacer'],
        gof_preds['strand']
    )
)
 
print(f'Unique (pid, guide) pairs with GOF risk: {len(gof_guide_keys):,}')
 
# A guide is excluded if it has a GOF call on *any* PID it touches.
gof_guide_signatures = set(
    (editor, protospacer, strand)
    for _, editor, protospacer, strand in gof_guide_keys
)
print(f'Distinct guide signatures to remove    : {len(gof_guide_signatures):,}')
 
# -----------------------------------------------------------------------------
# 2. LOAD THE HOMOLOGY-FILTERED MINIMAL SET (new_bib = SGRNA_CSV)
#    This is the set produced by the greedy re-run after homology filtering.
#    We use it only for reporting / diagnostics; the actual greedy re-run below
#    operates on the full candidate pool (safe_clean) so it can recruit
#    replacement guides that were not in new_bib.
# -----------------------------------------------------------------------------
sg = pd.read_csv(SGRNA_CSV)
print(f'\nHomology-filtered minimal set (new_bib): {len(sg):,} guides')
 
sg['_sig'] = list(zip(sg['editor'], sg['protospacer'], sg['strand']))
sg['remove_due_to_gof'] = sg['_sig'].isin(gof_guide_signatures)
 
removed_from_minimal = sg[sg['remove_due_to_gof']].drop(columns=['_sig'])
kept_from_minimal    = sg[~sg['remove_due_to_gof']].drop(columns=['_sig'])
 
print(f'Guides removed from minimal set (GOF)  : {len(removed_from_minimal):,}')
print(f'Guides retained in minimal set         : {len(kept_from_minimal):,}')
 
#
#
#============================================================
#FILTERING sgRNAs BASED ON Evo2 GOF PREDICTIONS
#============================================================
#GOF-predicted mutation rows  : 4,641
#Unique (pid, guide) pairs with GOF risk: 3,570
#Distinct guide signatures to remove    : 68
#
#Homology-filtered minimal set (new_bib): 193 guides
#Guides removed from minimal set (GOF)  : 68
#Guides retained in minimal set         : 125
#

import ast
import numpy as np

# -----------------------------------------------------------------------------
# 3. TARGET COVERAGE COMPARISON (minimal-set level, for reporting)
# -----------------------------------------------------------------------------
def parse_covers(val):
    # Safely handle lists or arrays to prevent 'ambiguous truth value' error
    if isinstance(val, list):
        return val
    if isinstance(val, (set, tuple, frozenset, np.ndarray)):
        return list(val)
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(val)
    except Exception:
        return []

sg['covers_list'] = sg['covers'].apply(parse_covers)
all_targets_before_gof = set(pid for sub in sg['covers_list'] for pid in sub)

kept_from_minimal['covers_list'] = kept_from_minimal['covers'].apply(parse_covers)
targets_still_in_minimal = set(
    pid for sub in kept_from_minimal['covers_list'] for pid in sub
)
targets_lost_from_minimal = all_targets_before_gof - targets_still_in_minimal

print(f'\nTargets in new_bib before GOF filter   : {len(all_targets_before_gof):,}')
print(f'Targets still covered by kept guides   : {len(targets_still_in_minimal):,}')
print(f'Targets no longer covered (need rescue): {len(targets_lost_from_minimal):,}')


# -----------------------------------------------------------------------------
# 4. RE-RUN GREEDY SET COVER on safe_clean minus GOF-tainted guides
#    safe_clean was built earlier (all homology-clean candidates).
#    We now additionally exclude any guide whose signature appears in
#    gof_guide_signatures.  This lets the greedy algorithm recruit fresh
#    guides that cover the targets orphaned by GOF removal.
# -----------------------------------------------------------------------------
print('\nRe-running greedy set cover after GOF exclusion...')

safe_clean['_sig'] = list(zip(safe_clean['editor'],
                               safe_clean['protospacer'],
                               safe_clean['strand']))
safe_post_gof = safe_clean[~safe_clean['_sig'].isin(gof_guide_signatures)].drop(
    columns=['_sig']
).copy()

# Also drop the helper column from safe_clean so it stays clean
safe_clean.drop(columns=['_sig'], inplace=True, errors='ignore')

print(f'Candidate pool after GOF exclusion     : {len(safe_post_gof):,} rows  '
      f'({safe_post_gof["query_id"].nunique():,} PIDs)')

# Reuse the greedy_minimal_set function defined earlier in this script.
# It operates on the full per-PID candidate pool, not just new_bib, so it
# can find replacement guides for targets that lost all their coverage.
final_bib = greedy_minimal_set(safe_post_gof)

print(f'\nFinal minimal set (post-GOF greedy)    : {len(final_bib):,} guide(s)')

# Coverage comparison: new_bib → final_bib
final_bib['covers_list'] = final_bib['covers'].apply(parse_covers)
final_targets = set(pid for sub in final_bib['covers_list'] for pid in sub)
targets_lost_vs_new_bib   = all_targets_before_gof - final_targets
targets_recovered_by_greedy = targets_lost_from_minimal - targets_lost_vs_new_bib

print(f'Targets covered by final set           : {len(final_targets):,}')
print(f'Targets lost vs new_bib (unrecoverable): {len(targets_lost_vs_new_bib):,}')
print(f'Targets recovered by greedy re-run     : {len(targets_recovered_by_greedy):,}')


# -----------------------------------------------------------------------------
# 5. SAVE OUTPUTS
# -----------------------------------------------------------------------------
FILTER_OUT = OUT_BASE / 'filtered_new_bib'
FILTER_OUT.mkdir(exist_ok=True, parents=True)
 
# Diagnostics on what was pruned from the minimal set
kept_from_minimal.drop(columns=['covers_list'], errors='ignore').to_csv(
    FILTER_OUT / 'new_bib_kept_guides.csv', index=False
)
removed_from_minimal.to_csv(FILTER_OUT / 'new_bib_removed_gof_guides.csv', index=False)
 
# The authoritative output: greedy-optimal set after both homology AND GOF filtering
final_bib.drop(columns=['covers_list'], errors='ignore').to_csv(
    FILTER_OUT / 'sgrna_final_set.csv', index=False
)
 
pd.DataFrame({'lost_query_id': sorted(targets_lost_vs_new_bib)}).to_csv(
    FILTER_OUT / 'lost_targets_vs_new_bib.csv', index=False
)
 
summary_lines = [
    'SECTION 12 — GOF FILTER + GREEDY RE-RUN',
    '=' * 50,
    f'GOF-predicted mutation rows              : {len(gof_preds):,}',
    f'Distinct guide signatures with GOF risk  : {len(gof_guide_signatures):,}',
    '',
    f'Homology-filtered minimal set (new_bib)  : {len(sg):,}',
    f'  Guides removed (GOF)                   : {len(removed_from_minimal):,}',
    f'  Guides retained                        : {len(kept_from_minimal):,}',
    '',
    f'Candidate pool after GOF exclusion       : {safe_post_gof["query_id"].nunique():,} PIDs',
    f'Final greedy minimal set                 : {len(final_bib):,} guides',
    '',
    f'Targets in new_bib                       : {len(all_targets_before_gof):,}',
    f'Targets covered by final set             : {len(final_targets):,}',
    f'Targets recovered by greedy re-run       : {len(targets_recovered_by_greedy):,}',
    f'Targets lost (unrecoverable)             : {len(targets_lost_vs_new_bib):,}',
]
summary_text = '\n'.join(summary_lines)
print('\n' + summary_text)
(FILTER_OUT / 'summary.txt').write_text(summary_text)
 
print(f'\nFiltered outputs written to: {FILTER_OUT}')
print(f'Final sgRNA set            : {FILTER_OUT / "sgrna_final_set.csv"}')




import ast
import numpy as np
import pandas as pd

# =============================================================================
# ITERATIVE GOF & GREEDY SET COVER LOOP
# =============================================================================

# 1. Define the evaluation function (Insert your Evo2 script here)
def evaluate_gof_for_new_guides(new_guides_df):
    """
    Takes a DataFrame of newly recruited sgRNAs, evaluates them for GOF, 
    and returns a set of signatures that failed the test.
    """
    new_bad_signatures = set()
    
    # -------------------------------------------------------------------------
    # TODO: INSERT YOUR MUTATION GENERATION & SCORING CODE HERE
    # 
    # 1. Generate mutations for the sequences targeted by `new_guides_df`
    # 2. Extract the prefix_seq, wt_nuc, mut_nuc
    # 3. Look up logits in `prefix_to_logits`
    # 4. Calculate delta_ll = mut_score - wt_score
    # 5. If delta_ll > 0 (Gain of Function):
    #        new_bad_signatures.add((editor, protospacer, strand))
    # -------------------------------------------------------------------------
    
    # Example logic mapping your previous script:
    # df_mut = generate_mutations_for_guides(new_guides_df)
    # for row in df_mut:
    #     delta_ll = calculate_delta_ll(row)
    #     if delta_ll > 0:
    #         new_bad_signatures.add((row['editor'], row['protospacer'], row['strand']))

    return new_bad_signatures

# 2. Setup initial tracking variables
safe_clean['_sig'] = list(zip(safe_clean['editor'], safe_clean['protospacer'], safe_clean['strand']))

# We already know these guides cause GOF from your first run
known_gof_signatures = set(gof_guide_signatures)

# We already checked the guides in the first minimal set ('sg'), so we don't need to re-evaluate them
sg['_sig'] = list(zip(sg['editor'], sg['protospacer'], sg['strand']))
checked_signatures = set(sg['_sig'])

iteration = 1
final_bib = pd.DataFrame()

# 3. Start the Iterative Loop
while True:
    print(f"\n{'='*50}")
    print(f" ITERATION {iteration}: GREEDY SET COVER & GOF CHECK")
    print(f"{'='*50}")
    
    # Remove all known GOF signatures from the candidate pool
    current_pool = safe_clean[~safe_clean['_sig'].isin(known_gof_signatures)]
    
    # Run the greedy algorithm
    print(f"Running greedy algorithm on {len(current_pool):,} fully safe candidates...")
    current_bib = greedy_minimal_set(current_pool)
    
    # If the algorithm fails to find anything, break
    if current_bib.empty:
        print("Error: No guides left to cover targets.")
        final_bib = current_bib
        break
        
    current_bib['_sig'] = list(zip(current_bib['editor'], current_bib['protospacer'], current_bib['strand']))
    
    # Find newly recruited guides that haven't been GOF-checked yet
    new_guides = current_bib[~current_bib['_sig'].isin(checked_signatures)].copy()
    
    print(f"Current minimal set size : {len(current_bib)} guides")
    print(f"Newly recruited guides   : {len(new_guides)} guides")
    
    # -------------------------------------------------------------------------
    # STOPPING CONDITION: If there are 0 new guides, we are completely safe!
    # -------------------------------------------------------------------------
    if len(new_guides) == 0:
        print("\nSUCCESS! All guides in the minimal set have cleared the GOF filter.")
        final_bib = current_bib.copy()
        break
        
    # Evaluate the new guides for GOF risk
    print("Evaluating new guides for GOF mutations...")
    new_gof_sigs = evaluate_gof_for_new_guides(new_guides) 
    
    print(f"Found {len(new_gof_sigs)} new GOF-risk guides.")
    
    # Update tracking sets
    checked_signatures.update(new_guides['_sig'])
    known_gof_signatures.update(new_gof_sigs)
    
    iteration += 1

# =============================================================================
# FINAL SAVING & REPORTING
# =============================================================================

# Clean up helper column
final_bib.drop(columns=['_sig', 'covers_list'], errors='ignore', inplace=True)

FILTER_OUT = OUT_BASE / 'filtered_new_bib'
FILTER_OUT.mkdir(exist_ok=True, parents=True)

# The authoritative output: greedy-optimal set after absolute iterative homology AND GOF filtering
final_bib.to_csv(FILTER_OUT / 'ULTIMATE_CLEAN_sgrna_final_set.csv', index=False)
print(f"\nUltimate clean sgRNA set saved to: {FILTER_OUT / 'ULTIMATE_CLEAN_sgrna_final_set.csv'}")

# Calculate final target coverage
def parse_covers(val):
    if isinstance(val, list): return val
    if isinstance(val, (set, tuple, frozenset, np.ndarray)): return list(val)
    if pd.isna(val): return []
    try: return ast.literal_eval(val)
    except Exception: return []

final_bib['covers_list'] = final_bib['covers'].apply(parse_covers)
final_targets = set(pid for sub in final_bib['covers_list'] for pid in sub)
targets_lost_overall = all_targets_before_gof - final_targets

print(f'\nTargets originally in new_bib            : {len(all_targets_before_gof):,}')
print(f'Targets safely covered by ULTIMATE set   : {len(final_targets):,}')
print(f'Targets completely lost (unrecoverable)  : {len(targets_lost_overall):,}')




total_baseline = len(all_input_pids)

safely_covered = len(final_targets)
untargetable = len(targets_lost_overall)

# Calculate percentages
pct_covered = (safely_covered / total_baseline) * 100
pct_untargetable = (untargetable / total_baseline) * 100

print(f"\n{'='*50}")
print(" FINAL COVERAGE METRICS")
print(f"{'='*50}")
print(f"Total target baseline          : {total_baseline:,}")
print(f"Targets safely covered         : {safely_covered:,} ({pct_covered:.2f}%)")
print(f"Targets UNTARGETABLE (Lost)    : {untargetable:,} ({pct_untargetable:.2f}%)")
#
#==================================================
# FINAL COVERAGE METRICS
#==================================================
#Total target baseline          : 42,656
#Targets safely covered         : 36,808 (86.29%)
#Targets UNTARGETABLE (Lost)    : 2,551 (5.98%)
#

import ast

# 1. Load the data
df = pd.read_csv('crispr_results_twentytwo/ULTIMATE_CLEAN_sgrna_final_set.csv')

# 2. Convert the 'covers' column from a string to an actual Python list
df['covers_list'] = df['covers'].apply(ast.literal_eval)

# 3. Sort by the number of PIDs covered (descending) so the "best" sgRNAs are at the top
df = df.sort_values('n_pids_covered', ascending=False).reset_index(drop=True)

# 4. Get the total universe of unique targets covered by the ENTIRE library
total_unique_targets = set()
for targets in df['covers_list']:
    total_unique_targets.update(targets)

total_count = len(total_unique_targets)

# 5. Function to calculate and print coverage for top N sgRNAs
def print_top_n_coverage(n):
    top_targets = set()
    for targets in df.head(n)['covers_list']:
        top_targets.update(targets)
        
    top_count = len(top_targets)
    pct = (top_count / total_count) * 100
    print(f"The Top {n} sgRNAs cover {top_count} / {total_count} targets ({pct:.2f}%)")

# Calculate for top 20 and top 30
print_top_n_coverage(20)
print_top_n_coverage(30)


# Counter(glib['overlapping_pfams'].tolist())
#Counter({'DDE_Tnp_IS240': 2027, 'Beta-lactamase2': 117, 'Cpn60_TCP1': 115, '': 62, 'rve_3': 38, 'Peptidase_M56; Transpeptidase': 35, 'Beta-lactamase': 29, 'DDE_Tnp_IS240; TetR_C_1; TetR_N': 12, 'TetR_C_1; TetR_N': 11, 'DDE_Tnp_IS240; Y2_Tnp': 6, 'rve; HTH_38; DDE_Tnp_IS240': 6, 'AAA_11; DDE_Tnp_IS240': 6, 'DDE_Tnp_IS240; TelA': 6, 'DUF6685; DDE_Tnp_IS240': 4, 'EAL': 2, 'DUF6685': 2, 'Transposase_20; DEDD_Tnp_IS110': 1, 'Y2_Tnp': 1, 'rve; HTH_38': 1, 'DDE_Tnp_Tn3; DUF4158': 1, 'TelA': 1, 'HTH_7; Resolvase; DDE_Tnp_IS240': 1, 'MCPsignal; PAS_3; HAMP': 1, 'AAA_11': 1, 'DDE_Tnp_IS66; DDE_Tnp_IS66_C; zf-IS66; LZ_Tnp_IS66': 1})
#>>> Counter(glib['overlapping_genes'].tolist())
#Counter({'': 2415, 'TEM-206': 26, 'BcIII': 10, 'SHV-105': 9, 'CARB-3': 7, "AAC6'-Ib7": 7, 'aadA3': 7, "AAC6'-Ib9": 6})
#>>>

#Cpn60_TCP1: This is GroEL, a highly conserved, absolutely essential Class I chaperonin. It forms a physical barrel that misfolded proteins go inside to be re-folded correctly. It is heavily upregulated during heat shock or stress.

#MCPsignal; PAS_3; HAMP: These domains almost always appear together in Methyl-accepting Chemotaxis Proteins (MCPs). They are trans-membrane sensor receptors. The bacteria use them to "smell" their environment (e.g., swimming toward food or away from toxins).

#EAL: This domain breaks down cyclic-di-GMP (c-di-GMP), a massive secondary messenger in bacteria that controls the switch between swimming (motile) and settling down to form a biofilm.

#The rest are mobile related, AA1 and HTH probably involved to some extent also





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import LogLocator
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================
CRISPR_DIR = Path('crispr_results_twentytwo') 

N_PERMS  = 10
N_STEPS  = 100

# Kept at 1 to allow direct interactive console execution without multiprocessing errors
WORKERS  = 1  

# Load files
print("Loading data...")
# We load the full safe dataset to know ALL possible target sequences
unfiltered_safe = pd.read_csv(CRISPR_DIR / 'candidates_safe.csv')
sgrna_set       = pd.read_csv(CRISPR_DIR / 'ULTIMATE_CLEAN_sgrna_final_set.csv')

OUT_DIR = CRISPR_DIR / 'ultimate'
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 2. HISTOGRAM PLOT
# =============================================================================
print("Generating coverage histogram...")
vals = sgrna_set['n_pids_covered'].values

fig, ax = plt.subplots(figsize=(8,5))
bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 40)
ax.hist(vals, bins=bins, color='darkred', edgecolor='black', linewidth=0.2)
ax.set_xscale('log')
ax.set_xlabel('Sequences targeted per sgRNA')
ax.set_ylabel('Number of sgRNAs')
ax.set_title('Distribution of sgRNA target hit coverage (Ultimate Set)')

median = np.median(vals)
mean   = np.mean(vals)
ax.axvline(median, linestyle='--', linewidth=1, color='darkorange', label=f'median = {median:.0f}')
ax.axvline(mean,   linestyle=':',  linewidth=1, color='darkorange', label=f'mean = {mean:.0f}')
ax.legend()
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / 'sgRNA_PID_coverage_hist.png', dpi=300)
plt.close()


# =============================================================================
# 3. PREPARE FOR SIMULATION
# =============================================================================
print("Preparing simulation data...")

# Define the absolute universe of target sequences
all_pids = set(unfiltered_safe['query_id'].unique())
total_pids_count = len(all_pids)

# The sgRNAs we are allowed to use
ultimate_guides = set(zip(sgrna_set['editor'], sgrna_set['protospacer'], sgrna_set['strand']))

# Map genes to ALL of their sequences (including untargetable ones)
gene_to_pids = {
    g: frozenset(grp['query_id'].unique())
    for g, grp in unfiltered_safe.groupby('gene_name')
}

# Only map guides that exist in the ultimate set
guide_to_pids = {}
for (ed, proto, strand), grp in unfiltered_safe.groupby(['editor', 'protospacer', 'strand']):
    gk = (ed, proto, strand)
    if gk in ultimate_guides:
        guide_to_pids[gk] = frozenset(grp['query_id'].unique())

pid_to_guides = defaultdict(list)
for gk, pids in guide_to_pids.items():
    for pid in pids:
        pid_to_guides[pid].append(gk)

all_gene_names = sorted(gene_to_pids.keys())
n_genes_total  = len(all_gene_names)

checkpoints = sorted(set(
    np.unique(np.round(np.geomspace(1, n_genes_total, N_STEPS)).astype(int)).tolist()
))

def _greedy_incremental_sim(seed: int) -> list[tuple[int, int, int]]:
    rng            = np.random.default_rng(seed)
    perm           = rng.permutation(all_gene_names).tolist()
    uncovered:     set[str]   = set()
    chosen_guides: set[tuple] = set()
    
    results = []
    
    for step, gene in enumerate(perm, 1):
        truly_new = set()
        for pid in gene_to_pids[gene]:
            covered_by_chosen = any(pid in guide_to_pids[gk] for gk in chosen_guides)
            if not covered_by_chosen:
                truly_new.add(pid)
        
        uncovered |= truly_new
        candidate_guides = set()
        for pid in truly_new:
            candidate_guides.update(pid_to_guides[pid])
        candidate_guides -= chosen_guides
        
        local_uncovered = set(uncovered)
        while local_uncovered and candidate_guides:
            best_gk = max(candidate_guides, key=lambda gk: len(guide_to_pids[gk] & local_uncovered))
            newly_cov = guide_to_pids[best_gk] & local_uncovered
            
            if not newly_cov:
                break
                
            chosen_guides.add(best_gk)
            local_uncovered -= newly_cov
            
            candidate_guides.discard(best_gk)
            candidate_guides = {gk for gk in candidate_guides if guide_to_pids[gk] & local_uncovered}
            
        all_covered = set()
        for gk in chosen_guides:
            all_covered |= guide_to_pids[gk]
            
        # These are sequences of the genes we've looked at so far that CANNOT be targeted by ANY guide in the ultimate set
        real_uncovered = uncovered - all_covered
        
        if step in checkpoints:
            results.append((step, len(chosen_guides), len(real_uncovered)))
            
    return results


# =============================================================================
# 4. RUN SIMULATION & PLOT RESULTS
# =============================================================================
print(f'\nRunning {N_PERMS} gene permutation simulations '
      f'({n_genes_total} genes, {len(checkpoints)} checkpoints)')

all_runs: list[pd.DataFrame] = []

for seed in range(N_PERMS):
    rows = _greedy_incremental_sim(seed)
    df_r = pd.DataFrame(rows, columns=['n_genes', 'n_sgrnas', 'n_uncovered'])
    df_r['seed'] = seed
    all_runs.append(df_r)
    print(f'  permutation {seed} done')

scaling_df = pd.concat(all_runs, ignore_index=True)
scaling_df.to_csv(OUT_DIR / 'sgrna_scaling_by_gene.csv', index=False)
print(f'\nScaling data -> {OUT_DIR}/sgrna_scaling_by_gene.csv')

agg = (scaling_df.groupby('n_genes')[['n_sgrnas', 'n_uncovered']]
                  .agg(['mean', 'std']).reset_index())
agg.columns = ['n_genes', 'sgrnas_mean', 'sgrnas_std', 'uncov_mean', 'uncov_std']
agg['sgrnas_std'] = agg['sgrnas_std'].fillna(0)
agg['uncov_std']  = agg['uncov_std'].fillna(0)

x = agg['n_genes'].to_numpy()

# --- Plot 1: sgRNAs Required vs Genes ---
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(x, agg['sgrnas_mean'], color='black', linewidth=1.5, label='sgRNAs required')
ax1.fill_between(x,
                 agg['sgrnas_mean'] - agg['sgrnas_std'],
                 agg['sgrnas_mean'] + agg['sgrnas_std'],
                 alpha=0.3, color='grey', linewidth=0)

ax1.set_xlabel('Number of unique genes considered', fontsize=12)
ax1.set_ylabel('Cumulative sgRNAs required', color='black', fontsize=12)
ax1.set_xscale('log')
ax1.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=10))
ax1.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10)*0.1, numticks=50))
ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())

ax1.set_title(f'Number of sgRNAs needed vs Genes considered\n({n_genes_total} total genes)', fontsize=13)
ax1.spines[['top', 'right']].set_visible(False)
ax1.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig(OUT_DIR / 'sgrna_scaling_by_gene.png', dpi=300)
plt.close()
print(f'Scaling plot 1 -> {OUT_DIR}/sgrna_scaling_by_gene.png')

# --- Plot 2: Untargeted Sequences vs Genes ---
fig, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(x, agg['uncov_mean'], color='darkred', linewidth=1.5, label='Untargetable sequences')
ax2.fill_between(x,
                 agg['uncov_mean'] - agg['uncov_std'],
                 agg['uncov_mean'] + agg['uncov_std'],
                 alpha=0.2, color='darkred', linewidth=0)

ax2.set_xlabel('Number of unique genes considered', fontsize=12)
ax2.set_ylabel('Cumulative untargetable sequences', color='darkred', fontsize=12)
ax2.set_xscale('log')
ax2.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=10))
ax2.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10)*0.1, numticks=50))
ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())

ax2.set_title(f'Target sequences lost as genes are added\n(Sequences that cannot be hit by the Ultimate sgRNA set)', fontsize=13)
ax2.spines[['top', 'right']].set_visible(False)
ax2.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig(OUT_DIR / 'sgrna_untargeted_by_gene.png', dpi=300)
plt.close()
print(f'Scaling plot 2 -> {OUT_DIR}/sgrna_untargeted_by_gene.png')


