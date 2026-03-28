
import re
import math
import ast
from collections import defaultdict
from pathlib import Path
import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
import os
import csv
import subprocess
import tempfile
from pathlib import Path
from Bio import AlignIO
from Bio.SeqRecord import SeqRecord
import requests, time
from io import StringIO
from Bio.PDB import PDBParser
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.SeqUtils import seq1
from Bio.PDB import PDBParser


GOF_MUTATIONS = {
    'TEM': [
        #confirmed resistance-expanding
        {'ref_pos': 104, 'ref_aa': 'E', 'alt_aa': None,  'category': 'resistance',   'note': 'Glu104'},
        {'ref_pos': 164, 'ref_aa': 'R', 'alt_aa': None,  'category': 'resistance',   'note': 'Arg164'},
        {'ref_pos': 179, 'ref_aa': 'D', 'alt_aa': None,  'category': 'resistance',   'note': 'Asp179'},
        {'ref_pos': 237, 'ref_aa': 'A', 'alt_aa': None,  'category': 'resistance',   'note': 'Ala237'},
        {'ref_pos': 238, 'ref_aa': 'G', 'alt_aa': None,  'category': 'resistance',   'note': 'Gly238'},
        {'ref_pos': 240, 'ref_aa': 'E', 'alt_aa': None,  'category': 'resistance',   'note': 'Glu240'},
        {'ref_pos': 173, 'ref_aa': 'I', 'alt_aa': 'V',   'category': 'resistance',   'note': 'I173V'},
        {'ref_pos': 254, 'ref_aa': 'D', 'alt_aa': 'G',   'category': 'resistance',   'note': 'D254G'},
        #non-significant p but under adaptive selection
        {'ref_pos': 184, 'ref_aa': 'A', 'alt_aa': 'V',   'category': 'adaptive',     'note': 'A184V'},
        {'ref_pos': 265, 'ref_aa': 'T', 'alt_aa': 'M',   'category': 'adaptive',     'note': 'T265M'},
        {'ref_pos': 268, 'ref_aa': 'S', 'alt_aa': 'G',   'category': 'adaptive',     'note': 'S268G'},
        {'ref_pos': 175, 'ref_aa': 'N', 'alt_aa': 'I',   'category': 'adaptive',     'note': 'N175I'},
        {'ref_pos':  21, 'ref_aa': 'L', 'alt_aa': 'F',   'category': 'adaptive',     'note': 'L21F'},
        {'ref_pos': 224, 'ref_aa': 'A', 'alt_aa': 'V',   'category': 'adaptive',     'note': 'A224V'},
        {'ref_pos':  39, 'ref_aa': 'Q', 'alt_aa': 'K',   'category': 'adaptive',     'note': 'Q39K'},
        {'ref_pos': 275, 'ref_aa': 'R', 'alt_aa': 'L',   'category': 'adaptive',     'note': 'R275L'},
        #stability implicated
        {'ref_pos':  42, 'ref_aa': 'A', 'alt_aa': 'V',   'category': 'stability',    'note': 'Ala42Val'},
        {'ref_pos':  51, 'ref_aa': 'L', 'alt_aa': 'P',   'category': 'stability',    'note': 'Leu51Pro'},
        {'ref_pos':  69, 'ref_aa': None,'alt_aa': None,   'category': 'stability',    'note': 'pos69'},
        {'ref_pos': 130, 'ref_aa': None,'alt_aa': None,   'category': 'stability',    'note': 'pos130'},
        {'ref_pos': 187, 'ref_aa': None,'alt_aa': None,   'category': 'stability',    'note': 'pos187'},
        {'ref_pos': 244, 'ref_aa': None,'alt_aa': None,   'category': 'stability',    'note': 'pos244'},
        {'ref_pos': 275, 'ref_aa': None,'alt_aa': None,   'category': 'stability',    'note': 'pos275'},
        {'ref_pos': 276, 'ref_aa': None,'alt_aa': None,   'category': 'stability',    'note': 'pos276'},
        #stability clinical
        {'ref_pos':  31, 'ref_aa': 'V', 'alt_aa': 'R',   'category': 'stability_clinical', 'note': 'V31R'},
        {'ref_pos':  47, 'ref_aa': 'I', 'alt_aa': 'V',   'category': 'stability_clinical', 'note': 'I47V'},
        {'ref_pos':  60, 'ref_aa': 'F', 'alt_aa': 'Y',   'category': 'stability_clinical', 'note': 'F60Y'},
        {'ref_pos':  62, 'ref_aa': 'P', 'alt_aa': 'S',   'category': 'stability_clinical', 'note': 'P62S'},
        {'ref_pos':  78, 'ref_aa': 'G', 'alt_aa': 'A',   'category': 'stability_clinical', 'note': 'G78A'},
        {'ref_pos':  80, 'ref_aa': 'V', 'alt_aa': 'I',   'category': 'stability_clinical', 'note': 'V80I'},
        {'ref_pos':  82, 'ref_aa': 'S', 'alt_aa': 'H',   'category': 'stability_clinical', 'note': 'S82H'},
        {'ref_pos':  92, 'ref_aa': 'G', 'alt_aa': 'D',   'category': 'stability_clinical', 'note': 'G92D'},
        {'ref_pos': 120, 'ref_aa': 'R', 'alt_aa': 'G',   'category': 'stability_clinical', 'note': 'R120G'},
        {'ref_pos': 147, 'ref_aa': 'E', 'alt_aa': 'G',   'category': 'stability_clinical', 'note': 'E147G'},
        {'ref_pos': 153, 'ref_aa': 'H', 'alt_aa': 'R',   'category': 'stability_clinical', 'note': 'H153R'},
        {'ref_pos': 182, 'ref_aa': 'M', 'alt_aa': 'T',   'category': 'stability_clinical', 'note': 'M182T'},
        {'ref_pos': 184, 'ref_aa': 'A', 'alt_aa': 'V',   'category': 'stability_clinical', 'note': 'A184V'},
        {'ref_pos': 188, 'ref_aa': 'T', 'alt_aa': 'I',   'category': 'stability_clinical', 'note': 'T188I'},
        {'ref_pos': 201, 'ref_aa': 'L', 'alt_aa': 'P',   'category': 'stability_clinical', 'note': 'L201P'},
        {'ref_pos': 208, 'ref_aa': 'I', 'alt_aa': 'M',   'category': 'stability_clinical', 'note': 'I208M'},
        {'ref_pos': 224, 'ref_aa': 'A', 'alt_aa': 'V',   'category': 'stability_clinical', 'note': 'A224V'},
        {'ref_pos': 240, 'ref_aa': 'E', 'alt_aa': 'H',   'category': 'stability_clinical', 'note': 'E240H'},
        {'ref_pos': 241, 'ref_aa': 'R', 'alt_aa': 'H',   'category': 'stability_clinical', 'note': 'R241H'},
        {'ref_pos': 247, 'ref_aa': 'I', 'alt_aa': 'V',   'category': 'stability_clinical', 'note': 'I247V'},
        {'ref_pos': 265, 'ref_aa': 'T', 'alt_aa': 'M',   'category': 'stability_clinical', 'note': 'T265M'},
        {'ref_pos': 275, 'ref_aa': 'R', 'alt_aa': 'Q',   'category': 'stability_clinical', 'note': 'R275Q'},
        {'ref_pos': 275, 'ref_aa': 'R', 'alt_aa': 'L',   'category': 'stability_clinical', 'note': 'R275L'},
        {'ref_pos': 276, 'ref_aa': 'N', 'alt_aa': 'D',   'category': 'stability_clinical', 'note': 'N276D'},
        #from top section
        {'ref_pos':  15, 'ref_aa': 'A', 'alt_aa': 'T',   'category': 'resistance',   'note': 'A15T'},
        {'ref_pos':  39, 'ref_aa': 'Q', 'alt_aa': 'K',   'category': 'resistance',   'note': 'Q39K'},
        {'ref_pos':  39, 'ref_aa': 'Q', 'alt_aa': 'R',   'category': 'resistance',   'note': 'Q39R'},
        {'ref_pos':  51, 'ref_aa': 'L', 'alt_aa': 'P',   'category': 'resistance',   'note': 'L51P'},
        {'ref_pos': 139, 'ref_aa': 'L', 'alt_aa': None,  'category': 'resistance',   'note': 'L139'},
    ],
    'SHV': [
        {'ref_pos': 146, 'ref_aa': 'A', 'alt_aa': None,  'category': 'resistance',   'note': 'Ala146'},
        {'ref_pos': 156, 'ref_aa': 'G', 'alt_aa': None,  'category': 'resistance',   'note': 'Gly156'},
        {'ref_pos': 169, 'ref_aa': 'L', 'alt_aa': None,  'category': 'resistance',   'note': 'Leu169'},
        {'ref_pos': 179, 'ref_aa': 'D', 'alt_aa': None,  'category': 'resistance',   'note': 'Asp179'},
        {'ref_pos': 205, 'ref_aa': 'R', 'alt_aa': None,  'category': 'resistance',   'note': 'Arg205'},
        {'ref_pos': 238, 'ref_aa': 'G', 'alt_aa': None,  'category': 'resistance',   'note': 'Gly238'},
        {'ref_pos': 240, 'ref_aa': 'E', 'alt_aa': None,  'category': 'resistance',   'note': 'Glu240'},
        {'ref_pos':  69, 'ref_aa': None,'alt_aa': None,   'category': 'stability',    'note': 'pos69'},
        {'ref_pos': 130, 'ref_aa': None,'alt_aa': None,   'category': 'stability',    'note': 'pos130'},
        {'ref_pos': 187, 'ref_aa': None,'alt_aa': None,   'category': 'stability',    'note': 'pos187'},
        {'ref_pos': 244, 'ref_aa': None,'alt_aa': None,   'category': 'stability',    'note': 'pos244'},
        {'ref_pos': 275, 'ref_aa': None,'alt_aa': None,   'category': 'stability',    'note': 'pos275'},
        {'ref_pos': 276, 'ref_aa': None,'alt_aa': None,   'category': 'stability',    'note': 'pos276'},
    ],
    'NDM': [
        {'ref_pos': 135, 'ref_aa': 'S', 'alt_aa': None,  'category': 'resistance',   'note': 'NDM-1 135S'},
        {'ref_pos': 154, 'ref_aa': 'M', 'alt_aa': 'L',  'category': 'resistance',   'note': 'NDM-1 M154L'},
    ],
    'AmpC': [
        {'ref_pos': 150, 'ref_aa': 'Y', 'alt_aa': None,  'category': 'resistance',   'note': 'Y150'},
        {'ref_pos': 346, 'ref_aa': 'N', 'alt_aa': None,  'category': 'resistance',   'note': 'N346'},
        {'ref_pos': 237, 'ref_aa': 'S', 'alt_aa': 'H',   'category': 'resistance',   'note': 'S237H'},
        {'ref_pos': 148, 'ref_aa': 'R', 'alt_aa': 'P',   'category': 'resistance',   'note': 'R148P'},
    ],
    'BlaC': [
        {'ref_pos': 105, 'ref_aa': 'I', 'alt_aa': 'F',   'category': 'resistance',   'note': 'I105F'},
        {'ref_pos': 184, 'ref_aa': 'H', 'alt_aa': 'R',   'category': 'resistance',   'note': 'H184R'},
        {'ref_pos': 263, 'ref_aa': 'V', 'alt_aa': 'I',   'category': 'resistance',   'note': 'V263I'},
    ],
    'CTX-M': [
        {'ref_pos':  77, 'ref_aa': 'A', 'alt_aa': 'V',   'category': 'stability',    'note': 'A77V'},
    ],
    'KPC': [
        #KPC-3/11 vs KPC-2
        {'ref_pos': 104, 'ref_aa': 'P', 'alt_aa': 'R',   'category': 'resistance',   'note': 'P104R'},
        {'ref_pos': 104, 'ref_aa': 'P', 'alt_aa': 'L',   'category': 'resistance',   'note': 'P104L'},
        {'ref_pos': 240, 'ref_aa': 'V', 'alt_aa': 'G',   'category': 'resistance',   'note': 'V240G'},
        {'ref_pos': 240, 'ref_aa': 'V', 'alt_aa': 'A',   'category': 'resistance',   'note': 'V240A'},
        {'ref_pos': 274, 'ref_aa': 'H', 'alt_aa': 'Y',   'category': 'resistance',   'note': 'H274Y'},
    ],
}


#family name aliases to norm for lookups
FAMILY_ALIASES = {
    'bla-ampc': 'AmpC', 'ampc': 'AmpC', 'blaampc': 'AmpC',
    'blac': 'BlaC',
    'tem': 'TEM', 'shv': 'SHV', 'ndm': 'NDM',
    'ctx-m': 'CTX-M', 'ctxm': 'CTX-M',
    'kpc': 'KPC',
}

def get_gof(family):
    key = FAMILY_ALIASES.get(family.lower(), family)
    return GOF_MUTATIONS.get(key, [])




BASE_EDITORS = {
    'BE3': {
        'type':               'CBE',           
        'cas_variant':        'SpCas9',
        'PAM':                'NGG',
        'PAM_position':       '21-23',          #ie within 23nt protospacer
        'protospacer_len':    23,
        'activity_window':    (4, 8),           #1-indexed from 5' of protospacer
        'window_threshold':   0.50,             #50% of max activity defines window
        'buffer_bp':          2,                #extra bp either side for safety
        'context_preference': {                 #preceding base for target C
            'T': 'best', 'C': 'good', 'A': 'good', 'G': 'poor'
        },
        'ref': 'Komor et al. 2016 Nature; Kim et al. 2017 Nat Biotechnol',
    },
    'SaBE3': {
        'type':               'CBE',  #C->T
        'cas_variant':        'SaCas9',
        'PAM':                'NNGRRT',
        'PAM_position':       '22-27',
        'protospacer_len':    21,
        'activity_window':    (4, 8),
        'window_threshold':   0.50,
        'buffer_bp':          2,
        'efficiency_note':    "50-75% HEK293T",
        'context_preference': {
            'T': 'best', 'C': 'good', 'A': 'good', 'G': 'poor'
        },
        'ref': 'Kim et al. 2017 Nat Biotechnol',
    },
    'VQR-BE3': {
        'type':               'CBE',
        'cas_variant':        'VQR-SpCas9',
        'PAM':                'NGAN',
        'PAM_position':       '21-24',
        'protospacer_len':    23,
        'activity_window':    (4, 5),           #narrowed to 1-2nt in paper
        'window_threshold':   0.50,
        'buffer_bp':          1,
        'efficiency_note':    'up to 50%',
        'context_preference': {
            'T': 'best', 'C': 'good', 'A': 'good', 'G': 'poor'
        },
        'ref': 'Kim et al. 2017 Nat Biotechnol',
    },
    'EQR-BE3': {
        'type':               'CBE',
        'cas_variant':        'EQR-SpCas9',
        'PAM':                'NGAG',
        'PAM_position':       '21-24',
        'protospacer_len':    23,
        'activity_window':    (4, 8),
        'window_threshold':   0.50,
        'buffer_bp':          2,
        'efficiency_note':    'up to 50%',
        'context_preference': {
            'T': 'best', 'C': 'good', 'A': 'good', 'G': 'poor'
        },
        'ref': 'Kim et al. 2017 Nat Biotechnol',
    },
    'VRER-BE3': {
        'type':               'CBE',
        'cas_variant':        'VRER-SpCas9',
        'PAM':                'NGCG',
        'PAM_position':       '21-24',
        'protospacer_len':    23,
        'activity_window':    (4, 8),
        'window_threshold':   0.50,
        'buffer_bp':          2,
        'efficiency_note':    'up to 50%',
        'context_preference': {
            'T': 'best', 'C': 'good', 'A': 'good', 'G': 'poor'
        },
        'ref': 'Kim et al. 2017 Nat Biotechnol',
    },
    'SaKKH-BE3': {
        'type':               'CBE',
        'cas_variant':        'SaKKH-Cas9',
        'PAM':                'NNNRRT',
        'PAM_position':       '22-27',
        'protospacer_len':    21,
        'activity_window':    (4, 8),
        'window_threshold':   0.50,
        'buffer_bp':          2,
        'efficiency_note':    'up to 62%',
        'context_preference': {
            'T': 'best', 'C': 'good', 'A': 'good', 'G': 'poor'
        },
        'ref': 'Kim et al. 2017 Nat Biotechnol',
    },
    'Valdez_narrow_ABE': {
        'type':               'ABE',           #A->G
        'cas_variant':        'SpCas9',
        'PAM':                'NGG',
        'PAM_position':       '21-23',
        'protospacer_len':    23,
        'activity_window':    (4, 7),           #narrowed to 4nt, positions 4-7
        'window_threshold':   0.20,             #defined at 20% of max, whyyyy
        'buffer_bp':          1,
        'efficiency_note':    'reduced bystander editing',
        'context_preference': {                 #preceding base for target A
            'T': 'best', 'G': 'poor', 'A': 'poor', 'C': 'good'
        },
        'ref': 'Valdez et al. 2025 Nat Commun',
    },
    'CRISPR-cBEST': {
        'type':               'CBE',
        'cas_variant':        'Streptomyces-optimised',
        'PAM':                'NGG',
        'PAM_position':       '21-23',
        'protospacer_len':    23,
        'activity_window':    (4, 10),          #7nt window positions 4-10
        'optimal_subwindow':  (2, 4),           #optimal within window (relative to window start)
        'window_threshold':   0.50,
        'buffer_bp':          2,
        'context_preference': {
            'T': 'best', 'A': 'good', 'G': 'reduced',  #C-G* reduced by ~half
            'C': 'poor'
        },
        'efficiency_note':    'C preceded by G reduces activity ~50%',
        'ref': 'Tong et al. 2019 PNAS',
    },
    'CRISPR-aBEST': {
        'type':               'ABE',
        'cas_variant':        'Streptomyces-optimised',
        'PAM':                'NGG',
        'PAM_position':       '21-23',
        'protospacer_len':    23,
        'activity_window':    (1, 6),           #6nt window
        'optimal_subwindow':  (1, 3),           #best positions 1-3 within window
        'window_threshold':   0.50,
        'buffer_bp':          2,
        'context_preference': {                 #preceding base for target A: TA>GA>AA>CA
            'T': 'best', 'G': 'good', 'A': 'moderate', 'C': 'poor'
        },
        'ref': 'Tong et al. 2019 PNAS',
    },
    'BE4': {
        'type':               'CBE',
        'cas_variant':        'SpCas9',
        'PAM':                'NGG',
        'PAM_position':       '21-23',
        'protospacer_len':    23,
        'activity_window':    (4, 8),
        'window_threshold':   0.50,
        'buffer_bp':          2,
        'context_preference': {
            #Pallaseni et al. 2022: preceding T broadens to 3-9, preceding A/G -> 4-7
            #preceding C -> canonical 4-8
            'T': 'best',   #window 3-9 (broadened)
            'C': 'good',   #canonical 4-8
            'A': 'good',   #window 4-7
            'G': 'reduced' #window 5-7
        },
        'context_window_modifiers': {
            #how the window shifts per preceding base
            'T': (3, 9),
            'C': (4, 8),
            'A': (4, 7),
            'G': (5, 7),
        },
        'ref': 'Komor et al. 2016; Pallaseni et al. 2022 NAR',
    },
    'ABE8e': {
        'type':               'ABE',
        'cas_variant':        'SpCas9',
        'PAM':                'NGG',
        'PAM_position':       '21-23',
        'protospacer_len':    23,
        'activity_window':    (4, 8),
        'window_threshold':   0.50,
        'buffer_bp':          2,
        'efficiency_note':    'slightly elevated editing positions 9-10',
        'context_preference': {
            #Pallaseni 2022: preceding T -> 3-11, preceding A/G -> 4-8
            'T': 'best',   #window 3-11 (broadened)
            'A': 'good',   #window 4-8
            'G': 'good',   #window 4-8
            'C': 'moderate'
        },
        'context_window_modifiers': {
            'T': (3, 11),
            'A': (4, 8),
            'G': (4, 8),
            'C': (4, 8),
        },
        'ref': 'Richter et al. 2020; Pallaseni et al. 2022 NAR',
    },
}


AMR_CSV         = 'amrfindermapped_beta_lactamases.csv'
MSA_DIR         = Path('beta_lactamase_msa_results/gene_family')
NUC_FA          = Path('card_gof_reference/all_query_sequences.fa')
PROT_FA         = Path('card_gof_reference/all_query_sequences_prot.fa')
GOF_OUT_DIR     = Path('gof_mapping_results')
THREADS         = 4

#families to analyse must match keys in GOF_MUTATIONS
FAMILIES_OF_INTEREST = list(GOF_MUTATIONS.keys())

GOF_OUT_DIR.mkdir(exist_ok=True)


PID_RE = re.compile(r'^(.+?_\d+_\d+)(?:_.+)?$')


test = pd.read_csv(AMR_CSV, low_memory=False)
test = test.rename(columns={'Element name': 'gene_name'})
test = test[test['gene_name'].apply(lambda x: isinstance(x, str))]
test = test[test['gene_family'].apply(lambda x: isinstance(x, str))]


def norm_family(f):
    return FAMILY_ALIASES.get(str(f).lower(), str(f))

test['gene_family'] = test['gene_family'].apply(norm_family)


PDB_REFS = {
    'TEM':   ('1BTL', 'A'),
    'SHV':   ('1SHV', 'A'),
    'CTX-M': ('1YLJ', 'A'),   #ie CTX-M-15
    'KPC':   ('2OV5', 'A'),
    'BlaC':  ('2GDN', 'A'),
    'AmpC':  ('2BLS', 'A'),
    'NDM':   ('3Q6X', 'A'),
}


#use PDB sequences to map lit GOF 
def fetch_pdb(pdb_id):
    r = requests.get(f'https://files.rcsb.org/download/{pdb_id}.pdb', timeout=30)
    r.raise_for_status()
    return r.text


def extract_residues(pdb_text, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('tmp', StringIO(pdb_text))
    residues = {}
    for res in structure[0][chain_id].get_residues():
        hetflag, seq_id, icode = res.get_id()
        if hetflag.strip():
            continue
        residues[seq_id] = seq1(res.get_resname(), undef_code='X')
    return residues


pdb_residue_maps  = {}
pdb_linear_seqs   = {}   #family -> (seq_str, [res_numbers])

for family, (pdb_id, chain) in PDB_REFS.items():
    print(f'get {pdb_id} ({family})')
    try:
        res_map = extract_residues(fetch_pdb(pdb_id), chain)
        pdb_residue_maps[family] = res_map
        sorted_res   = sorted((k,v) for k,v in res_map.items() if k > 0)
        seq_str      = ''.join(aa for _,aa in sorted_res)
        res_numbers  = [k for k,_ in sorted_res]
        pdb_linear_seqs[family] = (seq_str, res_numbers)
        print(f'  OK: {len(sorted_res)} residues, {res_numbers[0]}–{res_numbers[-1]}')
    except Exception as e:
        print(f'  FAILED: {e}')
    time.sleep(0.3)



LITERATURE_TO_FASTA_INDEX = {}   #family -> {lit_pos: 0-idx or None}
GOF_VERIFIED   = {}              #family -> {lit_pos: {'ref_aa', 'idx', 'pdb_aa', 'alts'}}
GOF_UNRESOLVED = {}              #family -> {lit_pos: {'ref_aa', 'reason'}}

for family, gof_list in GOF_MUTATIONS.items():
    res_map = pdb_residue_maps.get(family)
    if not res_map:
        print(f'{family}: no PDB so skip')
        continue
    seq_str, res_numbers = pdb_linear_seqs[family]
    resnum_to_idx = {rn: i for i, rn in enumerate(res_numbers)}
    LITERATURE_TO_FASTA_INDEX[family] = {}
    GOF_VERIFIED[family]   = {}
    GOF_UNRESOLVED[family] = {}
    #deduplicate gof_list by ref_pos (keep first occurrence)
    seen = {}
    for gof in gof_list:
        pos = gof['ref_pos']
        if pos not in seen:
            seen[pos] = gof
    for pos, gof in sorted(seen.items()):
        ref_aa  = gof.get('ref_aa')
        alts    = gof.get('alt_aa', [])
        if pos not in res_map:
            LITERATURE_TO_FASTA_INDEX[family][pos] = None
            GOF_UNRESOLVED[family][pos] = {
                'ref_aa': ref_aa, 'reason': 'missing_from_pdb'
            }
            continue
        idx     = resnum_to_idx[pos]
        pdb_aa  = res_map[pos]
        LITERATURE_TO_FASTA_INDEX[family][pos] = idx
        if ref_aa is None:
            GOF_VERIFIED[family][pos] = {'ref_aa': None, 'idx': idx, 'pdb_aa': pdb_aa, 'alts': alts}
        elif pdb_aa == ref_aa:
            GOF_VERIFIED[family][pos] = {'ref_aa': ref_aa, 'idx': idx, 'pdb_aa': pdb_aa, 'alts': alts}
        else:
            GOF_VERIFIED[family][pos] = {'ref_aa': ref_aa, 'idx': idx, 'pdb_aa': pdb_aa, 'alts': alts,
                                          'pdb_mismatch': pdb_aa}
            print(f'  NB {family} pos {pos} ref_aa={ref_aa} but PDB has {pdb_aa} (idx={idx} still valid)')




print('\ncheck if the gof mapping seems right:')
for fam in GOF_VERIFIED:
    n_ok    = sum(1 for v in GOF_VERIFIED[fam].values() if 'pdb_mismatch' not in v and v['ref_aa'])
    n_mm    = sum(1 for v in GOF_VERIFIED[fam].values() if 'pdb_mismatch' in v)
    n_unres = len(GOF_UNRESOLVED[fam])
    print(f'  {fam}: {n_ok} clean, {n_mm} pdb_mismatch (idx valid), {n_unres} unresolved (None)')



PDB_FASTA_DIR = Path('pdb_references')
OUT_FASTA_DIR   = Path('gof_mapping_results/query_fastas')
OUT_FASTA_DIR.mkdir(parents=True, exist_ok=True)


#my gene family names vs the ones PDB uses to make sure theres no fuckery
FAMILY_NORM = {
    'TEM': 'TEM', 'SHV': 'SHV', 'KPC': 'KPC',
    'NDM': 'NDM', 'CTX-M': 'CTX-M', 'CTXM': 'CTX-M',
    'AmpC': 'AmpC', 'AMPC': 'AmpC',
    'BlaC': 'BlaC', 'BLAC': 'BlaC',
}


def summarise_query(df):
    df = df.dropna(subset=['query_prot_idx'])
    return [{
        'lit_pos':     int(r.lit_pos),
        'ref_aa':      r.ref_aa,
        'query_aa':    r.query_aa,
        'alt_aas':     r.alt_aas,
        'is_gof_risk': bool(r.is_gof_risk),
        'prot_idx':    int(r.query_prot_idx),
        'nuc_start':   int(r.query_nuc_start) if pd.notna(r.query_nuc_start) else None,
        'nuc_end':     int(r.query_nuc_end)   if pd.notna(r.query_nuc_end) else None,
    } for r in df.itertuples()]


PDB_FASTA_DIR.mkdir(exist_ok=True)

pdb_fasta_paths = {}
for family, (seq_str, res_numbers) in pdb_linear_seqs.items():
    pdb_id = PDB_REFS[family][0]
    out_path = PDB_FASTA_DIR / f'{family}_{pdb_id}.fasta'
    record = SeqRecord(Seq(seq_str), id=f'{pdb_id}_{family}_ref',
                       description=f'PDB {pdb_id} chain {PDB_REFS[family][1]} linear sequence')
    SeqIO.write([record], out_path, 'fasta')
    pdb_fasta_paths[family] = out_path
    print(f'Written {out_path}  ({len(seq_str)} aa)')




def parse_alignment(aln_path):
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(aln_path, 'fasta')}


def ref_idx_to_aln_col(ref_aln_seq, ref_idx):
    ungapped = 0
    for col, aa in enumerate(ref_aln_seq):
        if aa != '-':
            if ungapped == ref_idx:
                return col
            ungapped += 1
    return None


def aln_col_to_query_protein_idx(query_aln_seq, col):
    if col is None or col >= len(query_aln_seq):
        return None
    if query_aln_seq[col] == '-':
        return None  #query has gap at this position means site absent
    #count non gap chars up to and including col
    return sum(1 for c in query_aln_seq[:col+1] if c != '-') - 1



STOP_CODONS = {'TAA', 'TAG', 'TGA'}
IUPAC = {
    'A':'A','C':'C','G':'G','T':'T','N':'[ACGT]',
    'R':'[AG]','Y':'[CT]','S':'[GC]','W':'[AT]',
    'K':'[GT]','M':'[AC]','B':'[CGT]','D':'[AGT]',
    'H':'[ACT]','V':'[ACG]',
}

OUT_DIR       = Path('crispr_results')
ENTROPY_CSV   = Path('beta_lactamase_msa_results/gene_family/entropy_gene_family_ALL.csv')
OUT_DIR.mkdir(exist_ok=True)


def parse_alts(gof_entry):
    """Return list of alt amino acids from a GOF dict entry."""
    val = gof_entry.get('alt_aa')
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]   #single string -> wrap in list



#PAM regex
def pam_to_regex(pam):
    return re.compile(''.join(IUPAC[c] for c in pam.upper()))



#context-adjusted window
def eff_window(editor_name, preceding_base):
    ed = BASE_EDITORS[editor_name]
    w  = ed['activity_window']
    if preceding_base and 'context_window_modifiers' in ed:
        return ed['context_window_modifiers'].get(preceding_base.upper(), w)
    return w


def efficiency_score(editor_name, preceding_base):
    ed     = BASE_EDITORS[editor_name]
    pref   = ed.get('context_preference', {})
    rating = pref.get(preceding_base.upper() if preceding_base else 'N', 'good')
    return {'best': 1.0, 'good': 0.75, 'moderate': 0.5,
            'reduced': 0.4, 'poor': 0.25}.get(rating, 0.5)




def scan_pam(nt_seq, editor_name):
    """
    get dict for PAMs
    proto_start_cds : 0-indexed start of protospacer in forward CDS
    win_s, win_e    : activity window in forward CDS (0-indexed), base window
    safe_s, safe_e  : win ± buffer, clamped  (context-independent, base window)
    protospacer     : sequence in 5'->3' protospacer orientation
    context-dependent safe window is computed per edit-position in the main loop
    using proto_start_cds + context-adjusted window bounds.
    """
    ed = BASE_EDITORS[editor_name]
    plen = ed['protospacer_len']
    pam_re = pam_to_regex(ed['PAM'])
    w_s, w_e = ed['activity_window']    #1-indexed from 5' of protospacer
    buf = ed.get('buffer_bp', 2)
    seq = nt_seq.upper()
    rc = str(Seq(seq).reverse_complement())
    L = len(seq)
    for strand, s in [('+', seq), ('-', rc)]:
        for m in pam_re.finditer(s):
            proto_s_in_strand = m.start() - plen
            if proto_s_in_strand < 0:
                continue
            proto_seq_strand = s[proto_s_in_strand: proto_s_in_strand + plen]
            if strand == '+':
                #protospacer start in CDS = same as in strand string
                proto_start_cds = proto_s_in_strand
                protospacer     = proto_seq_strand
            else:
                #protospacer start in CDS (forward, 0-indexed)
                #position 0 of protospacer in rc = proto_s_in_strand in rc
                #maps to CDS position L - proto_s_in_strand - plen
                proto_start_cds = L - proto_s_in_strand - plen
                protospacer     = str(Seq(proto_seq_strand).reverse_complement())
            #base activity window in CDS coords (0-indexed)
            ws_cds = proto_start_cds + (w_s - 1)
            we_cds = proto_start_cds + (w_e - 1)
            if we_cds < 0 or ws_cds >= L:
                continue
            #clamp to CDS
            ws_cds = max(0, ws_cds)
            we_cds = min(L - 1, we_cds)
            #base safe window
            safe_s = max(0,   ws_cds - buf)
            safe_e = min(L-1, we_cds + buf)
            yield {
                'strand':           strand,
                'protospacer':      protospacer,
                'proto_start_cds':  proto_start_cds,
                'win_s':            ws_cds,
                'win_e':            we_cds,
                'safe_s':           safe_s,
                'safe_e':           safe_e,
            }



#STOP CODON CHECK
def stop_hits_in_window(nt_seq, win_s, win_e, edit_type):
    """Return list of (nt_pos, aa_pos_0indexed, wt_codon, mut_codon)."""
    target = 'C' if edit_type == 'CBE' else 'A'
    sub    = 'T' if edit_type == 'CBE' else 'G'
    hits   = []
    for pos in range(max(0, win_s), min(win_e + 1, len(nt_seq))):
        if nt_seq[pos] != target:
            continue
        aa  = pos // 3
        pic = pos % 3
        cod = nt_seq[aa*3: aa*3+3]
        if len(cod) < 3:
            continue
        mut = cod[:pic] + sub + cod[pic+1:]
        if mut in STOP_CODONS:
            hits.append((pos, aa, cod, mut))
    return hits


#GOF OVERLAP  (uses only projected CDS windows)

def overlaps_gof_window(safe_s, safe_e, gof_windows):
    """Return list of (gs,ge) GOF codon windows that overlap [safe_s, safe_e]."""
    return [(gs, ge) for gs, ge in gof_windows if safe_s <= ge and safe_e >= gs]



#ENTROPY

entropy_df = pd.read_csv(ENTROPY_CSV) if ENTROPY_CSV.exists() else None

def get_entropy(gene_family):
    """Look up per-column entropy values for a gene family from the family-level MSA."""
    if entropy_df is None:
        return None
    row = entropy_df[entropy_df['alignment_name'] == gene_family]
    if row.empty:
        return None
    try:
        return ast.literal_eval(row.iloc[0]['entropy_values'])
    except Exception:
        return None

def pct_high_conservation_downstream(entropies, aa_pos_0, threshold=0.5):
    if entropies is None:
        return None
    downstream = entropies[aa_pos_0:]
    if not downstream:
        return 0.0
    return round(sum(1 for h in downstream if h <= threshold) / len(downstream), 3)



prot_lookup = {}  #query_id -> protein_seq
for rec in SeqIO.parse(str(PROT_FA), 'fasta'):
    seq = str(rec.seq)
    if '*' in seq:
        seq = seq[:seq.index('*')]
    prot_lookup[rec.id] = seq

nuc_lookup = {}   #query_id -> nucleotide_seq
for rec in SeqIO.parse(str(NUC_FA), 'fasta'):
    nuc_lookup[rec.id] = str(rec.seq).upper()


#grouped by family, write per-family FASTA with ref 
family_query_records = {}
missing_seqs = []
for _, row in test.iterrows():
    qid = row['query_id']
    fam_raw = row.get('gene_family', '')
    family = FAMILY_NORM.get(str(fam_raw).strip(), None)
    if family not in PDB_REFS:
        continue
    prot = prot_lookup.get(qid)
    if prot is None:
        missing_seqs.append(qid)
        continue
    rec = SeqRecord(Seq(prot), id=qid, description='')
    family_query_records.setdefault(family, []).append(rec)



print(f'\nsequences: {sum(len(v2) for v in family_query_records.values() for v2 in v)}')
print(f'missed {len(missing_seqs)}')
if missing_seqs[:5]:
    print(f'inc. {missing_seqs[:5]}')



#write combined fastas (ref + queries) per family
combined_fasta_paths = {}   #family -> path
for family, query_recs in family_query_records.items():
    ref_path = pdb_fasta_paths[family]
    ref_rec  = list(SeqIO.parse(ref_path, 'fasta'))[0]
    out_path = OUT_FASTA_DIR / f'{family}.fasta'
    all_recs = [ref_rec] + query_recs
    SeqIO.write(all_recs, out_path, 'fasta')
    combined_fasta_paths[family] = out_path
    print(f'  Written {out_path}: 1 ref + {len(query_recs)} queries')


#MSA queries in gene fam against PDB reference
#Use pre-computed gene_family alignment if available, otherwise run mafft



#per-query summary with all GOF sites with their nuc coordinates etc
#merge in gene name from test

ALN_DIR = Path('gof_mapping_results/alignments')
ALN_DIR.mkdir(exist_ok=True)
alignment_paths = {}

for family, fasta_path in combined_fasta_paths.items():
    out_aln = ALN_DIR / f'{family}.aln.fasta'
    if out_aln.exists():
        alignment_paths[family] = out_aln
        print(f'Using cached: {out_aln}')
        continue
    n_seqs = sum(1 for _ in SeqIO.parse(fasta_path, 'fasta'))
    if n_seqs <= 200:
        mafft_args = ['mafft', '--localpair', '--maxiterate', '1000',
                      '--thread', str(THREADS), '--quiet', str(fasta_path)]
    elif n_seqs <= 2000:
        mafft_args = ['mafft', '--genafpair', '--maxiterate', '16',
                      '--thread', str(THREADS), '--quiet', str(fasta_path)]
    else:
        mafft_args = ['mafft', '--auto',
                      '--thread', str(THREADS), '--quiet', str(fasta_path)]
    print(f'Aligning {family} ({n_seqs} seqs)...')
    try:
        with open(out_aln, 'w') as out_fh:
            result = subprocess.run(mafft_args, stdout=out_fh,
                                    stderr=subprocess.PIPE, timeout=3600)
        if result.returncode != 0:
            print(f'  MAFFT error: {result.stderr.decode()[:200]}')
            out_aln.unlink(missing_ok=True)
            continue
        alignment_paths[family] = out_aln
        print(f'  -> {out_aln}')
    except subprocess.TimeoutExpired:
        print(f'  TIMEOUT for {family}')
        out_aln.unlink(missing_ok=True)



def iter_fasta_streaming(path):
    rec_id, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith('>'):
                if rec_id is not None:
                    yield rec_id, ''.join(chunks)
                rec_id, chunks = line[1:].split()[0], []
            else:
                chunks.append(line)
    if rec_id is not None:
        yield rec_id, ''.join(chunks)



rows = []
for family, aln_path in alignment_paths.items():
    pdb_id = PDB_REFS[family][0]
    ref_id = f'{pdb_id}_{family}_ref'
    gof_sites = GOF_VERIFIED.get(family, {})
    if not gof_sites:
        continue
    #find ref sequence only
    ref_aln_seq = None
    for seq_id, seq in iter_fasta_streaming(aln_path):
        if seq_id == ref_id:
            ref_aln_seq = seq
            break
    if ref_aln_seq is None:
        print(f'WARNING: ref {ref_id} not in alignment {family}')
        continue
    #precomp alignment columns for GOF sites from ref
    site_cols = {
        lit_pos: ref_idx_to_aln_col(ref_aln_seq, info['idx'])
        for lit_pos, info in gof_sites.items()
    }
    for seq_id, query_aln_seq in iter_fasta_streaming(aln_path):
        if seq_id == ref_id:
            continue
        for lit_pos, info in gof_sites.items():
            ref_aa = info['ref_aa']
            alts = parse_alts(info)
            #alts = info.get('alts', [])
            #if isinstance(alts, str):
            #   alts = [alts] if alts else []
            pdb_aa = info['pdb_aa']
            col = site_cols[lit_pos]
            prot_idx = aln_col_to_query_protein_idx(query_aln_seq, col)
            if prot_idx is not None:
                nuc_start = prot_idx * 3
                nuc_end = nuc_start + 2
                nuc_idxs = (nuc_start, nuc_end)
            else:
                nuc_idxs = None
            query_aa = query_aln_seq[col] if col is not None else None
            if query_aa == '-':
                query_aa = None
            is_gof_risk = bool(query_aa and alts and query_aa in alts)
            rows.append({
                'query_id':        seq_id,
                'family':          family,
                'lit_pos':         lit_pos,
                'ref_aa':          ref_aa,
                'pdb_aa':          pdb_aa,
                'query_aa':        query_aa,
                'alt_aas':         ','.join(alts),
                'is_gof_risk':     is_gof_risk,
                'aln_col':         col,
                'query_prot_idx':  prot_idx,
                'query_nuc_start': nuc_idxs[0] if nuc_idxs else None,
                'query_nuc_end':   nuc_idxs[1] if nuc_idxs else None,
            })



gof_results = pd.DataFrame(rows)



pid_to_gene = test.set_index('query_id')['gene_name'].to_dict()
pid_to_family = test.set_index('query_id')['gene_family'].to_dict()


summary_rows = []
for query_id, grp in gof_results.groupby('query_id'):
    sites = summarise_query(grp)
    nuc_avoid = sorted({(s['nuc_start'], s['nuc_end']) for s in sites if s['nuc_start'] is not None})
    prot_avoid = sorted({s['prot_idx'] for s in sites})
    gof_risk = [s for s in sites if s['is_gof_risk']]
    summary_rows.append({
        'query_id':         query_id,
        'gene_name':        pid_to_gene.get(query_id, ''),
        'gene_family':      pid_to_family.get(query_id, ''),
        'family_canonical': grp['family'].iloc[0],
        'n_gof_sites_mapped': len(sites),
        'n_gof_risk':       len(gof_risk),
        'gof_risk_sites':   gof_risk,           #already carry a GOF aa
        'all_gof_sites':    sites,              #all mapped sites regardless
        'prot_idxs_avoid':  prot_avoid,         #0-indexed protein positions
        'nuc_ranges_avoid': nuc_avoid,          #(nuc_start, nuc_end) pairs
    })


gof_summary = pd.DataFrame(summary_rows)
print(f'Query-position pairs evaluated:  {len(gof_results)}')
print(f'Queries mapped:                  {len(gof_summary)}')
print(f'Queries already carrying GOF aa: {(gof_summary['n_gof_risk'] > 0).sum()}')
print(f'Sites absent from query (gap):   {gof_results['query_prot_idx'].isna().sum()}')


gof_results.to_csv('gof_mapping_results/gof_positions_per_pid.csv', index=False)
gof_summary.to_csv('gof_mapping_results/gof_risk_summary.csv', index=False)

#flat nuc-ranges file w one row per (query, site) for downstream lookup
nuc_rows = []
for _, r in gof_results.iterrows():
    if r['query_nuc_start'] is None:
        continue
    nuc_rows.append({
        'query_id':    r['query_id'],
        'gene_family': r['family'],
        'lit_pos':     r['lit_pos'],
        'ref_aa':      r['ref_aa'],
        'query_aa':    r['query_aa'],
        'is_gof_risk': r['is_gof_risk'],
        'prot_idx':    r['query_prot_idx'],
        'nuc_start':   r['query_nuc_start'],   #0-based in CDS
        'nuc_end':     r['query_nuc_end'],     #inclusive
    })



#GOF AVOIDANCE WINDOWS from nuc_rows

gof_nuc_df = pd.DataFrame(nuc_rows)   
pid_gof_windows = defaultdict(set)
for r in gof_nuc_df.itertuples():
    if pd.notna(r.nuc_start) and pd.notna(r.nuc_end):
        pid_gof_windows[r.query_id].add((int(r.nuc_start), int(r.nuc_end)))


#GENE SELECTION

fams = list(set(test['gene_family'].tolist()))
all_testable_genes = []
for fam in fams:
    all_testable_genes += list(set(test.loc[test['gene_family'] == fam, 'gene_name'].tolist()))

gene_names = all_testable_genes
gene_rows = test[test['gene_name'].isin(gene_names)].copy()
print(f'{len(gene_rows)} rows matched, {gene_rows['query_id'].nunique()} unique PIDs')



#BYSTANDER WARNING flags additional editable bases in the same codon or activity window.
#currently doesn't do cominational for all possible within window as per bystanders, same_codon warnings are the most safety-relevant because
#they might alter the predicted stop codon.

def bystander_warning(nt_seq, nt_pos, win_s, win_e, edit_type):
    target = 'C' if edit_type == 'CBE' else 'A'
    aa = nt_pos // 3
    codon_s = aa * 3
    codon_e = codon_s + 2
    warnings = []
    for pos in range(codon_s, codon_e + 1):
        if pos == nt_pos or pos < win_s or pos > win_e:
            continue
        if pos < len(nt_seq) and nt_seq[pos] == target:
            warnings.append(f'same_codon@{pos}')
    for pos in range(win_s, min(win_e + 1, len(nt_seq))):
        if pos == nt_pos or (codon_s <= pos <= codon_e):
            continue
        if nt_seq[pos] == target:
            warnings.append(f'adjacent_codon@{pos}(aa{pos // 3 + 1})')
    return '; '.join(warnings)



#MAIN SCAN

all_candidates = []
nt_seq_lookup  = {}

for _, row in gene_rows.iterrows():
    qid    = row['query_id']
    gene   = row['gene_name']
    family = row['gene_family']
    nt = nuc_lookup.get(qid)
    if nt is None:
        continue
    #trim to codon boundary
    nt = nt[:len(nt) - len(nt) % 3]
    nt_seq_lookup[qid] = nt
    gof_wins = pid_gof_windows.get(qid, set())
    gof_was_mapped = qid in pid_gof_windows
    entropies = get_entropy(family)   #family-level entropy from gene_family MSA
    cds_len_aa = len(nt) // 3
    for ed_name, ed in BASE_EDITORS.items():
        edit_type = ed['type']
        for hit in scan_pam(nt, ed_name):
            proto_start = hit['proto_start_cds']
            #use base window to find candidate positions
            stop_hits = stop_hits_in_window(nt, hit['win_s'], hit['win_e'], edit_type)
            if not stop_hits:
                continue
            for nt_pos, aa_pos_0, wt_codon, mut_codon in stop_hits:
                preceding = nt[nt_pos - 1] if nt_pos > 0 else 'N'
                #protospacer-relative position
                #pos_in_proto is 1-indexed from 5' end of protospacer
                pos_in_proto = (nt_pos - proto_start) + 1
                if pos_in_proto < 1:
                    #upstream of protospacer start
                    continue   
                #context-adjusted window (1-indexed protospacer positions)
                ctx_w_s, ctx_w_e = eff_window(ed_name, preceding)
                if not (ctx_w_s <= pos_in_proto <= ctx_w_e):
                    continue
                buf        = ed.get('buffer_bp', 2)
                ctx_ws_cds = proto_start + (ctx_w_s - 1)
                ctx_we_cds = proto_start + (ctx_w_e - 1)
                ctx_safe_s = max(0,         ctx_ws_cds - buf)
                ctx_safe_e = min(len(nt)-1, ctx_we_cds + buf)
                gof_overlap    = overlaps_gof_window(ctx_safe_s, ctx_safe_e, gof_wins)
                family_has_gof = bool(GOF_MUTATIONS.get(FAMILY_ALIASES.get(family.lower(), family), []))
                is_safe        = (not gof_overlap) and (gof_was_mapped or not family_has_gof)
                query_aa_at_edit = wt_codon
                eff         = efficiency_score(ed_name, preceding)
                pct_early   = round(aa_pos_0 / max(cds_len_aa, 1), 4)
                pct_cons_dn = pct_high_conservation_downstream(entropies, aa_pos_0)
                all_candidates.append({
                    'query_id':          qid,
                    'gene_name':         gene,
                    'family':            family,
                    'editor':            ed_name,
                    'editor_type':       edit_type,
                    'strand':            hit['strand'],
                    'protospacer':       hit['protospacer'],
                    'proto_start_cds':   proto_start,
                    'win_s':             ctx_ws_cds,         #context-adjusted
                    'win_e':             ctx_we_cds,
                    'safe_s':            ctx_safe_s,        
                    'safe_e':            ctx_safe_e,
                    'edit_nt_pos':       nt_pos,
                    'edit_aa_pos_1':     aa_pos_0 + 1,
                    'pos_in_proto':      pos_in_proto,       
                    'wt_codon':          wt_codon,
                    'stop_codon':        mut_codon,
                    'preceding_base':    preceding,
                    'efficiency_score':  eff,
                    'pct_early':         pct_early,
                    'pct_conserved_dn':  pct_cons_dn,
                    'gof_nuc_overlap':   str(gof_overlap) if gof_overlap else '',
                    'is_safe':           is_safe,
                    'gof_mapped':        gof_was_mapped, #i.e. was GOF projection available for this PID, get those without using candidates[~candidates['gof_mapped']]
                    'bystander_risk':    bystander_warning(
                                             nt, nt_pos,
                                             ctx_ws_cds, ctx_we_cds,
                                             edit_type),
                })


candidates = pd.DataFrame(all_candidates)
safe        = candidates[candidates['is_safe']].copy()

#strict safe: additionally exclude guides with same-codon bystander risk
safe_strict = safe[~safe['bystander_risk'].str.contains('same_codon', na=False)].copy()
print(f'Strict safe (no same-codon bystander): {len(safe_strict):,}')

print(f'Total stop-creating candidates: {len(candidates):,}')
print(f'Safe (no GOF overlap):          {len(safe):,}')
print(f'Unique PIDs with safe guides:   {safe['query_id'].nunique()}')



#RANK

safe['rank_score'] = (
    safe['efficiency_score'] * (1 - safe['pct_early']) *
    safe['pct_conserved_dn'].fillna(0.5)
)
safe = safe.sort_values('rank_score', ascending=False)



#GREEDY MINIMAL sgRNA SET

guide_cov = (safe.groupby(['editor','protospacer','strand'])
                  .agg(covered=('query_id', lambda x: frozenset(x)),
                       n_pids=('query_id','nunique'),
                       mean_eff=('efficiency_score','mean'),
                       mean_early=('pct_early','mean'),
                       mean_cons=('pct_conserved_dn','mean'),
                       family=('family','first'),
                       editor_type=('editor_type','first'))
                  .reset_index()
                  .sort_values(['n_pids','mean_eff','mean_early'],
                                ascending=[False, False, True]))

uncovered = set(safe['query_id'].unique())
sgrna_set = []
while uncovered:
    best_new, best_row = frozenset(), None
    for _, r in guide_cov.iterrows():
        new = r['covered'] & uncovered
        if len(new) > len(best_new):
            best_new, best_row = new, r
    if best_row is None:
        break
    sgrna_set.append({
        'editor':            best_row['editor'],
        'editor_type':       best_row['editor_type'],
        'protospacer':       best_row['protospacer'],
        'strand':            best_row['strand'],
        'family':            best_row['family'],
        'n_pids_covered':    len(best_new),
        'mean_efficiency':   round(best_row['mean_eff'], 3),
        'mean_pct_early':    round(best_row['mean_early'], 3),
        'mean_pct_cons_dn':  round(best_row['mean_cons'], 3) if pd.notna(best_row['mean_cons']) else None,
        'covers':            sorted(best_new),
    })
    uncovered -= best_new

sgrna_set = pd.DataFrame(sgrna_set)
print(f'\nMinimal sgRNA set: {len(sgrna_set)} guide(s)')
if not sgrna_set.empty:
    print(sgrna_set[['editor','protospacer','strand','n_pids_covered',
                      'mean_efficiency','mean_pct_early','mean_pct_cons_dn']].to_string(index=False))



candidates.to_csv(  OUT_DIR / 'candidates_all.csv',    index=False)
safe.to_csv(         OUT_DIR / 'candidates_safe.csv',   index=False)
safe_strict.to_csv(  OUT_DIR / 'candidates_safe_strict.csv', index=False)
sgrna_set.to_csv(  OUT_DIR / 'sgrna_minimal_set.csv', index=False)
print(f'\nSaved to {OUT_DIR}/')



#PER-PID SITE EXPANSION (applied sgRNAs only)

applied_protos = set(zip(sgrna_set['editor'], sgrna_set['protospacer'], sgrna_set['strand']))

applied_candidates = candidates[
    candidates.apply(
        lambda r: (r['editor'], r['protospacer'], r['strand']) in applied_protos, axis=1)
].copy()

pid_site_rows = []
for _, r in applied_candidates.iterrows():
    for nt_pos in range(int(r['safe_s']), int(r['safe_e']) + 1):
        aa_pos_0 = nt_pos // 3
        pid_site_rows.append({
            'query_id':           r['query_id'],
            'gene_name':          r['gene_name'],
            'family':             r['family'],
            'editor':             r['editor'],
            'editor_type':        r['editor_type'],
            'protospacer':        r['protospacer'],
            'strand':             r['strand'],
            'nt_pos_in_cds':      nt_pos,
            'aa_pos_1':           aa_pos_0 + 1,
            'in_activity_window': (int(r['win_s']) <= nt_pos <= int(r['win_e'])),
            'is_safe':            r['is_safe'],
        })

pid_sites = pd.DataFrame(pid_site_rows).drop_duplicates(
    subset=['query_id','editor','protospacer','nt_pos_in_cds'])
pid_sites.to_csv(OUT_DIR / 'per_pid_safe_window_sites.csv', index=False)
print(f'Per-PID safe window sites: {len(pid_sites):,} rows, '
      f'{pid_sites['query_id'].nunique()} PIDs')



all_input_pids   = set(gene_rows['query_id'].unique())
pids_with_any    = set(candidates['query_id'].unique())           #had >=1 stop-creating guide
pids_with_safe   = set(safe['query_id'].unique())                 #had >=1 safe guide
pids_in_min_set  = set(pid for _, r in sgrna_set.iterrows()
                        for pid in r['covers']) if not sgrna_set.empty else set()


#build a per-PID metadata lookup from gene_rows
pid_meta = (gene_rows[['query_id','gene_name','gene_family']]
            .drop_duplicates('query_id')
            .set_index('query_id'))

#add CDS length (from nuc_lookup, already trimmed to codon boundary)
pid_meta['cds_nt_len']  = pid_meta.index.map(
    lambda q: len(nt_seq_lookup[q]) if q in nt_seq_lookup else None)
pid_meta['cds_aa_len']  = pid_meta['cds_nt_len'].apply(
    lambda x: x // 3 if pd.notna(x) else None)

#add GOF mapping status
pid_meta['gof_mapped']  = pid_meta.index.map(lambda q: q in pid_gof_windows)

#classify reason for not being in minimal set
def _untarget_reason(qid):
    if qid not in pids_with_any:
        return 'no_stop_creating_guide'      #no C/A in any PAM activity window creates a stop
    if qid not in pids_with_safe:
        return 'all_guides_gof_unsafe'       #guides exist but all overlap a GOF window
    return 'not_covered_by_minimal_set'      #safe guides exist but greedy didn't pick one


untargetable_pids = all_input_pids - pids_in_min_set
untargetable_rows = []
for qid in sorted(untargetable_pids):
    meta = pid_meta.loc[qid] if qid in pid_meta.index else {}
    untargetable_rows.append({
        'query_id':     qid,
        'gene_name':    meta.get('gene_name',   '') if isinstance(meta, pd.Series) else '',
        'gene_family':  meta.get('gene_family', '') if isinstance(meta, pd.Series) else '',
        'cds_nt_len':   meta.get('cds_nt_len',  '') if isinstance(meta, pd.Series) else '',
        'cds_aa_len':   meta.get('cds_aa_len',  '') if isinstance(meta, pd.Series) else '',
        'gof_mapped':   meta.get('gof_mapped',  '') if isinstance(meta, pd.Series) else '',
        'reason':       _untarget_reason(qid),
    })

untargetable_df = pd.DataFrame(untargetable_rows)
untargetable_df.to_csv(OUT_DIR / 'untargetable_pids.csv', index=False)



n_total   = len(all_input_pids)
n_covered = len(pids_in_min_set)
n_unt     = len(untargetable_pids)

print('\n' + '='*60)
print('TARGETING COVERAGE SUMMARY')
print('='*60)
print(f'Total input PIDs:                    {n_total:>7,}')
print(f'PIDs with any stop-creating guide:   {len(pids_with_any):>7,}  '
      f'({len(pids_with_any)/n_total*100:.1f}%)')
print(f'PIDs with any safe guide:            {len(pids_with_safe):>7,}  '
      f'({len(pids_with_safe)/n_total*100:.1f}%)')
print(f'PIDs covered by minimal sgRNA set:   {n_covered:>7,}  '
      f'({n_covered/n_total*100:.1f}%)')
print(f'PIDs NOT covered (untargetable):     {n_unt:>7,}  '
      f'({n_unt/n_total*100:.1f}%)')
print()

if not untargetable_df.empty:
    reason_counts = untargetable_df['reason'].value_counts()
    print('Breakdown of untargetable reasons:')
    for reason, count in reason_counts.items():
        print(f'  {reason:<40s} {count:>6,}  ({count/n_unt*100:.1f}%)')
    print()
    print('Untargetable PIDs by family:')
    fam_counts = untargetable_df['gene_family'].value_counts().head(20)
    for fam, count in fam_counts.items():
        print(f'  {str(fam):<30s} {count:>6,}')
    if len(fam_counts) < untargetable_df['gene_family'].nunique():
        print(f'  ... (showing top 20 of {untargetable_df['gene_family'].nunique()} families)')

print('='*60)
print(f'Full untargetable list → {OUT_DIR}/untargetable_pids.csv')



##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################



import re
import math
import ast
from collections import defaultdict
from pathlib import Path
import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
import os
import csv
import subprocess
import tempfile
from pathlib import Path
from Bio import AlignIO
from Bio.SeqRecord import SeqRecord
import requests, time
from io import StringIO
from Bio.PDB import PDBParser
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.SeqUtils import seq1
from Bio.PDB import PDBParser
import numpy as np
import matplotlib.pyplot as plt



OUT_DIR = Path('crispr_results')

candidates = pd.read_csv(OUT_DIR / 'candidates_all.csv')
safe_strict = pd.read_csv(OUT_DIR / 'candidates_safe_strict.csv')
safe = pd.read_csv(OUT_DIR / 'candidates_safe.csv')
sgrna_set = pd.read_csv(OUT_DIR / 'sgrna_minimal_set.csv')




#PLOTS THE DISTRIBUTION OF HOW MANY SEQUENCES/PIDS THE SGRNAS HIT - THE POINT IS THAT A FEW COVER A LOT, WHICH IS WHY ITS LESS THAN I EXPECTED.


vals = sgrna_set['n_pids_covered'].values

fig, ax = plt.subplots(figsize=(8,5))

#log bins for heavy-tailed distribution
bins = np.logspace(
    np.log10(vals.min()),
    np.log10(vals.max()),
    40
)

ax.hist(
    vals,
    bins=bins,
    color='darkred',
    edgecolor='black',
    linewidth=0.2
)

ax.set_xscale('log')

ax.set_xlabel('Sequences targeted per sgRNA')
ax.set_ylabel('Number of sgRNAs')
ax.set_title('Distribution of sgRNA target hit coverage')

median = np.median(vals)
mean = np.mean(vals)

ax.axvline(median, linestyle='--', linewidth=1, color='darkorange', label=f'median = {median:.0f}')
ax.axvline(mean, linestyle=':', linewidth=1, color='darkorange', label=f'mean = {mean:.0f}')

ax.legend()

ax.spines[['top','right']].set_visible(False)

plt.tight_layout()

plt.savefig(
    'sgRNA_PID_coverage_hist.png',
    dpi=300
)

plt.close()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.ticker import LogLocator


N_PERMS      = 10        #independent gene permutations for mean +/- ribbon
N_STEPS      = 100       #log-spaced gene-count checkpoints
WORKERS      = 4         #parallel processes; set to 1 if memory is tight


#gene_name → list of query_ids that appear in `safe`
gene_to_pids: dict[str, frozenset] = {
    g: frozenset(grp['query_id'].unique())
    for g, grp in safe.groupby('gene_name')
}

#guide key → frozenset of PIDs it covers (among safe candidates)
guide_to_pids: dict[tuple, frozenset] = {
    (ed, proto, strand): frozenset(grp['query_id'].unique())
    for (ed, proto, strand), grp in
    safe.groupby(['editor', 'protospacer', 'strand'])
}

#pid → set of guide keys that cover it (inverted index for incremental updates)
pid_to_guides: dict[str, list[tuple]] = defaultdict(list)
for gkey, pids in guide_to_pids.items():
    for pid in pids:
        pid_to_guides[pid].append(gkey)

all_gene_names = sorted(gene_to_pids.keys())
n_genes_total  = len(all_gene_names)

#log-spaced checkpoints at which we record sgRNA count
checkpoints = sorted(set(
    np.unique(np.round(np.geomspace(1, n_genes_total, N_STEPS)).astype(int)).tolist()
))


def _greedy_incremental_sim(seed: int) -> list[tuple[int, int, int]]:
    """
    One permutation of all genes, incrementally adding one gene at a time.
    Returns list of (n_genes_added, n_sgrnas, n_pids_uncovered) at each checkpoint.
    Incremental greedy:
      - Maintain 'uncovered': set of PIDs not yet covered by chosen guides.
      - Maintain 'chosen_guides': set of guide keys in current minimal set.
      - When a new gene is added, its PIDs enter 'uncovered'. We then greedily
        pick guides only from those that cover at least one uncovered PID,
        preferring the guide that covers the most new PIDs at each step.
      - 'chosen_guides' only grows — guides are never removed — so the set size
        is a non-decreasing upper bound on the true incremental minimum set.
        (This is equivalent to greedy set cover re-seeded at each checkpoint.)
    """
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(all_gene_names).tolist()
    uncovered:     set[str]   = set()
    chosen_guides: set[tuple] = set()
    results = []
    for step, gene in enumerate(perm, 1):
        #add PIDs for this gene that are not yet covered
        new_pids = gene_to_pids[gene] - (gene_to_pids[gene] & uncovered)
        #more precisely: newly added, and not already covered by chosen guides
        truly_new = set()
        for pid in gene_to_pids[gene]:
            covered_by_chosen = any(
                pid in guide_to_pids[gk] for gk in chosen_guides
            )
            if not covered_by_chosen:
                truly_new.add(pid)
        uncovered |= truly_new
        #greedy extend: only examine guides touching uncovered pids
        candidate_guides = set()
        for pid in truly_new:
            candidate_guides.update(pid_to_guides[pid])
        candidate_guides -= chosen_guides
        #greedily pick until nothing more can be covered
        local_uncovered = set(uncovered)  #copy; we drain this locally
        while local_uncovered and candidate_guides:
            best_gk   = max(candidate_guides,
                            key=lambda gk: len(guide_to_pids[gk] & local_uncovered))
            newly_cov = guide_to_pids[best_gk] & local_uncovered
            if not newly_cov:
                break
            chosen_guides.add(best_gk)
            local_uncovered -= newly_cov
            candidate_guides.discard(best_gk)
            #also remove guides that now cover nothing new
            candidate_guides = {gk for gk in candidate_guides
                                if guide_to_pids[gk] & local_uncovered}
        #true uncovered = global uncovered minus what chosen guides cover
        all_covered = set()
        for gk in chosen_guides:
            all_covered |= guide_to_pids[gk]
        real_uncovered = uncovered - all_covered
        if step in checkpoints:
            results.append((step, len(chosen_guides), len(real_uncovered)))
    return results



#permutations
print(f'\nrunning {N_PERMS} gene permutation simulations '
      f'({n_genes_total} genes, {len(checkpoints)} checkpoints)')



all_runs: list[pd.DataFrame] = []

if WORKERS > 1:
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_greedy_incremental_sim, seed): seed
                for seed in range(N_PERMS)}
        for fut in as_completed(futs):
            seed = futs[fut]
            rows = fut.result()
            df_r = pd.DataFrame(rows, columns=['n_genes', 'n_sgrnas', 'n_uncovered'])
            df_r['seed'] = seed
            all_runs.append(df_r)
            print(f'  permutation {seed} done')
else:
    for seed in range(N_PERMS):
        rows = _greedy_incremental_sim(seed)
        df_r = pd.DataFrame(rows, columns=['n_genes', 'n_sgrnas', 'n_uncovered'])
        df_r['seed'] = seed
        all_runs.append(df_r)
        print(f'  permutation {seed} done')



scaling_df = pd.concat(all_runs, ignore_index=True)
scaling_df.to_csv(OUT_DIR / 'sgrna_scaling_by_gene.csv', index=False)
print(f'Scaling data → {OUT_DIR}/sgrna_scaling_by_gene.csv')

#aggregate mean +/- sd across permutations
agg = (scaling_df.groupby('n_genes')[['n_sgrnas', 'n_uncovered']]
                  .agg(['mean', 'std']).reset_index())
agg.columns = ['n_genes',
               'sgrnas_mean', 'sgrnas_std',
               'uncov_mean',  'uncov_std']
agg['sgrnas_std'] = agg['sgrnas_std'].fillna(0)
agg['uncov_std']  = agg['uncov_std'].fillna(0)




fig, ax1 = plt.subplots(figsize=(9, 6))
ax2 = ax1.twinx()
x = agg['n_genes'].to_numpy()
ax1.plot(x, agg['sgrnas_mean'], color='black', linewidth=1,
         label='sgRNAs required')
ax1.fill_between(x,
                 agg['sgrnas_mean'] - agg['sgrnas_std'],
                 agg['sgrnas_mean'] + agg['sgrnas_std'],
                 alpha=0.5, color='grey', linewidth=1)
ax1.set_xlabel('Number of unique genes to target', fontsize=11)
ax1.set_ylabel('(greedy) minimal number of sgRNAs', color='black', fontsize=11)
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xscale('log')
ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())



ax2.plot(x, agg['uncov_mean'], color='darkred', linewidth=1,
         linestyle='--', label='Sequences that could not be targeted')
ax2.fill_between(x,
                 agg['uncov_mean'] - agg['uncov_std'],
                 agg['uncov_mean'] + agg['uncov_std'],
                 alpha=0.15, color='grey')
ax2.set_ylabel('Untargeted sequences', color='darkred', fontsize=11)
ax2.tick_params(axis='y', labelcolor='darkred')



lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

ax1.set_title(f' Number of sgRNAs needed for a given gene set size\n'
              f'{n_genes_total} total genes)',
              fontsize=11)
ax1.spines['top'].set_visible(False)



plt.tight_layout()
plt.show()




fig, ax1 = plt.subplots(figsize=(9, 6))
ax2 = ax1.twinx()
x = agg['n_genes'].to_numpy()
ax1.plot(x, agg['sgrnas_mean'], color='black', linewidth=1,
         label='sgRNAs required')
ax1.fill_between(x,
                 agg['sgrnas_mean'] - agg['sgrnas_std'],
                 agg['sgrnas_mean'] + agg['sgrnas_std'],
                 alpha=0.5, color='grey', linewidth=1)
ax1.set_xlabel('Number of unique genes to target', fontsize=11)
ax1.set_ylabel('(greedy) minimal number of sgRNAs', color='black', fontsize=11)
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xscale('log')
ax1.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=10))
ax1.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10)*0.1, numticks=50))
ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax2.plot(x, agg['uncov_mean'], color='darkred', linewidth=1,
         linestyle='--', label='Sequences that could not be targeted')
ax2.fill_between(x,
                 agg['uncov_mean'] - agg['uncov_std'],
                 agg['uncov_mean'] + agg['uncov_std'],
                 alpha=0.15, color='grey')
ax2.set_ylabel('Untargeted sequences', color='darkred', fontsize=11)
ax2.tick_params(axis='y', labelcolor='darkred')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

ax1.set_title(f'Number of sgRNAs needed / Number of genes to target\n'
              f'({n_genes_total} total genes)', fontsize=11)
ax1.spines['top'].set_visible(False)

plt.tight_layout()
#plt.show()



plt.savefig(OUT_DIR / 'sgrna_scaling_by_gene.png', dpi=300)
plt.close()
print(f'Scaling plot  → {OUT_DIR}/sgrna_scaling_by_gene.png')