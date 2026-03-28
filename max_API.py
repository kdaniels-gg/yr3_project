


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
SGRNA_CSV   = Path('crispr_results/sgrna_minimal_set.csv')
GOF_NUC_CSV = Path('gof_mapping_results/gof_positions_per_pid.csv')
OUT_BASE    = Path('bystander_evo2_results_max')
OUT_BASE.mkdir(exist_ok=True)

PLSDB_META_PATH = Path('plsdb_meta')   # optional; taxonomy enrichment skipped if absent

NVIDIA_API_KEY = os.getenv(
    'NVIDIA_API_KEY',
    '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'
)
EVO2_URL = os.getenv(
    'EVO2_URL',
    'insert key here'
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

GOF_MUTATIONS = {
    'TEM': [
        {'ref_pos': 104, 'ref_aa': 'E', 'alt_aa': None,  'category': 'resistance',          'note': 'Glu104'},
        {'ref_pos': 164, 'ref_aa': 'R', 'alt_aa': None,  'category': 'resistance',          'note': 'Arg164'},
        {'ref_pos': 179, 'ref_aa': 'D', 'alt_aa': None,  'category': 'resistance',          'note': 'Asp179'},
        {'ref_pos': 237, 'ref_aa': 'A', 'alt_aa': None,  'category': 'resistance',          'note': 'Ala237'},
        {'ref_pos': 238, 'ref_aa': 'G', 'alt_aa': None,  'category': 'resistance',          'note': 'Gly238'},
        {'ref_pos': 240, 'ref_aa': 'E', 'alt_aa': None,  'category': 'resistance',          'note': 'Glu240'},
        {'ref_pos': 173, 'ref_aa': 'I', 'alt_aa': 'V',   'category': 'resistance',          'note': 'I173V'},
        {'ref_pos': 254, 'ref_aa': 'D', 'alt_aa': 'G',   'category': 'resistance',          'note': 'D254G'},
        {'ref_pos': 184, 'ref_aa': 'A', 'alt_aa': 'V',   'category': 'adaptive',            'note': 'A184V'},
        {'ref_pos': 265, 'ref_aa': 'T', 'alt_aa': 'M',   'category': 'adaptive',            'note': 'T265M'},
        {'ref_pos': 268, 'ref_aa': 'S', 'alt_aa': 'G',   'category': 'adaptive',            'note': 'S268G'},
        {'ref_pos': 175, 'ref_aa': 'N', 'alt_aa': 'I',   'category': 'adaptive',            'note': 'N175I'},
        {'ref_pos':  21, 'ref_aa': 'L', 'alt_aa': 'F',   'category': 'adaptive',            'note': 'L21F'},
        {'ref_pos': 224, 'ref_aa': 'A', 'alt_aa': 'V',   'category': 'adaptive',            'note': 'A224V'},
        {'ref_pos':  39, 'ref_aa': 'Q', 'alt_aa': 'K',   'category': 'adaptive',            'note': 'Q39K'},
        {'ref_pos': 275, 'ref_aa': 'R', 'alt_aa': 'L',   'category': 'adaptive',            'note': 'R275L'},
        {'ref_pos':  42, 'ref_aa': 'A', 'alt_aa': 'V',   'category': 'stability',           'note': 'Ala42Val'},
        {'ref_pos':  51, 'ref_aa': 'L', 'alt_aa': 'P',   'category': 'stability',           'note': 'Leu51Pro'},
        {'ref_pos':  69, 'ref_aa': None,'alt_aa': None,   'category': 'stability',           'note': 'pos69'},
        {'ref_pos': 130, 'ref_aa': None,'alt_aa': None,   'category': 'stability',           'note': 'pos130'},
        {'ref_pos': 187, 'ref_aa': None,'alt_aa': None,   'category': 'stability',           'note': 'pos187'},
        {'ref_pos': 244, 'ref_aa': None,'alt_aa': None,   'category': 'stability',           'note': 'pos244'},
        {'ref_pos': 275, 'ref_aa': None,'alt_aa': None,   'category': 'stability',           'note': 'pos275'},
        {'ref_pos': 276, 'ref_aa': None,'alt_aa': None,   'category': 'stability',           'note': 'pos276'},
        {'ref_pos':  31, 'ref_aa': 'V', 'alt_aa': 'R',   'category': 'stability_clinical',  'note': 'V31R'},
        {'ref_pos':  47, 'ref_aa': 'I', 'alt_aa': 'V',   'category': 'stability_clinical',  'note': 'I47V'},
        {'ref_pos':  60, 'ref_aa': 'F', 'alt_aa': 'Y',   'category': 'stability_clinical',  'note': 'F60Y'},
        {'ref_pos':  62, 'ref_aa': 'P', 'alt_aa': 'S',   'category': 'stability_clinical',  'note': 'P62S'},
        {'ref_pos':  78, 'ref_aa': 'G', 'alt_aa': 'A',   'category': 'stability_clinical',  'note': 'G78A'},
        {'ref_pos':  80, 'ref_aa': 'V', 'alt_aa': 'I',   'category': 'stability_clinical',  'note': 'V80I'},
        {'ref_pos':  82, 'ref_aa': 'S', 'alt_aa': 'H',   'category': 'stability_clinical',  'note': 'S82H'},
        {'ref_pos':  92, 'ref_aa': 'G', 'alt_aa': 'D',   'category': 'stability_clinical',  'note': 'G92D'},
        {'ref_pos': 120, 'ref_aa': 'R', 'alt_aa': 'G',   'category': 'stability_clinical',  'note': 'R120G'},
        {'ref_pos': 147, 'ref_aa': 'E', 'alt_aa': 'G',   'category': 'stability_clinical',  'note': 'E147G'},
        {'ref_pos': 153, 'ref_aa': 'H', 'alt_aa': 'R',   'category': 'stability_clinical',  'note': 'H153R'},
        {'ref_pos': 182, 'ref_aa': 'M', 'alt_aa': 'T',   'category': 'stability_clinical',  'note': 'M182T'},
        {'ref_pos': 184, 'ref_aa': 'A', 'alt_aa': 'V',   'category': 'stability_clinical',  'note': 'A184V'},
        {'ref_pos': 188, 'ref_aa': 'T', 'alt_aa': 'I',   'category': 'stability_clinical',  'note': 'T188I'},
        {'ref_pos': 201, 'ref_aa': 'L', 'alt_aa': 'P',   'category': 'stability_clinical',  'note': 'L201P'},
        {'ref_pos': 208, 'ref_aa': 'I', 'alt_aa': 'M',   'category': 'stability_clinical',  'note': 'I208M'},
        {'ref_pos': 224, 'ref_aa': 'A', 'alt_aa': 'V',   'category': 'stability_clinical',  'note': 'A224V'},
        {'ref_pos': 240, 'ref_aa': 'E', 'alt_aa': 'H',   'category': 'stability_clinical',  'note': 'E240H'},
        {'ref_pos': 241, 'ref_aa': 'R', 'alt_aa': 'H',   'category': 'stability_clinical',  'note': 'R241H'},
        {'ref_pos': 247, 'ref_aa': 'I', 'alt_aa': 'V',   'category': 'stability_clinical',  'note': 'I247V'},
        {'ref_pos': 265, 'ref_aa': 'T', 'alt_aa': 'M',   'category': 'stability_clinical',  'note': 'T265M'},
        {'ref_pos': 275, 'ref_aa': 'R', 'alt_aa': 'Q',   'category': 'stability_clinical',  'note': 'R275Q'},
        {'ref_pos': 275, 'ref_aa': 'R', 'alt_aa': 'L',   'category': 'stability_clinical',  'note': 'R275L'},
        {'ref_pos': 276, 'ref_aa': 'N', 'alt_aa': 'D',   'category': 'stability_clinical',  'note': 'N276D'},
        {'ref_pos':  15, 'ref_aa': 'A', 'alt_aa': 'T',   'category': 'resistance',          'note': 'A15T'},
        {'ref_pos':  39, 'ref_aa': 'Q', 'alt_aa': 'K',   'category': 'resistance',          'note': 'Q39K'},
        {'ref_pos':  39, 'ref_aa': 'Q', 'alt_aa': 'R',   'category': 'resistance',          'note': 'Q39R'},
        {'ref_pos':  51, 'ref_aa': 'L', 'alt_aa': 'P',   'category': 'resistance',          'note': 'L51P'},
        {'ref_pos': 139, 'ref_aa': 'L', 'alt_aa': None,  'category': 'resistance',          'note': 'L139'},
    ],
    'SHV': [
        {'ref_pos': 146, 'ref_aa': 'A', 'alt_aa': None,  'category': 'resistance',  'note': 'Ala146'},
        {'ref_pos': 156, 'ref_aa': 'G', 'alt_aa': None,  'category': 'resistance',  'note': 'Gly156'},
        {'ref_pos': 169, 'ref_aa': 'L', 'alt_aa': None,  'category': 'resistance',  'note': 'Leu169'},
        {'ref_pos': 179, 'ref_aa': 'D', 'alt_aa': None,  'category': 'resistance',  'note': 'Asp179'},
        {'ref_pos': 205, 'ref_aa': 'R', 'alt_aa': None,  'category': 'resistance',  'note': 'Arg205'},
        {'ref_pos': 238, 'ref_aa': 'G', 'alt_aa': None,  'category': 'resistance',  'note': 'Gly238'},
        {'ref_pos': 240, 'ref_aa': 'E', 'alt_aa': None,  'category': 'resistance',  'note': 'Glu240'},
        {'ref_pos':  69, 'ref_aa': None,'alt_aa': None,   'category': 'stability',   'note': 'pos69'},
        {'ref_pos': 130, 'ref_aa': None,'alt_aa': None,   'category': 'stability',   'note': 'pos130'},
        {'ref_pos': 187, 'ref_aa': None,'alt_aa': None,   'category': 'stability',   'note': 'pos187'},
        {'ref_pos': 244, 'ref_aa': None,'alt_aa': None,   'category': 'stability',   'note': 'pos244'},
        {'ref_pos': 275, 'ref_aa': None,'alt_aa': None,   'category': 'stability',   'note': 'pos275'},
        {'ref_pos': 276, 'ref_aa': None,'alt_aa': None,   'category': 'stability',   'note': 'pos276'},
    ],
    'NDM': [
        {'ref_pos': 135, 'ref_aa': 'S', 'alt_aa': None,  'category': 'resistance',  'note': 'NDM-1 135S'},
        {'ref_pos': 154, 'ref_aa': 'M', 'alt_aa': 'L',   'category': 'resistance',  'note': 'NDM-1 M154L'},
    ],
    'AmpC': [
        {'ref_pos': 150, 'ref_aa': 'Y', 'alt_aa': None,  'category': 'resistance',  'note': 'Y150'},
        {'ref_pos': 346, 'ref_aa': 'N', 'alt_aa': None,  'category': 'resistance',  'note': 'N346'},
        {'ref_pos': 237, 'ref_aa': 'S', 'alt_aa': 'H',   'category': 'resistance',  'note': 'S237H'},
        {'ref_pos': 148, 'ref_aa': 'R', 'alt_aa': 'P',   'category': 'resistance',  'note': 'R148P'},
    ],
    'BlaC': [
        {'ref_pos': 105, 'ref_aa': 'I', 'alt_aa': 'F',   'category': 'resistance',  'note': 'I105F'},
        {'ref_pos': 184, 'ref_aa': 'H', 'alt_aa': 'R',   'category': 'resistance',  'note': 'H184R'},
        {'ref_pos': 263, 'ref_aa': 'V', 'alt_aa': 'I',   'category': 'resistance',  'note': 'V263I'},
    ],
    'CTX-M': [
        {'ref_pos':  77, 'ref_aa': 'A', 'alt_aa': 'V',   'category': 'stability',   'note': 'A77V'},
    ],
    'KPC': [
        {'ref_pos': 104, 'ref_aa': 'P', 'alt_aa': 'R',   'category': 'resistance',  'note': 'P104R'},
        {'ref_pos': 104, 'ref_aa': 'P', 'alt_aa': 'L',   'category': 'resistance',  'note': 'P104L'},
        {'ref_pos': 240, 'ref_aa': 'V', 'alt_aa': 'G',   'category': 'resistance',  'note': 'V240G'},
        {'ref_pos': 240, 'ref_aa': 'V', 'alt_aa': 'A',   'category': 'resistance',  'note': 'V240A'},
        {'ref_pos': 274, 'ref_aa': 'H', 'alt_aa': 'Y',   'category': 'resistance',  'note': 'H274Y'},
    ],
}

FAMILIES_OF_INTEREST = list(GOF_MUTATIONS.keys())   # TEM SHV NDM AmpC BlaC CTX-M KPC

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

# -----------------------------------------------------------------------------
# STEP 3: INSTANTLY SCORE ALL MUTATIONS
# -----------------------------------------------------------------------------
print("\nApplying precomputed logits to calculate scores...")


input_columns = [c for c in df_mut.columns if c != '_fasta_uid']
evo2_columns  = ['scored_nt_pos_0based', 'wt_nt', 'mut_nt', 
                 'wt_score', 'mut_score', 'delta_ll', 'prediction']
with open(OUTPUT_CSV, 'w') as f:
    f.write(','.join(input_columns + evo2_columns) + '\n')
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
        f.write(','.join(out_row) + '\n')



print(f"\nDone! Scored results written to: {OUTPUT_CSV}")





prediction_res_df = pd.read_csv(OUTPUT_CSV)

uh_oh = prediction_res_df.loc[prediction_res_df['prediction']=='Gain of Function / Stable']


#ONLY 12 rows are GOF - 
#This only renders 8 GOF out of 28209 query ids.







# =============================================================================
# SECTION 11 — EVO2 API SCORING
# =============================================================================
# Scores every unique edited sequence in the all-families Evo2 input against
# Evo2-40B via the NVIDIA HealthAI API.
#
#OUTPUT_CSV = OUT_BASE / 'all_families' / 'evo2_scored_output.csv'
#FASTA_FILE = OUT_BASE / 'all_families' / 'nuc_seqs_for_evo2.fa'
#
## Wipe previous partial run if user wants a clean start.
## Comment out the next two lines to *resume* an interrupted run instead.
## OUTPUT_CSV.unlink(missing_ok=True)
#
#headers_api = {
#    "Authorization": f"Bearer {NVIDIA_API_KEY}",
#    "Content-Type":  "application/json",
#}
#
#print(f'\nLoading FASTA for Evo2 scoring: {FASTA_FILE}')
#edited_sequences = {}
#for record in SeqIO.parse(str(FASTA_FILE), 'fasta'):
#    # FASTA header format: query_id|editor|ntPOS_POS_...
#    # query_id is the first pipe-delimited field (or the whole id if no pipe)
#    qid_raw = record.id
#    edited_sequences[qid_raw] = str(record.seq).upper()
#print(f'  Loaded {len(edited_sequences):,} edited sequences.')
#
#print(f'Loading mutation candidates CSV: {MUTATIONS_CSV}')
#df_mut = pd.read_csv(MUTATIONS_CSV)
#if 'Unnamed: 0' in df_mut.columns:
#    df_mut = df_mut.drop(columns=['Unnamed: 0'])
#print(f'  {len(df_mut):,} rows loaded.')
#
## ── BUILD THE FASTA LOOKUP KEY matching the FASTA header exactly ─────────────
## The FASTA uid written in run_scope is:  {pid}|{ed_name}|nt{pos_pos_...}
## The mutations CSV has columns: query_id, editor, edited_nt_positions
#def make_fasta_uid(query_id, editor, edited_nt_positions):
#    pos_str = '_'.join(str(p) for p in str(edited_nt_positions).split(','))
#    return f"{query_id}|{editor}|nt{pos_str}"
#
#df_mut['_fasta_uid'] = df_mut.apply(
#    lambda r: make_fasta_uid(r['query_id'], r['editor'], r['edited_nt_positions']),
#    axis=1
#)
#
## ── MUTATION COUNT ────────────────────────────────────────────────────────────
#total_mutations = len(df_mut)
#print(f'\n{"="*60}')
#print(f'TOTAL MUTATIONS TO SCORE VIA EVO2 API: {total_mutations:,}')
#print(f'{"="*60}')
#if total_mutations > 100_000:
#    print(f'WARNING: {total_mutations:,} calls is a large run.')
#    print('  Consider scoring only gof_codon_overlap==True rows first.')
#    print('  To do so, filter df_mut before running the loop:')
#    print('    df_mut = df_mut[df_mut["gof_codon_overlap"] == True]')
#
#input_columns = [c for c in df_mut.columns if c != '_fasta_uid']
#evo2_columns  = ['scored_nt_pos_0based', 'wt_nt', 'mut_nt',
#                 'wt_score', 'mut_score', 'delta_ll', 'prediction']
#all_output_cols = input_columns + evo2_columns
#
#
#def make_uid(query_id: str, nt_positions_str: str) -> str:
#    return f"{query_id}|{nt_positions_str}"
#
#
## ── RESUME support ────────────────────────────────────────────────────────────
#if OUTPUT_CSV.exists() and os.stat(OUTPUT_CSV).st_size > 0:
#    resume_df = pd.read_csv(OUTPUT_CSV)
#    if 'edited_nt_positions' in resume_df.columns:
#        processed_ids = set(
#            make_uid(str(r['query_id']), str(r['edited_nt_positions']))
#            for _, r in resume_df.iterrows()
#        )
#    else:
#        processed_ids = set(
#            make_uid(str(r['query_id']), str(r.get('scored_nt_pos_0based', '')))
#            for _, r in resume_df.iterrows()
#        )
#    print(f'Resuming — {len(processed_ids):,} entries already processed.')
#else:
#    processed_ids = set()
#    with open(OUTPUT_CSV, 'w') as fh:
#        fh.write(','.join(all_output_cols) + '\n')
#
#
## ── Evo2 API call ─────────────────────────────────────────────────────────────
#def fetch_evo2_logits(sequence: str, max_retries: int = 5):
#    payload = {"sequence": sequence, "num_tokens": 1, "enable_logits": True}
#    for attempt in range(max_retries):
#        try:
#            r = requests.post(url=EVO2_URL, headers=headers_api,
#                              json=payload, timeout=60)
#            if 400 <= r.status_code < 500 and r.status_code != 429:
#                print(f'    [!] Permanent client error {r.status_code}: {r.text[:300]}')
#                return None
#            if r.status_code == 429 or r.status_code >= 500:
#                r.raise_for_status()
#            response_data = r.json()
#            if 'logits' not in response_data or not response_data['logits']:
#                return None
#            logits = response_data['logits'][0]
#            if len(logits) < 85:   # need at least ASCII index 84 (T)
#                return None
#            return logits
#        except requests.exceptions.RequestException as e:
#            wait = 2 ** attempt
#            print(f'    [!] Network error (attempt {attempt+1}/{max_retries}): {e}. '
#                  f'Retrying in {wait}s...')
#            time.sleep(wait)
#        except (KeyError, ValueError, IndexError) as e:
#            wait = 2 ** attempt
#            print(f'    [!] Parse error (attempt {attempt+1}/{max_retries}): {e}. '
#                  f'Retrying in {wait}s...')
#            time.sleep(wait)
#    return None
#
#
## ── Bounded LRU prefix cache — prevents unbounded memory growth ───────────────
#class BoundedLRUCache:
#    """OrderedDict-backed LRU cache with a fixed capacity."""
#    def __init__(self, maxsize):
#        self._cache   = OrderedDict()
#        self._maxsize = maxsize
#
#    def get(self, key):
#        if key not in self._cache:
#            return None
#        self._cache.move_to_end(key)
#        return self._cache[key]
#
#    def set(self, key, value):
#        if key in self._cache:
#            self._cache.move_to_end(key)
#        else:
#            if len(self._cache) >= self._maxsize:
#                self._cache.popitem(last=False)   # evict oldest
#        self._cache[key] = value
#
#    def __len__(self):
#        return len(self._cache)
#
#
#prefix_cache   = BoundedLRUCache(PREFIX_CACHE_MAX)
#api_calls_made = 0
#cache_hits     = 0
#
#skipped_no_seq   = 0
#skipped_bad_pos  = 0
#skipped_api_fail = 0
#skipped_bad_nuc  = 0
#
#t_start = time.time()
#
## =============================================================================
## MAIN SCORING LOOP
## =============================================================================
#
#with open(OUTPUT_CSV, 'a') as f:
#    for index, row in df_mut.iterrows():
#        query_id   = str(row['query_id'])
#        nt_pos_str = str(row['edited_nt_positions'])
#        uid        = make_uid(query_id, nt_pos_str)
#
#        if uid in processed_ids:
#            continue
#
#        # Look up the edited sequence via the FASTA uid
#        fasta_uid  = row['_fasta_uid']
#        edited_seq = edited_sequences.get(fasta_uid)
#
#        # Fallback: some tools strip the suffix — try query_id alone
#        if edited_seq is None:
#            edited_seq = edited_sequences.get(query_id)
#
#        if edited_seq is None:
#            skipped_no_seq += 1
#            continue
#
#        try:
#            # edited_nt_positions is already 0-based (comma-separated)
#            positions_0based = [int(p) for p in nt_pos_str.split(',')]
#        except ValueError:
#            skipped_bad_pos += 1
#            continue
#
#        score_pos = max(positions_0based)
#
#        if score_pos >= len(edited_seq):
#            skipped_bad_pos += 1
#            continue
#
#        editor  = str(row.get('editor', 'CBE'))
#        mut_nuc = edited_seq[score_pos]
#        wt_nuc  = get_wt_nucleotide(mut_nuc, editor)
#
#        if mut_nuc not in 'ACGT' or wt_nuc not in 'ACGT':
#            skipped_bad_nuc += 1
#            continue
#
#        prefix_seq = edited_seq[:score_pos]
#
#        # ── Cached or fresh API call ──────────────────────────────────────
#        cached = prefix_cache.get(prefix_seq)
#        if cached is not None:
#            logits     = cached
#            cache_hits += 1
#            status_tag = 'CACHED'
#        else:
#            logits = fetch_evo2_logits(prefix_seq)
#            if logits is None:
#                skipped_api_fail += 1
#                continue
#            prefix_cache.set(prefix_seq, logits)
#            api_calls_made += 1
#            status_tag = 'API CALL'
#            time.sleep(0.5)   # rate-limit only on real API calls
#
#        wt_score   = logits[ord(wt_nuc)]
#        mut_score  = logits[ord(mut_nuc)]
#        delta_ll   = mut_score - wt_score
#        prediction = 'Gain of Function / Stable' if delta_ll > 0 else 'Damaging / LoF'
#
#        # ETA
#        elapsed   = time.time() - t_start
#        done_now  = api_calls_made + cache_hits
#        total_rem = total_mutations - len(processed_ids) - done_now
#        if done_now > 0 and total_rem > 0:
#            rate    = done_now / elapsed
#            eta_s   = int(total_rem / rate) if rate > 0 else 0
#            eta_str = f'ETA ~{eta_s//3600}h{(eta_s%3600)//60}m'
#        else:
#            eta_str = ''
#
#        row_idx_display = index + 1
#        print(f'[{row_idx_display}/{total_mutations}] {query_id} | '
#              f'pos {score_pos} | len {len(prefix_seq)} | {status_tag}  {eta_str}')
#        print(f'    WT({wt_nuc}): {wt_score:.3f} | '
#              f'Mut({mut_nuc}): {mut_score:.3f} | '
#              f'dLL: {delta_ll:.3f} -> {prediction}')
#
#        out_row = [str(row[col]) if pd.notna(row[col]) else ''
#                   for col in input_columns]
#        out_row.extend([
#            str(score_pos),
#            wt_nuc,
#            mut_nuc,
#            f'{wt_score:.5f}',
#            f'{mut_score:.5f}',
#            f'{delta_ll:.5f}',
#            prediction,
#        ])
#        f.write(','.join(out_row) + '\n')
#        f.flush()
#
#
## =============================================================================
## SECTION 12 — FINAL REPORT
## =============================================================================
#
#print(f"""
#{"="*60}
#Evo2 scoring complete.
#{"="*60}
#  Total mutations submitted  : {total_mutations:,}
#  API calls made             : {api_calls_made:,}
#  Cache hits                 : {cache_hits:,}
#  Cache size at end          : {len(prefix_cache):,} / {PREFIX_CACHE_MAX:,}
#
#  Skipped — no sequence      : {skipped_no_seq:,}
#  Skipped — bad position     : {skipped_bad_pos:,}
#  Skipped — bad nucleotide   : {skipped_bad_nuc:,}
#  Skipped — API failure      : {skipped_api_fail:,}
#
#  Output CSV                 : {OUTPUT_CSV}
#""")
