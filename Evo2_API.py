import re
import os
import ast
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

# =============================================================================
# SECTION 1 — PATHS
# =============================================================================

AMR_CSV     = Path('amrfindermapped_beta_lactamases.csv')
NUC_FA      = Path('card_gof_reference/all_query_sequences.fa')
SGRNA_CSV   = Path('crispr_results_final/sgrna_minimal_set.csv')
GOF_NUC_CSV = Path('gof_mapping_results_final/gof_positions_per_pid.csv')
OUT_BASE    = Path('bystander_evo2_results_final')
OUT_BASE.mkdir(exist_ok=True)

# Enumeration settings
MAX_RUN_LEN         = 4     # cap on contiguous-C run length before skipping run combos
INCLUDE_RUN_COMBOS  = True  # Tier 2: contiguous-C run subsets
INCLUDE_CODON_PAIRS = True  # Tier 3: same-codon C pairs

print('Paths configured.')
print(f'  sgRNA set : {SGRNA_CSV}')
print(f'  Output    : {OUT_BASE}')
print(f'  Run combos (Tier 2): {INCLUDE_RUN_COMBOS}, max run len: {MAX_RUN_LEN}')
print(f'  Codon pairs (Tier 3): {INCLUDE_CODON_PAIRS}')


# =============================================================================
# SECTION 2 — CONSTANTS
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
# SECTION 3 — HELPERS
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
    Return list of 'WTpos MUT' strings for every codon that changes.
    E.g. ['A42V', 'G238*'].  Synonymous changes noted as 'R104R (syn)'.
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
    Given a sorted list of integer positions, return a list of runs where each
    run is a list of consecutive integers.
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


def combos_for_window(c_positions, nt_seq):
    """
    Return a set of frozensets, each representing one edit combination to try.

    Tier 1: every single C position.
    Tier 2: all non-empty subsets within each contiguous run (len <= MAX_RUN_LEN).
    Tier 3: all pairs of Cs within the same codon.
    """
    combos = set()

    # Tier 1 — singles (always)
    for p in c_positions:
        combos.add(frozenset([p]))

    if INCLUDE_RUN_COMBOS:
        for run in contiguous_runs(sorted(c_positions)):
            if len(run) <= MAX_RUN_LEN:
                for r in range(2, len(run) + 1):   # pairs, triples, ... within run
                    for sub in itertools.combinations(run, r):
                        combos.add(frozenset(sub))
            # if run > MAX_RUN_LEN, we still get all singles from Tier 1

    if INCLUDE_CODON_PAIRS:
        # group by codon
        by_codon = defaultdict(list)
        for p in c_positions:
            by_codon[p // 3].append(p)
        for codon_positions in by_codon.values():
            if len(codon_positions) >= 2:
                for pair in itertools.combinations(codon_positions, 2):
                    combos.add(frozenset(pair))
                if len(codon_positions) >= 3:
                    combos.add(frozenset(codon_positions))

    return combos


def gof_overlaps(combo_positions, gof_wins):
    """True if any edited position's codon overlaps a GOF nucleotide window."""
    for p in combo_positions:
        codon_s = (p // 3) * 3
        codon_e = codon_s + 2
        if any(gs <= codon_e and codon_s <= ge for gs, ge in gof_wins):
            return True
    return False


print('Helpers ready.')


# =============================================================================
# SECTION 4 — LOAD SHARED DATA
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
    print('WARNING: GOF nuc CSV not found — gof_overlap will be empty')


# =============================================================================
# SECTION 5 — CORE STREAMING ENUMERATOR
# =============================================================================
# Writes rows directly to CSV as each guide is processed.
# Returns (n_combo_rows, n_evo2_rows) counts.

CSV_COLS = [
    'query_id', 'gene_name', 'gene_family',
    'editor', 'protospacer', 'strand',
    'win_s_cds', 'win_e_cds',
    'n_c_in_window', 'combo_tier', 'combo_size',
    'edited_nt_positions',    # comma-sep 0-indexed CDS positions
    'edited_aa_pos_1',        # comma-sep 1-indexed AA positions (deduplicated)
    'aa_changes',             # e.g. "A42V; G238*"  or "synonymous"
    'has_premature_stop',
    'gof_codon_overlap',
    'edited_cds',
]


def run_scope(scope_pids, pid_to_gene, pid_to_family, out_dir, label):
    """
    Stream enumeration for one scope.  Writes CSV and FASTA incrementally.

    Parameters
    ----------
    scope_pids    : set of query_id strings in this scope
    pid_to_gene   : dict query_id -> gene_name
    pid_to_family : dict query_id -> gene_family
    out_dir       : Path for output files
    label         : string for progress printing
    """
    out_dir.mkdir(exist_ok=True)
    csv_path  = out_dir / 'bystander_mutations.csv'
    evo2_path = out_dir / 'bystander_EVO2_input.csv'
    fa_path   = out_dir / 'nuc_seqs_for_evo2.fa'

    # Seen set for Evo2 dedup: (query_id, edited_cds) -> True
    # We track as a set of hashes to avoid holding full strings twice.
    # Key: (pid, hash(edited_cds)) — collision risk negligible at this scale.
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
                          f'| combos so far: {n_combos:,}  evo2: {n_evo2:,}')

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

                # ── Collect C positions passing context-adjusted window ────
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

                # ── Get combos and classify tier ────────────────────────────
                # Build mapping combo -> tier label
                combo_tier_map = {}

                # Tier 1 singles
                for p in c_positions:
                    fs = frozenset([p])
                    combo_tier_map[fs] = 'single'

                if INCLUDE_RUN_COMBOS:
                    for run in contiguous_runs(sorted(c_positions)):
                        if len(run) <= MAX_RUN_LEN:
                            for r in range(2, len(run) + 1):
                                for sub in itertools.combinations(run, r):
                                    fs = frozenset(sub)
                                    combo_tier_map.setdefault(fs, f'run{len(sub)}')

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

                # ── Emit one row per combo ───────────────────────────────────
                for combo_fs, tier in combo_tier_map.items():
                    combo = tuple(sorted(combo_fs))
                    edited_nt  = apply_c_to_t(nt, combo)
                    changes    = aa_changes(nt, edited_nt)
                    has_stop   = any('*' in c for c in changes)
                    gof_hit    = gof_overlaps(combo, gof_wins)
                    aa_pos_str = ','.join(str(p // 3 + 1) for p in combo)
                    # deduplicate AA positions (multiple Cs in same codon -> same aa pos)
                    aa_pos_dedup = ','.join(
                        dict.fromkeys(str(p // 3 + 1) for p in combo))

                    row = {
                        'query_id':           pid,
                        'gene_name':          gene,
                        'gene_family':        family,
                        'editor':             ed_name,
                        'protospacer':        protospacer,
                        'strand':             strand,
                        'win_s_cds':          win_s,
                        'win_e_cds':          win_e,
                        'n_c_in_window':      k,
                        'combo_tier':         tier,
                        'combo_size':         len(combo),
                        'edited_nt_positions': ','.join(str(p) for p in combo),
                        'edited_aa_pos_1':    aa_pos_dedup,
                        'aa_changes':         '; '.join(changes) if changes else 'synonymous',
                        'has_premature_stop': has_stop,
                        'gof_codon_overlap':  gof_hit,
                        'edited_cds':         edited_nt,
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
    # Re-read CSV for breakdown stats (small enough now that it's fine)
    df = pd.read_csv(csv_path, usecols=[
        'query_id', 'editor', 'protospacer', 'strand',
        'combo_tier', 'combo_size', 'has_premature_stop', 'gof_codon_overlap',
    ])
    n_pids   = df['query_id'].nunique()
    n_guides = df[['editor','protospacer','strand']].drop_duplicates().shape[0]

    lines = [
        f'Scope : {label}',
        f'PIDs with mutations enumerated : {n_pids:>8,}',
        f'Unique CBE guides applied      : {n_guides:>8,}',
        f'Total combo rows               : {n_combos:>8,}',
        f'Unique Evo2 calls (deduped)    : {n_evo2:>8,}',
        f'  of which GOF overlap         : {int(df["gof_codon_overlap"].sum()):>8,}',
        f'  of which premature stop      : {int(df["has_premature_stop"].sum()):>8,}',
        '',
        'By tier:',
    ]
    for tier, cnt in df['combo_tier'].value_counts().sort_index().items():
        lines.append(f'  {tier:<15s} : {cnt:,}')
    lines += ['', 'By combo size:']
    for sz, cnt in df['combo_size'].value_counts().sort_index().items():
        lines.append(f'  size {sz} : {cnt:,}')
    lines += ['', 'By editor:']
    for ed, cnt in df['editor'].value_counts().items():
        lines.append(f'  {ed:<20s} : {cnt:,}')

    summary = '\n'.join(lines) + '\n'
    (out_dir / 'summary.txt').write_text(summary)
    print(f'\n--- {label} ---')
    print(summary)
    return n_combos, n_evo2


print('Enumerator ready.')


# =============================================================================
# SECTION 6 — SCOPE A: TEM-1 ONLY
# =============================================================================

print('\n' + '='*60)
print('SCOPE A — TEM-1 only')
print('='*60)

tem1_rows = amr[amr['gene_family'] == 'TEM'].copy()
tem1_rows = tem1_rows[tem1_rows['gene_name'].str.upper().str.startswith('TEM-1')]
tem1_pids = set(tem1_rows['query_id'].unique())

pid_to_gene_A   = tem1_rows.set_index('query_id')['gene_name'].to_dict()
pid_to_family_A = tem1_rows.set_index('query_id')['gene_family'].to_dict()

print(f'TEM-1 PIDs: {len(tem1_pids):,}')
n_A, evo2_A = run_scope(
    tem1_pids, pid_to_gene_A, pid_to_family_A,
    OUT_BASE / 'A_TEM1', 'TEM-1'
)


# =============================================================================
# SECTION 7 — SCOPE B: ALL TEMs
# =============================================================================

print('\n' + '='*60)
print('SCOPE B — all TEMs')
print('='*60)

tem_rows  = amr[amr['gene_family'] == 'TEM'].copy()
tem_pids  = set(tem_rows['query_id'].unique())

pid_to_gene_B   = tem_rows.set_index('query_id')['gene_name'].to_dict()
pid_to_family_B = tem_rows.set_index('query_id')['gene_family'].to_dict()

print(f'All-TEM PIDs: {len(tem_pids):,}')
n_B, evo2_B = run_scope(
    tem_pids, pid_to_gene_B, pid_to_family_B,
    OUT_BASE / 'B_allTEM', 'all-TEM'
)


# =============================================================================
# SECTION 8 — SCOPE C: ALL DATA
# =============================================================================

print('\n' + '='*60)
print('SCOPE C — all families')
print('='*60)

all_pids = set(amr['query_id'].unique())

pid_to_gene_C   = amr.set_index('query_id')['gene_name'].to_dict()
pid_to_family_C = amr.set_index('query_id')['gene_family'].to_dict()

print(f'All PIDs: {len(all_pids):,}')
n_C, evo2_C = run_scope(
    all_pids, pid_to_gene_C, pid_to_family_C,
    OUT_BASE / 'C_allData', 'all-data'
)


# =============================================================================
# SECTION 9 — CROSS-SCOPE SUMMARY
# =============================================================================

print('\n' + '='*60)
print('CROSS-SCOPE FEASIBILITY SUMMARY')
print('='*60)

cross = pd.DataFrame([
    {'scope': 'A_TEM1',   'total_combos': n_A, 'evo2_calls': evo2_A},
    {'scope': 'B_allTEM', 'total_combos': n_B, 'evo2_calls': evo2_B},
    {'scope': 'C_allData','total_combos': n_C, 'evo2_calls': evo2_C},
])
print(cross.to_string(index=False))
cross.to_csv(OUT_BASE / 'cross_scope_summary.csv', index=False)

print(f'\nAll outputs under {OUT_BASE}/')
for p in sorted(OUT_BASE.rglob('*')):
    if p.is_file():
        kb = p.stat().st_size / 1024
        print(f'  {str(p.relative_to(OUT_BASE)):<55s}  {kb:>8.1f} KB')




all_input_csv = pd.read_csv(Path('bystander_evo2_results_final/all_families/bystander_EVO2_input.csv'))
all_idf = all_input_csv.copy(deep=True)


#TEM_1_input_csv = pd.read_csv(Path('bystander_evo2_results_final/A_TEM1/bystander_EVO2_input.csv'))
#TEM_1_idf = TEM_1_input_csv.copy(deep=True)


PID_nuccore_pattern = re.compile(r'^(.+?)_\d+_\d+')
PID_nogene_pattern  = re.compile(r'^(.+?)_(\d+)_(\d+)$')

#TEM_1_plas = list(set([PID_nuccore_pattern.match(x).group(1) if PID_nuccore_pattern.match(x) else None
#    for x in TEM_1_idf['query_id'].tolist()]))

all_plas = list(set([PID_nuccore_pattern.match(x).group(1) if PID_nuccore_pattern.match(x) else None
    for x in all_idf['query_id'].tolist()]))

plsdb_meta_path = Path('plsdb_meta')
nuc_df  = pd.read_csv(plsdb_meta_path / 'nuccore_only.csv')
tax_df  = pd.read_csv(plsdb_meta_path / 'taxonomy.csv')
nuc_taxuid = dict(zip(nuc_df['NUCCORE_ACC'], nuc_df['TAXONOMY_UID']))


taxuid_kingdom = dict(zip(tax_df['TAXONOMY_UID'], tax_df['TAXONOMY_superkingdom']))
taxuid_phylum = dict(zip(tax_df['TAXONOMY_UID'], tax_df['TAXONOMY_phylum']))
taxuid_order = dict(zip(tax_df['TAXONOMY_UID'], tax_df['TAXONOMY_order']))
taxuid_genus = dict(zip(tax_df['TAXONOMY_UID'], tax_df['TAXONOMY_genus']))
taxuid_species = dict(zip(tax_df['TAXONOMY_UID'], tax_df['TAXONOMY_species']))
taxuid_classs = dict(zip(tax_df['TAXONOMY_UID'], tax_df['TAXONOMY_class']))

plas_to_taxstring = {}

#for plas in TEM_1_plas:
#    tax_uid = nuc_taxuid[plas]
#    kingdom = taxuid_kingdom.get(tax_uid) if tax_uid in taxuid_kingdom.keys() else None
#    phylum = taxuid_phylum.get(tax_uid) if tax_uid in taxuid_phylum.keys() else None
#    classs = taxuid_classs.get(tax_uid) if tax_uid in taxuid_classs.keys() else None
#    order = taxuid_order.get(tax_uid) if tax_uid in taxuid_order.keys() else None
#    genus = taxuid_genus.get(tax_uid) if tax_uid in taxuid_genus.keys() else None
#    species = taxuid_species.get(tax_uid) if tax_uid in taxuid_species.keys() else None
#    if kingdom and phylum and order and genus and species:
#        tax_string = f'k__[{kingdom}];p__[{phylum}];c__[{classs}];o__[{order}];g__[{genus}];s__[{species}]|'
#        plas_to_taxstring[plas] = tax_string
#


for plas in all_plas:
    tax_uid = nuc_taxuid[plas]
    kingdom = taxuid_kingdom.get(tax_uid) if tax_uid in taxuid_kingdom.keys() else None
    phylum = taxuid_phylum.get(tax_uid) if tax_uid in taxuid_phylum.keys() else None
    classs = taxuid_classs.get(tax_uid) if tax_uid in taxuid_classs.keys() else None
    order = taxuid_order.get(tax_uid) if tax_uid in taxuid_order.keys() else None
    genus = taxuid_genus.get(tax_uid) if tax_uid in taxuid_genus.keys() else None
    species = taxuid_species.get(tax_uid) if tax_uid in taxuid_species.keys() else None
    if kingdom and phylum and order and genus and species:
        tax_string = f'k__[{kingdom}];p__[{phylum}];c__[{classs}];o__[{order}];g__[{genus}];s__[{species}]|'
        plas_to_taxstring[plas] = tax_string


#TEM_1_idf['plasmid'] = [PID_nuccore_pattern.match(x).group(1) if PID_nuccore_pattern.match(x) else None
#    for x in TEM_1_idf['query_id'].tolist()]
#
#
#TEM_1_idf['taxonomy_string'] = [plas_to_taxstring.get(x) if x in plas_to_taxstring.keys() else None for x in TEM_1_idf['plasmid'].tolist()]
#
#
#TEM_1_idf.to_csv(Path('bystander_evo2_results_final/A_TEM1/bystander_EVO2_input_with_tax_info.csv'))
#
#
#
#tax_lookup = dict(zip(TEM_1_idf['query_id'], TEM_1_idf['taxonomy_string']))



all_idf['plasmid'] = [PID_nuccore_pattern.match(x).group(1) if PID_nuccore_pattern.match(x) else None
    for x in all_idf['query_id'].tolist()]


all_idf['taxonomy_string'] = [plas_to_taxstring.get(x) if x in plas_to_taxstring.keys() else None for x in all_idf['plasmid'].tolist()]


all_idf.to_csv(Path('bystander_evo2_results_final/all_families/bystander_EVO2_input_with_tax_info.csv'))



tax_lookup = dict(zip(all_idf['query_id'], all_idf['taxonomy_string']))




##################################################


from pathlib import Path
#Path("bystander_evo2_results_final/A_TEM1/output.csv").unlink(missing_ok=True)
Path("bystander_evo2_results_final/all_families/output.csv").unlink(missing_ok=True)

import os
import time
import requests
import pandas as pd
from Bio import SeqIO

# =============================================================================
# EVO2 API SCORING  —  CBE/ABE bystander mutation fitness assessment
# =============================================================================

# --- CONFIGURATION ---
NVIDIA_API_KEY = "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"
URL = os.getenv("URL", "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate")

#FASTA_FILE    = Path('bystander_evo2_results_final/A_TEM1/nuc_seqs_for_evo2.fa')
#MUTATIONS_CSV = Path('bystander_evo2_results_final/A_TEM1/bystander_EVO2_input_with_tax_info.csv')
#OUTPUT_CSV    = Path('bystander_evo2_results_final/A_TEM1/output.csv')

FASTA_FILE    = Path('bystander_evo2_results_final/all_families/nuc_seqs_for_evo2.fa')
MUTATIONS_CSV = Path('bystander_evo2_results_final/all_families/bystander_EVO2_input_with_tax_info.csv')
OUTPUT_CSV    = Path('bystander_evo2_results_final/all_families/output.csv')

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

headers = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Content-Type": "application/json"
}

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

def fetch_evo2_logits(sequence: str, max_retries: int = 5):
    payload = {"sequence": sequence, "num_tokens": 1, "enable_logits": True}
    for attempt in range(max_retries):
        try:
            r = requests.post(url=URL, headers=headers, json=payload, timeout=60)
            if 400 <= r.status_code < 500 and r.status_code != 429:
                print(f"    [!] Permanent client error {r.status_code}: {r.text[:300]}")
                return None
            if r.status_code == 429 or r.status_code >= 500:
                r.raise_for_status()

            response_data = r.json()
            if 'logits' not in response_data or not response_data['logits']:
                return None
            
            logits = response_data['logits'][0]
            # Verify the logit vector is long enough to contain ASCII token indices (T = 84)
            if len(logits) < 85:
                return None
            return logits

        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f"    [!] Network error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
        except (KeyError, ValueError, IndexError) as e:
            wait = 2 ** attempt
            print(f"    [!] Parse error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)

    return None

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading edited CDS sequences from FASTA...")
edited_sequences = {}
for record in SeqIO.parse(FASTA_FILE, "fasta"):
    qid = record.id.split('|')[0]
    edited_sequences[qid] = str(record.seq).upper()
print(f"  Loaded {len(edited_sequences):,} edited sequences.")

print("Loading mutation candidates CSV...")
df = pd.read_csv(MUTATIONS_CSV)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
print(f"  {len(df):,} rows loaded.")

input_columns = list(df.columns)
evo2_columns  = ['scored_nt_pos_0based', 'wt_nt', 'mut_nt', 'wt_score', 'mut_score', 'delta_ll', 'prediction']
all_columns   = input_columns + evo2_columns

def make_uid(qid: str, nt_positions_str: str) -> str:
    return f"{qid}|{nt_positions_str}"

if OUTPUT_CSV.exists() and os.stat(OUTPUT_CSV).st_size > 0:
    resume_df = pd.read_csv(OUTPUT_CSV)
    if 'edited_nt_positions' in resume_df.columns:
        processed_ids = set(make_uid(str(r['query_id']), str(r['edited_nt_positions'])) for _, r in resume_df.iterrows())
    else:
        processed_ids = set(make_uid(str(r['query_id']), str(r.get('scored_nt_pos_0based', ''))) for _, r in resume_df.iterrows())
    print(f"Resuming — {len(processed_ids):,} entries already processed.")
else:
    processed_ids = set()

if not OUTPUT_CSV.exists() or os.stat(OUTPUT_CSV).st_size == 0:
    with open(OUTPUT_CSV, 'w') as fh:
        fh.write(",".join(all_columns) + "\n")

# =============================================================================
# MAIN SCORING LOOP
# =============================================================================
skipped_no_seq   = 0
skipped_bad_pos  = 0
skipped_api_fail = 0
skipped_bad_nuc  = 0

# --- THE CACHE DICTIONARY ---
prefix_cache = {}
api_calls_made = 0
cache_hits = 0

with open(OUTPUT_CSV, 'a') as f:
    for index, row in df.iterrows():
        query_id   = str(row['query_id'])
        nt_pos_str = str(row['edited_nt_positions'])
        uid        = make_uid(query_id, nt_pos_str)

        if uid in processed_ids:
            continue

        edited_seq = edited_sequences.get(query_id)
        if not edited_seq:
            skipped_no_seq += 1
            continue

        try:
            # FIX: Removed the "- 1". The edited_nt_positions column is ALREADY 0-based.
            positions_0based = [int(p) for p in nt_pos_str.split(',')]
        except ValueError:
            skipped_bad_pos += 1
            continue

        score_pos = max(positions_0based)

        if score_pos >= len(edited_seq):
            skipped_bad_pos += 1
            continue

        editor  = str(row.get('editor', 'CBE'))
        mut_nuc = edited_seq[score_pos]               
        wt_nuc  = get_wt_nucleotide(mut_nuc, editor)  

        # Ensure we only process valid standard DNA nucleotides
        if mut_nuc not in 'ACGT' or wt_nuc not in 'ACGT':
            skipped_bad_nuc += 1
            continue

        prefix_seq = edited_seq[:score_pos]

        # --- PREFIX CACHING LOGIC ---
        if prefix_seq in prefix_cache:
            logits = prefix_cache[prefix_seq]
            cache_hits += 1
            status_tag = "CACHED"
        else:
            logits = fetch_evo2_logits(prefix_seq)
            if logits is None:
                skipped_api_fail += 1
                continue
            
            # Save to cache for next time
            prefix_cache[prefix_seq] = logits
            api_calls_made += 1
            status_tag = "API CALL"
            time.sleep(0.5) # ONLY pause if an actual API call was made

        # Use ord() to get the correct Evo2 token indices (e.g., A=65, C=67, G=71, T=84)
        wt_score   = logits[ord(wt_nuc)]
        mut_score  = logits[ord(mut_nuc)]
        delta_ll   = mut_score - wt_score
        
        # A negative delta_ll means the mutation makes the sequence LESS fit/likely
        prediction = "Gain of Function / Stable" if delta_ll > 0 else "Damaging / LoF"

        print(f"[{index+1}/{len(df)}] {query_id} | pos {score_pos} | len {len(prefix_seq)} | {status_tag}")
        print(f"    WT({wt_nuc}): {wt_score:.3f} | Mut({mut_nuc}): {mut_score:.3f} | dLL: {delta_ll:.3f} -> {prediction}")

        out_row = [str(row[col]) if pd.notna(row[col]) else "" for col in input_columns]
        out_row.extend([
            str(score_pos),
            wt_nuc,
            mut_nuc,
            f"{wt_score:.5f}",
            f"{mut_score:.5f}",
            f"{delta_ll:.5f}",
            prediction,
        ])
        f.write(",".join(out_row) + "\n")
        f.flush()

print(f"""
Processing complete.
  Total API Calls Made  : {api_calls_made}
  Total Cache Hits      : {cache_hits}
  
  Skipped (no sequence)   : {skipped_no_seq}
  Skipped (bad position)  : {skipped_bad_pos}
  Skipped (bad nucleotide): {skipped_bad_nuc}
  Skipped (API failure)   : {skipped_api_fail}
""")