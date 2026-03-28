import os, math
from pathlib import Path
from Bio import SeqIO

COMBINED_FA  = Path('card_gof_reference/all_query_sequences_prot.fa')
CHUNK_DIR    = Path('blast_chunks')
CHUNK_SIZE   = 50_000    # sequences per chunk; 50k × blastx ~15-25 min per job at 36 threads

CHUNK_DIR.mkdir(exist_ok=True)

seqs      = list(SeqIO.parse(COMBINED_FA, 'fasta'))
n_total   = len(seqs)
n_chunks  = math.ceil(n_total / CHUNK_SIZE)
print(f'{n_total:,} sequences  →  {n_chunks} chunks of ≤{CHUNK_SIZE:,}')

for i in range(n_chunks):
    chunk_seqs = seqs[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
    out = CHUNK_DIR / f'chunk_{i+1:04d}.fa'
    SeqIO.write(chunk_seqs, out, 'fasta')

print(f'Chunks written to {CHUNK_DIR}/')
print(f'Submit with:  sbatch --array=1-{n_chunks} run_blast_array.slurm')





########################################################


import re
import json
import os
from pathlib import Path
import subprocess
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# ── config ───────────────────────────────────────────────────────────────────
BLAST_RESULTS_DIR = Path('blast_results')       # per-chunk TSVs from SLURM array
CARD_DIR          = Path('card_gof_reference')
COMBINED_FA       = CARD_DIR / 'all_query_sequences.fa'          # nucleotide
PROT_FA           = CARD_DIR / 'all_query_sequences_prot.fa'  # protein
OUT_DIR           = Path('all_card')
PIDENT_MIN        = 70.0
QCOV_MIN          = 70.0
PIDENT_RESCUE     = 50.0
QCOV_RESCUE       = 50.0
THREADS           = 8   # only used if re-running rescue blast locally

COLS = ['query_id','subject_id','pident','length','qlen','slen','evalue','bitscore','qcovs']


CARD_JSON    = Path(os.path.join(os.getcwd(), 'all_card/card.json'))

with open(CARD_JSON) as f:
    card = json.load(f)

aro_rows = []
ref_fasta_path = OUT_DIR / 'card_betalactamase_refs.fa'

with open(ref_fasta_path, 'w') as ref_out:
    for aro_id, entry in card.items():
        if not isinstance(entry, dict):
            continue
        categories = entry.get('ARO_category', {})
        is_bl = any(
            'beta-lactamase' in v.get('category_aro_name', '').lower()
            for v in categories.values()
        )
        if not is_bl:
            continue
        families = '; '.join(
            v.get('category_aro_name', '')
            for v in categories.values()
            if v.get('category_aro_class_name') == 'AMR Gene Family'
        )
        drug_classes = '; '.join(
            v.get('category_aro_name', '')
            for v in categories.values()
            if v.get('category_aro_class_name') == 'Drug Class'
        )
        for seq_data in entry.get('model_sequences', {}).get('sequence', {}).values():
            prot = seq_data.get('protein_sequence', {})
            acc  = prot.get('accession', '')
            seq  = prot.get('sequence', '')
            if not seq:
                continue
            aro_acc  = entry.get('ARO_accession', '')
            aro_name = entry.get('ARO_name', '')
            ref_out.write(f'>{aro_acc}|{aro_name}\n{seq}\n')
            aro_rows.append({
                'blast_id':      f'{aro_acc}|{aro_name}',
                'ARO_accession': aro_acc,
                'ARO_name':      aro_name,
                'gene_family':   families,
                'drug_class':    drug_classes,
                'ref_prot_acc':  acc,
            })

aro_df = pd.DataFrame(aro_rows).drop_duplicates('blast_id')



chunk_files = sorted(BLAST_RESULTS_DIR.glob('chunk_*.tsv'))
print(f'Merging {len(chunk_files)} chunk files...')

blast_df = pd.concat(
    [pd.read_csv(f, sep='\t', header=None, names=COLS) for f in tqdm(chunk_files)],
    ignore_index=True
)
print(f'Raw hits: {len(blast_df):,}')

blast_df = blast_df[
    (blast_df['pident'] >= PIDENT_MIN) &
    (blast_df['qcovs']  >= QCOV_MIN)
]
print(f'After filter: {len(blast_df):,}')

best_hits = (
    blast_df.sort_values('bitscore', ascending=False)
    .groupby('query_id', as_index=False)
    .first()
)
best_hits[['ARO_accession','ARO_name']] = best_hits['subject_id'].str.split('|', n=1, expand=True)
best_hits = best_hits.merge(
    aro_df[['ARO_accession','ARO_name','gene_family','drug_class','ref_prot_acc']],
    on=['ARO_accession','ARO_name'], how='left'
)
print(f'Unique query hits after best-hit selection: {len(best_hits):,}')


print("Rebuilding query_source index...")

query_source = {}
blfasta_dir = Path("beta_lactam_fastas")
pfam_fa = Path("pfam_betalactamase_genesequences.fa")

def extract_headers(fasta):
    result = subprocess.run(
        ["grep", "^>", str(fasta)],
        capture_output=True,
        text=True
    )
    return [line[1:].split()[0] for line in result.stdout.splitlines()]

# directory fastas
for fa in blfasta_dir.glob("*.fa"):
    for q in extract_headers(fa):
        query_source[q] = {"source": "fasta_dir", "filename": fa.name}

# pfam fasta
for q in extract_headers(pfam_fa):
    query_source[q] = {"source": "pfam", "filename": str(pfam_fa)}

print(f"query_source entries: {len(query_source):,}")

source_df = pd.DataFrame(
    ((k, v["source"], v["filename"]) for k, v in query_source.items()),
    columns=["query_id", "source", "filename"]
)



source_df_out_path = Path('query_source_index.csv')
#source_df.to_parquet(source_df_out_path, compression='zstd')


source_df = pd.read_parquet(source_df_out_path)


def parse_locus(qid):
    m = re.match(r'^(.+?)_(\d+)_(\d+)', qid)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return qid, None, None

result_df = source_df.merge(
    best_hits[['query_id','ARO_accession','ARO_name','gene_family',
               'drug_class','ref_prot_acc','pident','qcovs','evalue','bitscore']],
    on='query_id', how='left'
)
result_df[['plasmid','start','stop']] = pd.DataFrame(
    result_df['query_id'].apply(parse_locus).tolist(), index=result_df.index
)

mapped   = result_df['ARO_name'].notna().sum()
unmapped = result_df['ARO_name'].isna().sum()
print(f'mapped:   {mapped:,}')
print(f'unmapped: {unmapped:,}')
result_df.to_csv(OUT_DIR / 'betalactamase_card_mapping.csv', index=False)


clean_result_df = result_df[result_df['ARO_name'].apply(lambda x: isinstance(x, str))].copy()
clean_result_df['gene_family_rough'] = [
    '-'.join(x.split('-')[:-1]) for x in clean_result_df['ARO_name']
]
clean_result_df.to_csv(OUT_DIR / 'betalactamase_card_mapping_clean.csv', index=False)


unmapped_result_df = result_df[~result_df['ARO_name'].apply(lambda x: isinstance(x, str))].copy()
unmapped_ids = set(unmapped_result_df['query_id'])
print(f'Attempting rescue on {len(unmapped_ids):,} unmapped queries...')

rescue_hits = (
    blast_df[  # blast_df already loaded above — reuse with rescue thresholds
        (blast_df['query_id'].isin(unmapped_ids)) &
        (blast_df['pident'] >= PIDENT_RESCUE) &
        (blast_df['qcovs']  >= QCOV_RESCUE)
    ]
    .sort_values('bitscore', ascending=False)
    .groupby('query_id', as_index=False)
    .first()
)

if not rescue_hits.empty:
    rescue_hits[['ARO_accession','ARO_name']] = rescue_hits['subject_id'].str.split('|', n=1, expand=True)
    rescue_hits = rescue_hits.merge(
        aro_df[['ARO_accession','ARO_name','gene_family','drug_class','ref_prot_acc']],
        on=['ARO_accession','ARO_name'], how='left'
    )

    # NOTE: if rescue_hits is empty it means those queries genuinely had no hits
    # even at relaxed thresholds in the original blast run.
    # In that case you may want to re-blast unmapped_queries.fa with -evalue 1e-5
    # using a SLURM job (same template, just point QUERY at unmapped_queries.fa
    # and use a single job rather than array since the set should be much smaller).

    rescued = unmapped_result_df.drop(
        columns=['ARO_accession','ARO_name','gene_family','drug_class',
                 'ref_prot_acc','pident','qcovs','evalue','bitscore'], errors='ignore'
    ).merge(
        rescue_hits[['query_id','ARO_accession','ARO_name','gene_family',
                     'drug_class','ref_prot_acc','pident','qcovs','evalue','bitscore']],
        on='query_id', how='left'
    )
    newly_mapped   = rescued['ARO_name'].notna().sum()
    print(f'rescued:        {newly_mapped:,}')


    still_unmapped = rescued['ARO_name'].isna().sum()
    print(f'still unmapped: {still_unmapped:,}')

    # save unmapped fasta for optional manual/HPC rescue blast
    unmapped_fa = OUT_DIR / 'unmapped_queries.fa'
    unmapped_prot_fa = OUT_DIR / 'unmapped_queries_prot.fa'

    unmapped_prot_ids = set(rescued[rescued['ARO_name'].isna()]['query_id'])
    with open(unmapped_fa, 'w') as out_nuc, open(unmapped_prot_fa, 'w') as out_prot:
        for rec in SeqIO.parse(COMBINED_FA, 'fasta'):
            if rec.id in unmapped_prot_ids:
                out_nuc.write(f'>{rec.id}\n{rec.seq}\n')
        for rec in SeqIO.parse(PROT_FA, 'fasta'):
            if rec.id in unmapped_prot_ids and len(rec.seq) >= 50:
                out_prot.write(f'>{rec.id}\n{rec.seq}\n')

    print(f'Wrote {unmapped_prot_fa} for optional rescue blast')
    test1 = rescued.loc[rescued['source'] == 'fasta_dir'].copy()
    test2 = rescued.loc[rescued['ARO_name'].apply(lambda x: isinstance(x, str))].copy()
    test1['supposed_gene'] = [x.split('_')[-1] for x in test1['query_id']]
    test2['supposed_gene'] = None

    test3 = pd.concat([test1, test2]).drop_duplicates()
    test3['gene_family_rough'] = test3.apply(
        lambda r: r['supposed_gene']
        if r['supposed_gene'] is not None
        else ('-'.join(r['ARO_name'].split('-')[:-1]) if isinstance(r['ARO_name'], str) else None),
        axis=1
    )
    test3.to_csv(OUT_DIR / 'betalactamase_card_mapping_rescued.csv', index=False)
    cleantomerge = clean_result_df.copy()
    cleantomerge['supposed_gene'] = None
    test4 = pd.concat([test3, cleantomerge]).drop_duplicates()
    test4.to_csv(OUT_DIR / 'betalactamase_card_mapping_merged.csv', index=False)
    print(f'total overall: {len(test4):,}')
else:
    test4 = clean_result_df


REGULATORY_GENES = {
    'mecA': 'mec_system', 'mecB': 'mec_system',
    'mecI': 'mec_system', 'mecR1': 'mec_system',
    'blaI': 'bla_regulatory', 'blaR1': 'bla_regulatory',
    'blaMCA': 'bla_regulatory', 'bla': 'bla_unclassified',
    'blaOXA': 'OXA', 'blaCTX-M': 'CTX-M', 'blaTEM': 'TEM',
}

def extract_gene_family(rough_name):
    if rough_name != rough_name or rough_name is None:
        return float('nan')
    rough_name = str(rough_name).strip()
    if rough_name in REGULATORY_GENES:
        return REGULATORY_GENES[rough_name]
    m = re.match(r'^bla([A-Z].*)$', rough_name)
    if m:
        return m.group(1)
    m = re.match(r'^(.+?)-\d+[a-z]?$', rough_name)
    if m:
        candidate = m.group(1)
        if re.match(r'^[A-Za-z][A-Za-z0-9\-]*[A-Za-z]$|^[A-Za-z]+$', candidate):
            return candidate
    return rough_name

test4['gene_family_true'] = test4['gene_family_rough'].apply(extract_gene_family)
test5 = test4.loc[test4['gene_family_true'].apply(lambda x: isinstance(x, str))]
test5.to_csv('beta_lactamases_geneandfamily_mapped.csv', index=False)
print(f'Final mapped rows written: {len(test5):,}')
print(f'Unique gene families: {test5["gene_family_true"].nunique()}')
print(f'Unique gene names:    {test5["gene_probably"].nunique() if "gene_probably" in test5.columns else "N/A"}')


testy = test5[test5['ARO_name'].apply(lambda x: isinstance(x, str))]


test5 = pd.read_csv('beta_lactamases_geneandfamily_mapped.csv')



RESCUE_TSV = Path('blast_results/rescue.tsv')
COLS = ['query_id','subject_id','pident','length','qlen','slen',
        'evalue','bitscore','qcovs']

rescue_raw = pd.read_csv(RESCUE_TSV, sep='\t', header=None, names=COLS)
print(f'\nRescue blast hits (raw): {len(rescue_raw):,}')

rescue_filtered = rescue_raw[
    (rescue_raw['pident'] >= 50.0) &
    (rescue_raw['qcovs']  >= 50.0)
]
print(f'After 50/50 filter: {len(rescue_filtered):,}')

rescue_best = (rescue_filtered
               .sort_values('bitscore', ascending=False)
               .groupby('query_id', as_index=False).first())

# ── map subject_id -> ARO info via card.json ──────────────────────────────────
model_seq_to_aro = {}
for aro_id, entry in card.items():
    if not isinstance(entry, dict):
        continue
    model_id = entry.get('model_id')
    if model_id is None:
        continue
    for seq_id in entry.get('model_sequences', {}).get('sequence', {}).keys():
        key = f'{model_id}_{seq_id}'
        model_seq_to_aro[key] = {
            'ARO_accession': entry.get('ARO_accession', ''),
            'ARO_name':      entry.get('ARO_name', ''),
            'gene_family':   '; '.join(
                v.get('category_aro_name', '')
                for v in entry.get('ARO_category', {}).values()
                if v.get('category_aro_class_name') == 'AMR Gene Family'),
            'drug_class':    '; '.join(
                v.get('category_aro_name', '')
                for v in entry.get('ARO_category', {}).values()
                if v.get('category_aro_class_name') == 'Drug Class'),
        }

print(f'model_seq_to_aro entries: {len(model_seq_to_aro):,}')

rescue_best['ARO_accession'] = rescue_best['subject_id'].map(
    lambda x: model_seq_to_aro.get(x, {}).get('ARO_accession'))
rescue_best['ARO_name'] = rescue_best['subject_id'].map(
    lambda x: model_seq_to_aro.get(x, {}).get('ARO_name'))
rescue_best['gene_family'] = rescue_best['subject_id'].map(
    lambda x: model_seq_to_aro.get(x, {}).get('gene_family'))
rescue_best['drug_class'] = rescue_best['subject_id'].map(
    lambda x: model_seq_to_aro.get(x, {}).get('drug_class'))

print(f'Newly mappable from rescue: {rescue_best["ARO_name"].notna().sum():,}')

# ── keep only beta-lactamases, exclude anything already in test5 ──────────────
rescue_bl = rescue_best[
    rescue_best['ARO_name'].notna() &
    rescue_best['gene_family'].str.contains('beta-lactamase', case=False, na=False)
].copy()
print(f'Rescue rows that are beta-lactamases: {len(rescue_bl):,}')

all_current_ids = set(test5['query_id'])
new_rows = rescue_bl[~rescue_bl['query_id'].isin(all_current_ids)].copy()
print(f'New query_ids not yet in test5: {len(new_rows):,}')

if not new_rows.empty:
    locus = new_rows['query_id'].str.extract(r'^(.+?)_(\d+)_(\d+)')
    new_rows['plasmid'] = locus[0]
    new_rows['start']   = locus[1].astype(float)
    new_rows['stop']    = locus[2].astype(float)

    source_lookup = source_df.set_index('query_id')[['source', 'filename']].to_dict('index')
    lookup = new_rows['query_id'].map(source_lookup)
    new_rows['source']   = lookup.map(lambda x: x['source']   if isinstance(x, dict) else None)
    new_rows['filename'] = lookup.map(lambda x: x['filename'] if isinstance(x, dict) else None)

    def make_rough(aro_name):
        if not isinstance(aro_name, str):
            return None
        parts = aro_name.split('-')
        if len(parts) > 1 and re.match(r'^\d+[a-z]?$', parts[-1]):
            rough = '-'.join(parts[:-1])
        else:
            rough = aro_name
        return rough if rough else aro_name

    new_rows['gene_family_rough'] = new_rows['ARO_name'].apply(make_rough)
    new_rows['gene_family_true']  = new_rows['gene_family_rough'].apply(extract_gene_family)

    new_rows = new_rows[new_rows['gene_family_true'].apply(lambda x: isinstance(x, str))]
    print(f'New rows after family resolution: {len(new_rows):,}')
    print(new_rows[['query_id', 'ARO_name', 'gene_family_rough', 'gene_family_true']].head(10))

    for c in test5.columns:
        if c not in new_rows.columns:
            new_rows[c] = None
    new_rows = new_rows[test5.columns]

    test5 = pd.concat([test5, new_rows], ignore_index=True)
    test5 = test5.drop_duplicates(subset='query_id', keep='first')
else:
    print('No new beta-lactamase rows to add from rescue.')





test5.to_csv('beta_lactamases_geneandfamily_mapped.csv', index=False)
print(f'\nFinal rows: {len(test5):,}')
print(f'Unique gene_family_true: {test5["gene_family_true"].nunique()}')
print(f'Unique gene_probably:    {test5["gene_probably"].nunique() if "gene_probably" in test5.columns else "N/A"}')
print(f'NaN gene_family_rough:   {test5["gene_family_rough"].isna().sum():,}')
print(f'NaN gene_family_true:    {test5["gene_family_true"].isna().sum():,}')





#NOTE THAT THE _OLD VERSION OF MAPPED BETA LACTAMASE GENE NAMES CONSIDERS THOSE WITH NO GOOD BLAST EVIDENCE
#BUT GENE NAME FROM THE ORIGINAL PLSDB FASTAS, THE NEW VERSION DOESN'T TO BE MEAN







































































#
#
#
#RESCUE_TSV = Path('blast_results/rescue.tsv')
#COLS = ['query_id','subject_id','pident','length','qlen','slen',
#        'evalue','bitscore','qcovs']
#
##if RESCUE_TSV.exists():
#rescue_raw = pd.read_csv(RESCUE_TSV, sep='\t', header=None, names=COLS)
#print(f'\nRescue blast hits (raw): {len(rescue_raw):,}')
## apply relaxed thresholds
#rescue_filtered = rescue_raw[
#    (rescue_raw['pident'] >= 50.0) &
#    (rescue_raw['qcovs']  >= 50.0)
#]
#print(f'After 50/50 filter: {len(rescue_filtered):,}')
#rescue_best = (rescue_filtered
#               .sort_values('bitscore', ascending=False)
#               .groupby('query_id', as_index=False).first())
## parse subject_id -> ARO_accession | ARO_name
#
## build a lookup from CARD model_id -> ARO info
## card.json structure: top-level keys are ARO numeric IDs,
## each entry has 'model_id' and nested model_sequences with sequence IDs
#
#
#
#model_seq_to_aro = {}
#for aro_id, entry in card.items():
#    if not isinstance(entry, dict):
#        continue
#    model_id = entry.get('model_id')
#    if model_id is None:
#        continue
#    for seq_id in entry.get('model_sequences', {}).get('sequence', {}).keys():
#        # subject_id in blast = f"{model_id}_{seq_id}"
#        key = f'{model_id}_{seq_id}'
#        model_seq_to_aro[key] = {
#            'ARO_accession': entry.get('ARO_accession', ''),
#            'ARO_name':      entry.get('ARO_name', ''),
#            'gene_family':   '; '.join(
#                v.get('category_aro_name','') for v in entry.get('ARO_category',{}).values()
#                if v.get('category_aro_class_name') == 'AMR Gene Family'),
#            'drug_class':    '; '.join(
#                v.get('category_aro_name','') for v in entry.get('ARO_category',{}).values()
#                if v.get('category_aro_class_name') == 'Drug Class'),
#        }
#
#print(f'model_seq_to_aro entries: {len(model_seq_to_aro):,}')
#
#print(list(model_seq_to_aro.items())[:3])
#
#
#rescue_best['ARO_accession'] = rescue_best['subject_id'].map(
#    lambda x: model_seq_to_aro.get(x, {}).get('ARO_accession'))
#rescue_best['ARO_name'] = rescue_best['subject_id'].map(
#    lambda x: model_seq_to_aro.get(x, {}).get('ARO_name'))
#rescue_best['gene_family'] = rescue_best['subject_id'].map(
#    lambda x: model_seq_to_aro.get(x, {}).get('gene_family'))
#rescue_best['drug_class'] = rescue_best['subject_id'].map(
#    lambda x: model_seq_to_aro.get(x, {}).get('drug_class'))
#
#
#
#newly_mapped = rescue_best['ARO_name'].notna().sum()
#print(f'Newly mappable from rescue: {newly_mapped:,}')
#
#
#rescue_ids = set(rescue_best['query_id'])
#rescue_lookup = rescue_best.set_index('query_id')
#
#
#source_lookup = source_df.set_index('query_id')[['source','filename']].to_dict('index')
#all_current_ids = set(test5['query_id'])
#
#new_rows = rescue_best[
#    rescue_best['query_id'].isin(rescue_ids - all_current_ids) &
#    rescue_best['ARO_name'].notna()
#].copy()
#
#locus = new_rows['query_id'].str.extract(r'^(.+?)_(\d+)_(\d+)')
#new_rows['plasmid'] = locus[0]
#new_rows['start']   = locus[1].astype(float)
#new_rows['stop']    = locus[2].astype(float)
#lookup = new_rows['query_id'].map(source_lookup)
#new_rows['source'] = lookup.map(lambda x: x['source'] if isinstance(x, dict) else None)
#new_rows['filename'] = lookup.map(lambda x: x['filename'] if isinstance(x, dict) else None)
#new_rows['gene_family_rough'] = new_rows['ARO_name'].apply(
#    lambda x: '-'.join(str(x).split('-')[:-1]) if isinstance(x, str) else None
#)
#new_rows['gene_family_true'] = new_rows['gene_family_rough'].apply(extract_gene_family)
#for c in test5.columns:
#    if c not in new_rows.columns:
#        new_rows[c] = None
#new_rows = new_rows[test5.columns]
#test6 = pd.concat([test5, new_rows], ignore_index=True)
#test6 = test6.drop_duplicates(subset='query_id', keep='first')
#print(f'Added {len(new_rows):,} entirely new rows from rescue')
#
#
#test6.to_csv('beta_lactamases_geneandfamily_mapped_full.csv', index=False)
#print(f'Updated test5: {len(test5):,} rows, '
#      f'{test5["gene_family_true"].nunique()} families')
#print(f'Still unmapped: {test5["ARO_name"].isna().sum():,}')
#
#
##else:
##    print(f'Rescue TSV not found at {RESCUE_TSV} — submit rescue BLAST job first:')
##    print('  sbatch --wrap="blastp -query all_card/unmapped_queries_prot.fa \\')
##    print('      -db localDB/protein.db -out blast_results/rescue.tsv \\')
##    print('      -outfmt \'6 qseqid sseqid pident length qlen slen evalue bitscore qcovs\' \\')
##    print('      -evalue 1e-5 -num_threads 36 -max_target_seqs 5" \\')