import json
import re
import pandas as pd
from pathlib import Path
from Bio.Seq import Seq
import subprocess
import os
import polars as pl
import json
import math
import re
import subprocess
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
import os
from collections import Counter

CARD_DIR = Path('card_gof_reference')
BLFASTA_DIR    = Path('beta_lactam_fastas')
BLPFAM_FA      = Path('pfam_betalactamase_genesequences.fa')



beta_fastas = os.listdir(BLFASTA_DIR)
beta_fromfasta_ids = [f'{'_'.join(x.split('_')[:-1])}' for x in beta_fastas]

#make beta-lactamase fastas from pfam


data_dir = Path(os.path.join(os.getcwd(),'plasmid_motif_network/intermediate'))
files = sorted(data_dir.glob("parsed_selected_nonoverlap_*.parquet"))

df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()
df_merged = df_merged.with_columns(
    pl.col('strand').cast(pl.Int32).alias('strand')
)

beta_pfam_df1 = df_merged.filter(pl.col('target_name').str.contains('lactamase'))
beta_frompfam_ids1 = list(set(list(beta_pfam_df1['query_name'])))
betadesc_frompfam_ids1 = list(set(list(beta_pfam_df1['target_name'])))
len(beta_frompfam_ids1)

beta_pfam_df2 = df_merged.filter(pl.col('target_name').str.contains('Beta-lactamase'))
beta_frompfam_ids2 = list(set(list(beta_pfam_df2['query_name'])))
betadesc_frompfam_ids2 = list(set(list(beta_pfam_df2['target_name'])))
len(beta_frompfam_ids2)

domain_df = pd.read_csv('Pfam-A.clans.tsv', sep='\t', header=None)
domain_dict = dict(zip(domain_df[3].tolist(), domain_df[1].tolist()))
beta_clan = 'CL0013'

df_bl_in_clan = df_merged.filter(
    pl.col('target_name').is_in(list(domain_dict.keys())) & 
    pl.col('target_name').map_elements(lambda x: domain_dict.get(x) == beta_clan, return_dtype=pl.Boolean)
)




beta_pfam_df3 = df_merged.filter(pl.col('target_name').str.contains('OXA'))
beta_frompfam_ids3 = list(set(list(beta_pfam_df3['query_name'])))
betadesc_frompfam_ids3 = list(set(list(beta_pfam_df3['target_name'])))
len(beta_frompfam_ids3)

clan_blids = list(set(list(df_bl_in_clan['query_name'])))
all_other_bl = list(set(beta_frompfam_ids1 + beta_frompfam_ids2))
missed_bl = [x for x in all_other_bl if x not in clan_blids]

comp_bl = list(set(missed_bl + clan_blids))

beta_pfam_df = df_merged.filter(pl.col('query_name').is_in(comp_bl))
beta_frompfam_ids = list(beta_pfam_df['query_name'])

plasmids_path = Path(os.path.join(os.getcwd(), 'plasmids'))
pfamhits_ids = []
pfamhits_orientation = []

with open('hmminput_allplasmid_proteins_strandorientation.fa', 'r') as g:
    for line in g:
        if line.startswith('>'):
            pfamhits_ids.append(line[1:-1])
        else:
            pfamhits_orientation.append(line[:-1])


pfamhit_id_to_orientation = dict(zip(pfamhits_ids, pfamhits_orientation))

with open('pfam_betalactamase_genesequences.fa', 'w') as g:
    for x in beta_frompfam_ids:
        plasmid_id = '_'.join(x.split('_')[:-2])
        start = int(x.split('_')[-2])
        stop = int(x.split('_')[-1])
        plasmid_fasta_path = os.path.join(plasmids_path, f'{plasmid_id}.fa')
        with open(plasmid_fasta_path, 'r') as f:
            sequence = f.readlines()[1]
            geneseq = sequence[start:stop]
            if pfamhit_id_to_orientation.get(x) == 'minus':
                geneseq = str(Seq(geneseq).reverse_complement())
        g.write(f'>{x}\n{geneseq}\n')


g.close()






#combine beta-lactmases from pfam and plsdb into multifasta for nuc and prot
#NB - at some point all_query_seuqences.fa and others were made to be literally all possible proteins as per pfam
#and the fastas directory, but I found that BLAST doesn't augment the data at all when using this approach.

query_source = {}
combined_fa = CARD_DIR / 'all_query_sequences.fa'

with open(combined_fa, 'w') as out:
    for fa in BLFASTA_DIR.glob('*.fa'):
        filename = fa.name
        with open(fa, 'r') as f:
            last_line = ""
            for line in f:
                out.write(line)
                if line.startswith('>'):
                    query_id = line[1:].split()[0]
                    query_source[query_id] = {'source': 'fasta_dir', 'filename': filename}
                last_line = line
            if last_line and not last_line.endswith('\n'):
                out.write('\n')
    pfam_filename = str(BLPFAM_FA)
    with open(BLPFAM_FA, 'r') as f:
        last_line = ""
        for line in f:
            out.write(line)
            if line.startswith('>'):
                query_id = line[1:].split()[0]
                query_source[query_id] = {'source': 'pfam', 'filename': pfam_filename}
            last_line = line
        if last_line and not last_line.endswith('\n'):
            out.write('\n')


out.close()
print(f'total query sequences: {len(query_source)}')


err_to_discard = []
for fa in BLFASTA_DIR.glob('*.fa'):
    filename = fa.name
    with open(fa, 'r') as f:
        last_line = ""
        for line in f:
            if line.startswith('>'):
                text = line[1:-1]
                genenamein = re.match(r'^(.+?)_(\d+)_(\d+)_', text)
                if not genenamein:
                    err_to_discard.append(text)


#clean up nuc fasta
kept, dropped = [], []
current_id = None
current_seq = []
with open(combined_fa) as f:
    for line in f:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith('>'):
            if current_id is not None:
                seq = ''.join(current_seq).strip()
                if seq and len(seq) >= 2 and current_id not in err_to_discard:
                    kept.append((current_id, seq))
                else:
                    dropped.append((current_id, len(seq)))
            current_id = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line)
    if current_id is not None:
        seq = ''.join(current_seq).strip()
        if seq and len(seq) >= 2 and current_id not in err_to_discard:
            kept.append((current_id, seq))
        else:
            dropped.append((current_id, len(seq)))



with open(combined_fa, 'w') as out:
    for qid, seq in kept:
        out.write(f'>{qid}\n{seq}\n')


out.close()



all_headers = []
with open(combined_fa, 'r') as f:
    for line in f:
        if line.startswith('>'):
            all_headers.append(line)


all_headers = [x[1:-1] for x in all_headers]
len(all_headers)
len(list(set(all_headers)))


bug1 = Counter(all_headers)
bugged = [x for x in all_headers if bug1.get(x) > 1]

seen = set()
out_fa = Path('card_gof_reference/all_query_sequences_dedup.fa')
with open(combined_fa, 'r') as fin, open(out_fa, 'w') as fout:
    keep = False
    for line in fin:
        if line.startswith('>'):
            header = line[1:].strip()
            if header not in seen:
                seen.add(header)
                keep = True
                fout.write(line)
            else:
                keep = False
        else:
            if keep:
                fout.write(line)


if os.path.exists(combined_fa):
  os.remove(combined_fa)


combined_fa = CARD_DIR / 'all_query_sequences.fa'
os.rename(out_fa, combined_fa)



combined_fa_prot = CARD_DIR / 'all_query_sequences_prot.fa'

error = 0
errors_info = {}

with open(combined_fa_prot, 'w') as out:
    for r in SeqIO.parse(combined_fa, 'fasta'):
        if len(str(r.seq))%3 == 0:
            prot = Seq(str(r.seq)).translate(to_stop=True)
            if len(str(prot)) >= 1:
                out.write(f'>{r.id}\n{prot}\n')
        else:
            error +=1 
            errors_info[r.id] = {'protein':str(Seq(str(r.seq)).translate(to_stop=False)), 'nucleotide':str(r.seq)}

out.close()



#SETUP AMRFINDERPLUS DATABASE
#source "$HOME/miniforge3/etc/profile.d/conda.sh"
#conda create -n amrfinder -c bioconda -c conda-forge ncbi-amrfinderplus
#conda activate amrfinder
#amrfinder --update --database ./amrfinderplus_db
#conda install -c bioconda polars
#conda install -c bioconda pandas
#conda install -c bioconda biopython
#conda install -c bioconda blast
#conda install -c bioconda tdqm
#conda install -c bioconda mafft

# AMRFINDER FIRST RUN: ONLY CHECK THE BETA-LACTAMASES AS IDENTIFIED BY PLSDB AND PFAM. COULD DO ALL QUERIES IN HPC LATER IF DESIRED

###################OVER TO CLUSTER
#RUN AMRFINDERPLUS
#-p / --protein   : protein input — activates both BLAST and HMM searches
#--print_node     : adds HMM node column, useful for partial/HMM-only hits
#--ident_min 0.5  : report BLAST hits down to 50% identity (curated per-gene
#                   cutoffs still apply internally; this just widens reporting)
#                   change to -1 to use curated cutoffs only (stricter)


##pre-process into chunks
#import math
#from pathlib import Path
#from Bio import SeqIO
#
#PROT_FA = Path("card_gof_reference/all_query_sequences_prot.fa")
#OUTDIR  = Path("amrfinder_batches")
#
#N_BATCH = 30
#
#OUTDIR.mkdir(exist_ok=True)
#
#records = list(SeqIO.parse(PROT_FA, "fasta"))
#n = len(records)
#
#batch_size = math.ceil(n / N_BATCH)
#
#print(f"Total sequences: {n}")
#print(f"Batch size: {batch_size}")
#
#for i in range(N_BATCH):
#
#    start = i * batch_size
#    end   = start + batch_size
#
#    subset = records[start:end]
#
#    if not subset:
#        continue
#
#    outfile = OUTDIR / f"batch_{i:02d}.fa"
#
#    with open(outfile, "w") as out:
#        SeqIO.write(subset, out, "fasta")
#
#    print(outfile, len(subset))
#
###run_amrfinder_batch.py
#import subprocess
#from pathlib import Path
#import sys
#
#BATCH_FA = Path(sys.argv[1])
#
#AMR_DB  = Path("amrfinderplus_db/2026-01-21.1")
#OUTDIR  = Path("amrfinder_results")
#
#THREADS = 8
#
#OUTDIR.mkdir(exist_ok=True)
#
#out_file = OUTDIR / f"{BATCH_FA.stem}_amrfinder.tsv"
#
#cmd = [
#    "amrfinder",
#    "--protein", str(BATCH_FA),
#    "--database", str(AMR_DB),
#    "--output", str(out_file),
#    "--threads", str(THREADS),
#    "--print_node",
#    "--ident_min", "0.5",
#]
#
#print("Running:", " ".join(cmd))
#
#result = subprocess.run(cmd, capture_output=True, text=True)
#
#print(result.stdout)
#
#if result.returncode != 0:
#    print(result.stderr)
#    raise RuntimeError("AMRFinder failed")
#
#print("Finished:", out_file)
#
#
###slurm
##!/bin/bash
##SBATCH --job-name=amr_run
##SBATCH --account=MICKLEM-SL3-CPU
##SBATCH --partition=icelake
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=8
##SBATCH --time=12:00:00
##SBATCH --output=logs/amr_%A_%a.out
##SBATCH --error=logs/amr_%A_%a.err
##SBATCH --array=1-29
#
#
#. /etc/profile.d/modules.sh
#module purge
#module load rhel7/default-peta4
#source "$HOME/miniforge3/etc/profile.d/conda.sh"
#conda activate amrfinder
#
#
#BATCH_DIR=amrfinder_batches
#
#BATCH=$(printf "%s/batch_%02d.fa" $BATCH_DIR $SLURM_ARRAY_TASK_ID)
#
#echo "Processing $BATCH"
#
#python run_amrfinder_batch.py $BATCH



###merge and extract, local

import pandas as pd
from pathlib import Path
import re


RESULT_DIR = Path("amrfinder_results")

files = sorted(RESULT_DIR.glob("batch_*_amrfinder.tsv"))

dfs = []

for f in files:
    df = pd.read_csv(f, sep="\t")
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)

merged.to_csv("amrfinder_all_hits.tsv", sep="\t", index=False)



merged2 = merged.copy(deep=True)

pat = re.compile(r"^(bla)(.*)?$")
matches = [pat.match(x) for x in merged['Element symbol'].tolist()]
test_values = [m.group(2) if m else None for m in matches]
merged2['gene_name'] = np.where([v is None or v == '' for v in test_values],merged2['Hierarchy node'], test_values)


FAMILY_EXCEPTIONS = {
    'bla-A':           'A',
    'bla-A_carba':     'A',
    'bla-A_firm':      'A',
    'bla-A2':          'A',
    'bla-B1':          'B1',
    'bla-C':           'ampC',
    'bla1':            'bla1',
    'blaZ':            'Z',
    'blaL2':           'L2',
    'blaIII':          'III',
    'blaPC1':          'PC1',
    'blaI_of_Z':       'Z',
    'blaSHV-LEN':      'SHV',
    'blaPEN-B':        'PEN',
    'blaPEN-J':        'PEN',
    'blaPEN-bcc':      'PEN',
    'blaR39':          'R',
    'blaR1':           'R',
    'blaR1-2':         'R',
    'blaR1_gen':       'R',
    'CMY2-MIR-ACT-EC': 'CMY',
    'mecA':            'mecA',
    'mecB':            'mecB',
    'mecR1':           'mecR1',
    'mecI_of_mecA':    'mecA',
    'cdiA':            'cdiA',
    'cfiA2':           'cfiA',
    'cfxA_fam':        'cfxA',
    'pbp2m':           'pbp2m',
}

def extract_gene_family(node: str) -> str:
    if node in FAMILY_EXCEPTIONS:
        return FAMILY_EXCEPTIONS[node]
    if node.startswith('bla'):
        suffix = node[3:]
        m = re.match(r'^([A-Za-z]+-[A-Za-z]+|[A-Za-z]+)', suffix)
        if m:
            return m.group(1)
    return node

merged2['gene_family'] = merged2['Hierarchy node'].apply(extract_gene_family)

#list(set(merged2['gene_family'].tolist()))




merged2.to_csv('amrfindermapped_beta_lactamases.csv', index=False)

