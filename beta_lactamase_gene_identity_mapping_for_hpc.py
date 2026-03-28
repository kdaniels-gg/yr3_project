

import math
from pathlib import Path
from Bio import SeqIO

PROT_FA = Path("card_gof_reference/all_queries_final_prot.fa")
OUTDIR  = Path("amrfinder_batches_final")

N_BATCH = 30

OUTDIR.mkdir(exist_ok=True)

records = list(SeqIO.parse(PROT_FA, "fasta"))
n = len(records)

batch_size = math.ceil(n / N_BATCH)

print(f"Total sequences: {n}")
print(f"Batch size: {batch_size}")

for i in range(N_BATCH):

    start = i * batch_size
    end   = start + batch_size

    subset = records[start:end]

    if not subset:
        continue

    outfile = OUTDIR / f"batch_{i:02d}.fa"

    with open(outfile, "w") as out:
        SeqIO.write(subset, out, "fasta")

    print(outfile, len(subset))




#!/bin/bash
#SBATCH --job-name=amr_run
#SBATCH --account=MICKLEM-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/amr_%A_%a.out
#SBATCH --error=logs/amr_%A_%a.err
#SBATCH --array=1-29


. /etc/profile.d/modules.sh
module purge
module load rhel7/default-peta4
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate amrfinder


BATCH_DIR=amrfinder_batches_final

BATCH=$(printf "%s/batch_%02d.fa" $BATCH_DIR $SLURM_ARRAY_TASK_ID)

echo "Processing $BATCH"

python run_amrfinder_batch_final.py $BATCH




import subprocess
from pathlib import Path
import sys

BATCH_FA = Path(sys.argv[1])

AMR_DB  = Path("amrfinderplus_db/2026-01-21.1")
OUTDIR  = Path("amrfinder_results_final")

THREADS = 8

OUTDIR.mkdir(exist_ok=True)

out_file = OUTDIR / f"{BATCH_FA.stem}_amrfinder.tsv"

cmd = [
    "amrfinder",
    "--protein", str(BATCH_FA),
    "--database", str(AMR_DB),
    "--output", str(out_file),
    "--threads", str(THREADS),
    "--print_node",
    "--ident_min", "0.5",
]

print("Running:", " ".join(cmd))

result = subprocess.run(cmd, capture_output=True, text=True)

print(result.stdout)

if result.returncode != 0:
    print(result.stderr)
    raise RuntimeError("AMRFinder failed")

print("Finished:", out_file)




import pandas as pd
from pathlib import Path
import re
import numpy as np

RESULT_DIR = Path("amrfinder_results_final")

files = sorted(RESULT_DIR.glob("batch_*_amrfinder.tsv"))

dfs = []

for f in files:
    df = pd.read_csv(f, sep="\t")
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)

merged.to_csv("amrfinder_all_hits_final.tsv", sep="\t", index=False)



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




merged2.to_csv('amrfindermapped_beta_lactamases_final.csv', index=False)

