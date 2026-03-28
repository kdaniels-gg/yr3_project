####msa_for_betas_array.py
import os
import re
import math
import subprocess
import tempfile
import csv
import logging
import argparse
from collections import Counter
from pathlib import Path
import pandas as pd
from Bio import SeqIO, AlignIO
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ── config ───────────────────────────────────────────────────────────────────
#CSV_PATH        = 'beta_lactamases_geneandfamily_mapped.csv'
CSV_PATH        = 'amrfindermapped_beta_lactamases.csv'
#GROUP_BY        = 'ARO_Name'
GROUP_BY        = 'gene_name'
OUTDIR          = 'beta_lactamase_msa_results/gene_name'
THREADS         = 12          # matches --cpus-per-task in SLURM header
MIN_SEQS        = 2
DEDUPLICATE     = True
SAVE_ALIGNMENTS = True
NUC_FA          = Path('card_gof_reference/all_query_sequences.fa')
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description='Per-task MSA runner for SLURM array jobs.')
    p.add_argument('--task-id', type=int, required=True,
                   help='1-based SLURM_ARRAY_TASK_ID')
    p.add_argument('--n-tasks', type=int, required=True,
                   help='Total number of array tasks (value of --array=1-N)')
    return p.parse_args()


PID_RE = re.compile(r'^(.+?_\d+_\d+)(?:_.+)?$')


def extract_pid(qid):
    m = PID_RE.match(qid)
    return m.group(1) if m else qid


def load_fasta_index(path):
    log.info(f'Indexing sequences from {path} ...')
    index = {}
    for rec in SeqIO.parse(str(path), 'fasta'):
        index[rec.id] = str(rec.seq).upper()
    log.info(f'  Indexed {len(index):,} sequences.')
    return index


def iter_seqs(qids, seq_index):
    seen = set()
    for qid in qids:
        seq = seq_index.get(qid)
        if seq is None:
            seq = seq_index.get(extract_pid(qid))
        if seq is None:
            continue
        if DEDUPLICATE:
            if seq in seen:
                continue
            seen.add(seq)
        yield qid, seq


def mafft_strategy(n):
    if n <= 200:
        return ['--localpair', '--maxiterate', '1000']
    if n <= 2000:
        return ['--genafpair', '--maxiterate', '16']
    if n <= 10000:
        return ['--auto']
    return ['--retree', '1', '--maxiterate', '0']


def column_entropy(col):
    total = len(col)
    H = 0.0
    for count in Counter(col).values():
        p = count / total
        H -= p * math.log2(p)
    return H


def process_group(group_name, qids, seq_index, writer, csvfile):
    """Align one group and write a result row.
    Returns a skip-tuple (name, n, reason) on failure, or None on success.
    """
    tmp_in  = tempfile.NamedTemporaryFile(suffix='.fa',         delete=False,
                                          dir=OUTDIR, mode='w')
    tmp_out = tempfile.NamedTemporaryFile(suffix='_aligned.fa', delete=False,
                                          dir=OUTDIR, mode='w')
    tmp_out.close()

    n = 0
    try:
        with tmp_in as fh:
            for qid, seq in iter_seqs(qids, seq_index):
                fh.write(f'>{qid}\n{seq}\n')
                n += 1
    except Exception as e:
        log.error(f'Failed writing seqs for {group_name}: {e}')
        for p in [tmp_in.name, tmp_out.name]:
            if os.path.exists(p):
                os.remove(p)
        return (group_name, 0, str(e))

    if n < MIN_SEQS:
        log.warning(f'Skipping {group_name!r}: {n} seqs (min={MIN_SEQS})')
        for p in [tmp_in.name, tmp_out.name]:
            if os.path.exists(p):
                os.remove(p)
        return (group_name, n, 'too_few_sequences')

    # single-sequence edge-case (guarded above by MIN_SEQS, kept for safety)
    if n == 1:
        seq_len = len(next(iter_seqs(qids, seq_index))[1])
        writer.writerow([group_name, 1, seq_len, str([qids[0]]), str([0.0] * seq_len)])
        for p in [tmp_in.name, tmp_out.name]:
            if os.path.exists(p):
                os.remove(p)
        return None

    cmd = (['mafft'] + mafft_strategy(n) +
           ['--thread', str(THREADS), '--quiet', tmp_in.name])

    try:
        with open(tmp_out.name, 'w') as out_fh:
            result = subprocess.run(cmd, stdout=out_fh, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode())

        alignment = AlignIO.read(tmp_out.name, 'fasta')
        n_seqs    = len(alignment)
        aln_len   = alignment.get_alignment_length()
        pids      = [extract_pid(rec.id) for rec in alignment]

        entropy_values = [
            round(column_entropy([alignment[j, i] for j in range(n_seqs)]), 6)
            for i in range(aln_len)
        ]
        del alignment

        writer.writerow([group_name, n_seqs, aln_len, str(pids), str(entropy_values)])
        csvfile.flush()

        if SAVE_ALIGNMENTS:
            safe_name = re.sub(r'[^\w\-.]', '_', str(group_name))
            dest = os.path.join(OUTDIR, f'{safe_name}_aligned.fa')
            os.rename(tmp_out.name, dest)

        log.info(f'  {group_name!r}: {n_seqs} seqs, {aln_len} sites')
        return None

    except Exception as e:
        log.error(f'FAILED {group_name!r}: {e}')
        return (group_name, n, str(e))

    finally:
        for p in [tmp_in.name, tmp_out.name]:
            if os.path.exists(p):
                os.remove(p)


def main():
    args    = parse_args()
    task_id = args.task_id   # 1-based
    n_tasks = args.n_tasks

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs('logs',  exist_ok=True)

    assert subprocess.run(['mafft', '--version'], capture_output=True).returncode == 0, \
        'MAFFT not found. Install: conda install -c bioconda mafft'
    log.info('MAFFT found.')

    if not NUC_FA.exists():
        raise FileNotFoundError(f'Nucleotide fasta not found: {NUC_FA}')
    seq_index = load_fasta_index(NUC_FA)

    log.info(f'Loading {CSV_PATH} ...')
    df = pd.read_csv(CSV_PATH, usecols=['query_id', GROUP_BY], low_memory=False)
    df = df[df[GROUP_BY].notna()]
    log.info(f'  {len(df):,} rows, {df[GROUP_BY].nunique()} groups.')

    group_map  = df.groupby(GROUP_BY)['query_id'].apply(list).to_dict()
    del df

    # ── partition groups across tasks using a stride slice ───────────────────
    # task 1 gets indices [0, n_tasks, 2*n_tasks, ...]
    # task 2 gets indices [1, n_tasks+1, ...]  etc.
    # Sorting group names first ensures deterministic, reproducible assignment.
    all_groups = sorted(group_map.keys())
    my_groups  = all_groups[task_id - 1 :: n_tasks]
    log.info(f'Task {task_id}/{n_tasks}: {len(my_groups)} groups '
             f'(of {len(all_groups)} total).')

    # ── per-task output files ─────────────────────────────────────────────────
    results_path = os.path.join(OUTDIR,
                                f'entropy_{GROUP_BY}_task{task_id:04d}.csv')
    skipped = []

    with open(results_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['alignment_name', 'n_seqs', 'alignment_length',
                         'pids', 'entropy_values'])

        for group_name in tqdm(my_groups,
                               desc=f'Task {task_id}/{n_tasks}',
                               unit='group'):
            skip = process_group(group_name, group_map[group_name],
                                 seq_index, writer, csvfile)
            if skip is not None:
                skipped.append(skip)

    log.info(f'Results -> {results_path}')

    if skipped:
        skip_path = os.path.join(OUTDIR,
                                 f'skipped_groups_task{task_id:04d}.csv')
        pd.DataFrame(skipped, columns=['group_name', 'n_seqs', 'reason']) \
          .to_csv(skip_path, index=False)
        log.info(f'Skipped -> {skip_path}')

    log.info(f'Task {task_id} done.')


if __name__ == '__main__':
    main()



##slurm


#!/bin/bash
#SBATCH --job-name=msa_run
#SBATCH --account=MICKLEM-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/msa_%A_%a.out
#SBATCH --error=logs/msa_%A_%a.err
#SBATCH --array=1-50


. /etc/profile.d/modules.sh
module purge
module load rhel7/default-peta4
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate amrfinder_env


python msa_for_betas_array.py \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --n-tasks  50

echo "Task ${SLURM_ARRAY_TASK_ID} complete"




######################################################
###merger, run local
import os
import glob
import pandas as pd

OUTDIR   = 'beta_lactamase_msa_results/gene_name'
GROUP_BY = 'gene_name'


def merge(pattern, out_path, label):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'No files matched: {pattern}')
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f'Merged {len(files)} {label} files -> {out_path}  ({len(df):,} rows)')


merge(
    os.path.join(OUTDIR, f'entropy_{GROUP_BY}_task*.csv'),
    os.path.join(OUTDIR, f'entropy_{GROUP_BY}_ALL.csv'),
    'entropy'
)

merge(
    os.path.join(OUTDIR, 'skipped_groups_task*.csv'),
    os.path.join(OUTDIR, 'skipped_groups_ALL.csv'),
    'skipped'
)

 
#Merged 50 entropy files -> beta_lactamase_msa_results/gene_name/entropy_ARO_name_ALL.csv  (352 rows)