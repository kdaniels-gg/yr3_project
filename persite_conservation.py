import os 
import re
import numpy as np
import pandas as pd
from pathlib import Path
import time
import math
from collections import Counter 
import glob
from Bio import AlignIO
import subprocess

base_dir = Path('/home/kd541/kdan2/kdan/data/')
cur_dir = os.getcwd()

clusters = pd.read_csv('typing.tsv', sep='\t')



all_communities = list(set(clusters['type'].tolist()))
cluster_to_plasmids = {k:[] for k in all_communities}

for i, row in clusters.iterrows():
    fam = row['type']
    accn = row['plasmid']
    cluster_to_plasmids[fam].append(accn)
    sub_fam = fam.split('_')[-1]
    if sub_fam != '0': 
        print(f'{fam}, {accn}')


#get communities with multiple subcommunities
#all_superfam = list(set([f'{x.split('_')[0]}_{x.split('_')[1]}' for x in all_communities]))


super_communities = list(set([f'{x.split('_')[0]}_{x.split('_')[1]}' for x in all_communities]))


plasmids_in_super_communities = {k: v for k, v in cluster_to_plasmids.items() if any(f'{x}_' in k for x in super_communities)}


pooled_super_community_plasmids = {k:[] for k in super_communities}

for k, v in plasmids_in_super_communities.items():
    super_k = f'{k.split('_')[0]}_{k.split('_')[1]}'
    pooled_super_community_plasmids[super_k] += [v]



super_community_sizes = {k:len(v) for k,v in pooled_super_community_plasmids.items()}
super_community_df = pd.DataFrame.from_dict({'super_community':list(super_community_sizes.keys()), 'number_plasmids':list(super_community_sizes.values())})

super_community_df.to_csv(os.path.join(base_dir, f'super_community_sizes.csv'), index=False)



def calculate_shannon_entropy(alignment, column_index):
    column = alignment[:, column_index]
    # would add not in ['N', '-'] if you wanted to discar/not account for gaps
    bases = [c.upper() for c in column if c not in ['N']]
    total_bases = len(bases)
    if total_bases == 0:
        return 0
    counts = Counter(bases)
    entropy = 0
    for base, count in counts.items():
        p_i = count / total_bases
        entropy -= p_i * math.log2(p_i)
    return entropy


def get_conservation_scores(fasta_file):
    #get conservation bit score i.e., max - entropy 
    #max = log2(5) because states = 4 nucleotide bases + gap
    try:
        alignment = AlignIO.read(fasta_file, 'fasta')
    except Exception as e:
        print(f'problem {fasta_file} {e}')
        return []
    conservation_scores = []
    for i in range(alignment.get_alignment_length()):
        entropy = calculate_shannon_entropy(alignment, i)
        cons_score = math.log2(5) - entropy
        conservation_scores.append(cons_score)
    return conservation_scores


MSA_path = Path(os.path.join(base_dir, 'MSA_ARG_fastas'))
os.makedirs(MSA_path, exist_ok = True)

conservation_path = os.path.join(base_dir, 'conservation_scores')
os.makedirs(conservation_path, exist_ok=True)



beta_lactam_gene_fastas = os.listdir(os.path.join(base_dir, 'fastas'))
for k, v in pooled_super_community_plasmids.items():
    MSA_supercommunity_path = Path(os.path.join(MSA_path, f'{k}'))
    os.makedirs(MSA_supercommunity_path, exist_ok = True)
    conservation_supercommunity_path = Path(os.path.join(conservation_path, f'{k}'))
    os.makedirs(conservation_supercommunity_path, exist_ok = True)
    plasmids = [x for xs in v for x in xs]
    beta_lactam_gene_files = [x for x in beta_lactam_gene_fastas if x.split('_')[0] in plasmids]
    plasmids_to_beta_lactam_genes = {k:[] for k in plasmids}
    plasmids_to_beta_lactam_gene_sequences = {k:[] for k in plasmids}
    for file in beta_lactam_gene_files:
        gene_name = file.split('_')[-1].replace('.fa', '')
        plasmid = file.split('_')[0]
        plasmids_to_beta_lactam_genes[plasmid] += [gene_name]
        with open(os.path.join(base_dir, f'fastas/{file}'), 'r') as f:
            text_lines = f.readlines() 
            gene_sequence = text_lines[1]
            plasmids_to_beta_lactam_gene_sequences[plasmid] += [gene_sequence]
    plasmid_beta_lactam_genes = [x for xs in list(plasmids_to_beta_lactam_genes.values()) for x in xs]
    plasmid_beta_lactam_gene_sequences = [x for xs in list(plasmids_to_beta_lactam_gene_sequences.values()) for x in xs]
    beta_lactam_genes_to_sequences = {k:[] for k in plasmid_beta_lactam_genes}
    for x in plasmid_beta_lactam_genes:
        i = plasmid_beta_lactam_genes.index(x)
        gene_sequence = plasmid_beta_lactam_gene_sequences[i]
        beta_lactam_genes_to_sequences[x] += [gene_sequence]
    for k1, v1 in beta_lactam_genes_to_sequences.items():
        with open(os.path.join(MSA_supercommunity_path, f'{k1}.fa'), 'w') as f:
            i = 0
            for x in v1:
                f.write(f'>{k1}_{i}\n{x}\n')
                i+=1
    f.close()
    output_dir = Path(os.path.join(base_dir, f'mafft_outputs/{k}'))
    output_dir.mkdir(parents=True, exist_ok=True)
    for fasta_file in MSA_supercommunity_path.glob('*.fa'):
        basename = fasta_file.stem 
        output_file = output_dir / f'{basename}_aligned.fasta'
        cmd = ['mafft', '--auto', '--thread', '-1', str(fasta_file)]
        try:
            with open(output_file, 'w') as out_f:
                subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            print(f'{basename}: {e}')
    msa_files = [os.path.join(output_dir, f'{x}') for x in os.listdir(output_dir)]    
    msa_file_geneconservation_persite = {}
    for msa_file in msa_files:
        scores = get_conservation_scores(msa_file)
        msa_file_geneconservation_persite[msa_file] = scores
    for eh, ey in msa_file_geneconservation_persite.items():
        if len(list(set(ey))) > 1:
            print(eh)
    for file, conservation_values in msa_file_geneconservation_persite.items():
        conservation_file_path = os.path.join(conservation_supercommunity_path, f'{file.split('/')[-1].split('_')[0]}.txt')
        conservation_text = ','.join([str(x) for x in conservation_values])
        with open(conservation_file_path, 'w') as f:
            f.write(conservation_text)
        f.close()




for i in range(len(pooled_super_community_plasmids)):
    print(f'{i}, {list(pooled_super_community_plasmids.keys())[i]}, {list(pooled_super_community_plasmids.values())[i]}\n')

k = list(pooled_super_community_plasmids.keys())[4]
v = list(pooled_super_community_plasmids.values())[4]





#Then return the most highly conserved residues or something idk, or warn if overlap with 
#editing and upper quartile conserved residues - write folder with files ARGNAME_conserved_sites.txt with list of the sites >= q3











within a cluster - which ARGs are the most widespread - 
get the plasmids, for each add instances of ARGs, then simply use count
















#over_that = 0
#under_that = 0
#
#for k, v in results.items():
#    q9 = np.quantile(v, 0.9)
#    q1 = np.quantile(v, 0.1)
#    i = 0
#    for x in v:
#        if x >= q9:
#            over_that += 1
#            file_to_conserved_sites[k] += [i]
#        if x <= q1:
#            under_that += 1
#            file_to_nonconserved_sites[k] += [i]
#        i += 1
#



#mkdir -p mafft_outputs
#for file in MSA_ARG_fastas/*.fa; do
#    basename=$(basename "$file" .fasta)
#    mafft --auto --thread -1 "$file" > "mafft_outputs/${basename}_aligned.fasta"
#    echo "Finished aligning $basename"
#done
#


#header_output_file = os.path.join(base_dir, 'headers_only.txt')
#
#
#PLSDB_fasta = Path(os.path.join(base_dir, 'PLSDB_sequences.fasta'))
#header_output_file = 'headers_only.txt'
#
#with open(header_output_file, "w") as outfile:
#    subprocess.run(["grep", "^>", PLSDB_fasta], stdout=outfile)
#
#
#headers = []
#with open(header_output_file, 'r') as f:
#    for line in f:
#        header = line.split(' ')[0]
#        headers.append(header)
#
#headers = [x.replace('>', '') for x in headers]
#
#
#ye = clusters['plasmid'].tolist()
#interest = [x for x in ye if x not in headers]
#
#only 8 of the test dataset are in the current rendition of 
#PLSDB....