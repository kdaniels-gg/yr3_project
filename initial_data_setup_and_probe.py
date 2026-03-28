import os 
import re
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.Seq import Seq
from Bio import Entrez, SeqIO
from io import StringIO
import time
import math

Entrez.email = 'dantemeeping@gmail.com'
Entrez.api_key = 'a50346ef7920719688d2fbd2938fd5bc3e08'

nuccore = pd.read_csv('nuccore.csv')
nuccore.head()

#Note - indexing here is odd, but is basically -1 shifted for python indexing, then extended 3bp to include the stop
# Some of the genes don't start ATG/GTG - not sure why, always end in stop

amr = pd.read_csv('amr.tsv', sep='\t')
amr.head()

AMR_ACC = amr['NUCCORE_ACC'][:].tolist()
AMR_ACC = list(set(AMR_ACC))

#Get plasmid sequences for the plasmids that contain AMR genes
def chunk(unchunked, chunk_size):
    for i in range(0, len(unchunked), chunk_size):
        yield unchunked[i:i+chunk_size]


def get_NCBIseq(accns: list[str]) -> dict[str, str]:
    time_start = time.time()
    chunk_size = 200
    return_sequences = {}
    num_chunks = math.ceil(len(accns)/chunk_size)
    i=0
    for chunk_bit in chunk(accns, chunk_size):
        with Entrez.efetch(db='nucleotide', id=','.join(chunk_bit), rettype='fasta', retmode='text') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                return_sequences[record.id] = str(record.seq)
        time_chunk = time.time()
        i += 1
        print(f'{time_chunk-time_start} seconds passed, processed {i} chunks out of {num_chunks}')
    return return_sequences




current_path = os.getcwd()

plasmids_path = Path(os.path.join(current_path, 'plasmids'))
os.makedirs(plasmids_path, exist_ok=True)

#Takes around 12 min to run
NUCCORE_TO_SEQ = get_NCBIseq(AMR_ACC)

#Write out plasmid sequences to fasta
with open('plasmids.fa', 'w') as f:
    for k, v in NUCCORE_TO_SEQ.items():
        f.write(f'>{k}\n{v}\n')


f.close()

#Write out plasmid sequences to multiple fastas in plasmids folder
for k, v in NUCCORE_TO_SEQ.items():
    filepath = Path(os.path.join(plasmids_path, f'{k}.fa'))
    with open(filepath, 'w') as f:
        f.write(f'>{k}\n{v}')
        f.close()




#Extract ARG sequences, write out to fasta - should document non-start starters in sep fasta
with open('arg.fa', 'w') as f:
    with open('arg_no_start.fa', 'w') as g:
        end_point = amr.shape[0]
        for i in list(range(0,end_point,1)):
            amr_strand = amr['strand_orientation'][i]
            amr_name = amr['gene_symbol'][i]
            amr_acc = amr['NUCCORE_ACC'][i]
            amr_start = amr['input_gene_start'][i]
            amr_stop = amr['input_gene_stop'][i]
            amr_ncbi = NUCCORE_TO_SEQ.get(amr_acc)
            if amr_ncbi:
                if amr_strand == '+':
                    amr_gene = amr_ncbi[amr_start-1:amr_stop+3]
                    start, stop = amr_start-1, amr_stop+3
                else:
                    amr_gene = amr_ncbi[amr_start-4:amr_stop]
                    start, stop = amr_stop, amr_start-4
                    amr_gene_seq = Seq(amr_gene).reverse_complement()
                    amr_gene = str(amr_gene_seq)
                f.write(f'>{amr_acc}_{start}_{stop}_{amr_name}\n{amr_gene}\n')
                amr_start = amr_gene[:3]
                if amr_start != 'ATG' and amr_start != 'GTG':
                    g.write(f'>{amr_acc}_{start}_{stop}_{amr_name}\n{amr_gene}\n')


f.close()
g.close()


fastas_path = Path(os.path.join(current_path, 'fastas'))
non_start_fastas_path = Path(os.path.join(current_path, 'non_start_fastas'))



os.makedirs(fastas_path, exist_ok=True)
os.makedirs(non_start_fastas_path, exist_ok=True)


gene_name_stripped_to_original = {}

end_point = amr.shape[0]
for i in list(range(0,end_point,1)):
    amr_strand = amr['strand_orientation'][i]
    amr_name = amr['gene_symbol'][i]
    amr_acc = amr['NUCCORE_ACC'][i]
    amr_start = amr['input_gene_start'][i]
    amr_stop = amr['input_gene_stop'][i]
    amr_ncbi = NUCCORE_TO_SEQ.get(amr_acc)
    if amr_ncbi:
        amr_name_stripped = amr_name.replace(" ", "").replace(")", "").replace("(", "").replace("/", "").replace("[", "").replace("]", "")
        gene_name_stripped_to_original[amr_name_stripped] = amr_name



end_point = amr.shape[0]
for i in list(range(0,end_point,1)):
    amr_strand = amr['strand_orientation'][i]
    amr_name = amr['gene_symbol'][i]
    amr_acc = amr['NUCCORE_ACC'][i]
    amr_start = amr['input_gene_start'][i]
    amr_stop = amr['input_gene_stop'][i]
    amr_ncbi = NUCCORE_TO_SEQ.get(amr_acc)
    if amr_ncbi:
        if amr_strand == '+':
            amr_gene = amr_ncbi[amr_start-1:amr_stop+3]
            start, stop = amr_start-1, amr_stop+3
        else:
            amr_gene = amr_ncbi[amr_start-4:amr_stop]
            start, stop = amr_stop, amr_start-4
            amr_gene_seq = Seq(amr_gene).reverse_complement()
            amr_gene = str(amr_gene_seq)
        amr_name_stripped = amr_name.replace(" ", "").replace(")", "").replace("(", "").replace("/", "").replace("[", "").replace("]", "")
        amr_filename = f'{amr_acc}_{start}_{stop}_{amr_name_stripped}.fa'
        amr_path = Path(os.path.join(fastas_path, amr_filename))
        amr_start = amr_gene[:3]
        with open(amr_path, 'w') as f:
            f.write(f'>{amr_acc}_{start}_{stop}_{amr_name}\n{amr_gene}')
            f.close()
        if amr_start != 'ATG' and amr_start != 'GTG':
            amr_non_start_path = Path(os.path.join(non_start_fastas_path, amr_filename))
            with open(amr_non_start_path, 'w') as g:
                g.write(f'>{amr_acc}_{start}_{stop}_{amr_name}\n{amr_gene}')
                g.close()



#Specifically for beta lactams

beta_lactam_fastas_path = Path(os.path.join(current_path, 'beta_lactam_fastas'))
beta_lactam_non_start_fastas_path = Path(os.path.join(current_path, 'beta_lactam_non_start_fastas'))
beta_lactam_plasmids_path = Path(os.path.join(current_path, 'beta_lactam_plasmids'))


os.makedirs(beta_lactam_fastas_path, exist_ok=True)
os.makedirs(beta_lactam_non_start_fastas_path, exist_ok=True)
os.makedirs(beta_lactam_plasmids_path, exist_ok=True)


beta_lactam_agents = ['METHICILLIN', 'CARBAPENEM', 'CARBAPENEM/TANIBORBACTAM', 'CEPHALOSPORIN', 'BETA-LACTAM', 'CEFIDEROCOL/CEPHALOSPORIN']

unknown_amr = amr.loc[amr['antimicrobial_agent'].isnull()]
unknown_gene_names = list(set(unknown_amr['gene_symbol'].tolist()))

beta_lactam_genes = []
with open('beta_lactamase_documented_genes.fa', 'r') as f:
    lines = f.readlines()
    lines = [x for x in lines if '>' in x]
    for line in lines:
        gene = line.strip().split(' ')[-1]
        beta_lactam_genes.append(gene)


beta_lactams_indata = []
beta_lactams_indata += [x for x in unknown_gene_names if x in beta_lactam_genes]
beta_lactams_indata += [x for x in unknown_gene_names if 'Bla' in x]
beta_lactams_indata += [x for x in unknown_gene_names if 'bla' in x]
beta_lactams_indata += [x for x in unknown_gene_names if 'OXA' in x]
beta_lactams_indata += [x for x in unknown_gene_names if 'TEM' in x]
beta_lactams_indata += [x for x in unknown_gene_names if 'NDM' in x]
beta_lactams_indata += [x for x in unknown_gene_names if 'CTX-M' in x]
beta_lactams_indata += [x for x in unknown_gene_names if 'KPC' in x]
beta_lactams_indata += [x for x in unknown_gene_names if 'SHV' in x]
beta_lactams_indata += [x for x in unknown_gene_names if 'IMP' in x]
beta_lactams_indata += [x for x in unknown_gene_names if 'VIM' in x]
beta_lactams_indata += [x for x in unknown_gene_names if 'PER' in x]
beta_lactams_indata += [x for x in unknown_gene_names if 'VEB' in x]
beta_lactams_indata += [x for x in unknown_gene_names if 'GES' in x]
beta_lactams_indata += [x for x in unknown_gene_names if 'CMY' in x]

with open('beta_lactamases_in_data.txt', 'w') as f:
    for x in beta_lactams_indata:
        f.write(f'{x}\n')


f.close()

beta_lactam_amr_total = amr.loc[(amr['antimicrobial_agent'].isin(beta_lactam_agents)) | (amr['gene_symbol'].isin(beta_lactams_indata))]
beta_lactam_amr_total = beta_lactam_amr_total.reset_index()

#write out plasmids

end_point = beta_lactam_amr_total.shape[0]
for i in list(range(0,end_point,1)):
    amr_strand = beta_lactam_amr_total['strand_orientation'][i]
    amr_name = beta_lactam_amr_total['gene_symbol'][i]
    amr_acc = beta_lactam_amr_total['NUCCORE_ACC'][i]
    amr_start = beta_lactam_amr_total['input_gene_start'][i]
    amr_stop = beta_lactam_amr_total['input_gene_stop'][i]
    amr_ncbi = NUCCORE_TO_SEQ.get(amr_acc)
    amr_filename = f'{amr_acc}.fa'
    amr_path = Path(os.path.join(beta_lactam_plasmids_path, amr_filename))
    with open(amr_path, 'w') as f:
        f.write(f'>{amr_acc}\n{amr_ncbi}')
        f.close()



#write out args
end_point = beta_lactam_amr_total.shape[0]
for i in list(range(0,end_point,1)):
    amr_strand = beta_lactam_amr_total['strand_orientation'][i]
    amr_name = beta_lactam_amr_total['gene_symbol'][i]
    amr_acc = beta_lactam_amr_total['NUCCORE_ACC'][i]
    amr_start = beta_lactam_amr_total['input_gene_start'][i]
    amr_stop = beta_lactam_amr_total['input_gene_stop'][i]
    amr_ncbi = NUCCORE_TO_SEQ.get(amr_acc)
    if amr_ncbi:
        if amr_strand == '+':
            amr_gene = amr_ncbi[amr_start-1:amr_stop+3]
            start, stop = amr_start-1, amr_stop+3
        else:
            amr_gene = amr_ncbi[amr_start-4:amr_stop]
            start, stop = amr_stop, amr_start-4
            amr_gene_seq = Seq(amr_gene).reverse_complement()
            amr_gene = str(amr_gene_seq)
        amr_name_stripped = amr_name.replace(" ", "").replace(")", "").replace("(", "").replace("/", "").replace("[", "").replace("]", "")
        amr_filename = f'{amr_acc}_{start}_{stop}_{amr_name_stripped}.fa'
        amr_path = Path(os.path.join(beta_lactam_fastas_path, amr_filename))
        amr_start = amr_gene[:3]
        with open(amr_path, 'w') as f:
            f.write(f'>{amr_acc}_{start}_{stop}_{amr_name}\n{amr_gene}')
            f.close()
        if amr_start != 'ATG' and amr_start != 'GTG':
            amr_non_start_path = Path(os.path.join(beta_lactam_non_start_fastas_path, amr_filename))
            with open(amr_non_start_path, 'w') as g:
                g.write(f'>{amr_acc}_{start}_{stop}_{amr_name}\n{amr_gene}')
                g.close()




#Link ARGs to CARD prevalence 


card = pd.read_csv('card_prevalence.txt', sep='\t')


with open('beta_lactamases_in_data.txt', 'r') as f:
    bls = f.readlines()
    bls = [f.strip() for f in bls]


bls = list(set(bls))

card_genes = list(set(card['Name'].tolist()))

overlap = [x for x in bls if x in card_genes]
non_overlap = [x for x in bls if x not in card_genes]

#209 out of 372 in CARD

bl_card = card.loc[card['Name'].isin(overlap)]
bl_card = bl_card.reset_index()

bl_card.to_csv('beta_lactamase_card_data.csv', index=False)
































##Proof of principle - single case for accession, check consistency between NCBI extracted 
#sequence and database info


acc_from_data = nuccore['NUCCORE_ACC'][:100]
length_from_data = nuccore['NUCCORE_Length'][:100]
acc_to_length = dict(zip(acc_from_data, length_from_data))

#Get nucleotide (DNA) sequence from NCBI given accession (string)
def get_NCBIseqsingle(accn):
    with Entrez.efetch(db='nucleotide', id=str(accn), rettype='fasta', retmode='text') as handle:
        record = SeqIO.read(handle, 'fasta')
    return str(record.seq)

#batch process needed

same_length = 0
for k,v in acc_to_length.items():
    from_ncbi = get_NCBIseqsingle(k)
    if len(from_ncbi) == v:
        same_length +=1 


print(f'{same_length}%')

##################################################





#trial extract ARGs - check non-starts called, check len always divisable by 3
for i in list(range(0,150,1)):
    amr_strand = amr['strand_orientation'][i]
    amr_name = amr['gene_symbol'][i]
    amr_acc = amr['NUCCORE_ACC'][i]
    amr_start = amr['input_gene_start'][i]
    amr_stop = amr['input_gene_stop'][i]
    amr_ncbi = NUCCORE_TO_SEQ.get(amr_acc)
    if amr_ncbi:
        if amr_strand == '+':
            amr_gene = amr_ncbi[amr_start-1:amr_stop+3]
            start, stop = amr_start-1, amr_stop+3
        else:
            amr_gene = amr_ncbi[amr_start-4:amr_stop]
            start, stop = amr_stop, amr_start-4
            amr_gene_seq = Seq(amr_gene).reverse_complement()
            amr_gene = str(amr_gene_seq)
        print(f'>{amr_acc}_{start}_{stop}_{amr_name}\n{amr_gene}\n')
        print(f'gene length:{len(amr_gene)}')
        amr_start = amr_gene[:3]
        if amr_start != 'ATG' and amr_start != 'GTG':
            print(f'**************NON START**************\n >{amr_acc}_{start}_{stop}_{amr_name}\n{amr_gene}\n')
            print(f'gene length:{len(amr_gene)}')












































