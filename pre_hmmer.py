

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
from Bio.Seq import Seq
from Bio import Entrez, SeqIO
from io import StringIO

cur_dir = os.getcwd()


plasmids_files = os.listdir(Path(os.path.join(cur_dir, 'plasmids')))
plasmids_accn = [x.replace('.fa', '') for x in plasmids_files]

proteins = pd.read_csv('protein.csv')



with open('proteins.csv', 'r') as f:
    with open('relevent_proteins.csv', 'w') as g:
        i = 0
        for line in f:
            nuccore = line.split(',')[0]
            if nuccore in plasmids_accn or i == 0:
                g.write(f'{line}')
            i += 1    
g.close()



proteins = pd.read_csv('relevent_proteins.csv')


test_path = Path(os.path.join(cur_dir,  'plasmids/NZ_CP157106.1.fa'))
with open(test_path, 'r') as f:
    test_seq = f.readlines()[1]



gitline = []

with open('relevent_proteins.csv', 'r') as f:
    for line in f:
        gitline.append(line)           


gittlines = gitline[1:]


temp = gitline[1]
tempp =  temp.split(',')
tempp[2] # prot name
tempp[4] #prot id for cross-reference with proteins.fasta
tempp[6] #transl_table for seqIO
tempp[8] #start for seq
tempp[9] #stop for seq
tempp[10] #strand for seq




with open('hmminput_allplasmid_proteins.fa', 'w') as g:
    ind = 0
    end = len(gittlines)
    for i in gittlines[:]:
        i = re.sub(r'"[^"]*"', lambda m: m.group(0).replace(',', ''), i)
        split_i = i.split(',')
        nuccore, prot_name, prot_id, transl_table, start_pos, stop_pos, strand = split_i[0], split_i[2], split_i[4], split_i[6], split_i[8], split_i[9], split_i[10]
        with open(Path(os.path.join(cur_dir,  f'plasmids/{nuccore}.fa')), 'r') as f:
            plasmid_sequence = f.readlines()[1]
        gene_seq = plasmid_sequence[int(start_pos):int(stop_pos)]
        if strand == '-1.0\n':
            gene_seq = str(Seq(gene_seq).reverse_complement())
        prot_seq = str(Seq(gene_seq).translate(table=int(transl_table)))
        if len(gene_seq) % 3 != 0 or '*' in prot_seq[:-1]:
            g.write(f'>{nuccore}_{start_pos}_{stop_pos}_JUNK {prot_id} {prot_name}\n{prot_seq}\n')
        else:
            g.write(f'>{nuccore}_{start_pos}_{stop_pos} {prot_id} {prot_name}\n{prot_seq}\n')
        ind += 1
        print(f'{ind} out of {end}')

g.close()





with open('hmminput_allplasmid_proteins.fa') as f:
    with open('hmminput_allplasmid_proteins_nojunk.fa', 'w') as g:
        write_record = False
        for line in f:
            if line.startswith('>'):
                write_record = '_JUNK' not in line
            if write_record:
                g.write(line)


g.close()
           
 
 
cases = []
with open('hmminput_allplasmid_proteins_nojunk.fa') as f:
    i = 0
    for line in f:
        if not line.startswith('>'):
            if '*' in line.replace('\n', '')[:-1]:
                cases.append(i)
        i += 1


hit_list = [x-1 for x in cases]
hit_list_all = hit_list + cases




with open('hmminput_allplasmid_proteins_nojunk.fa') as f:
    with open('hmminput_allplasmid_proteins_clean.fa', 'w') as g:
        i = 0
        for line in f:
            if i not in hit_list_all:
                g.write(line)
            i += 1

g.close()
        




with open('hmminput_allplasmid_proteins_strandorientation.fa', 'w') as g:
    ind = 0
    end = len(gittlines)
    for i in gittlines[:]:
        i = re.sub(r'"[^"]*"', lambda m: m.group(0).replace(',', ''), i)
        split_i = i.split(',')
        nuccore, prot_name, prot_id, transl_table, start_pos, stop_pos, strand = split_i[0], split_i[2], split_i[4], split_i[6], split_i[8], split_i[9], split_i[10]
        if strand == '-1.0\n':
            g.write(f'>{nuccore}_{start_pos}_{stop_pos}\nminus\n')
        else:
            g.write(f'>{nuccore}_{start_pos}_{stop_pos}\nplus\n')
        ind += 1
        print(f'{ind} out of {end}')
g.close()




#check against proteins from fasta...







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

would have to read line by line and filter for in plasmids