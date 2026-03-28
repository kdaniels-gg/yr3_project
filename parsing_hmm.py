import polars as pl
import sys

domtbl = sys.argv[1]
outpath = sys.argv[2]


def parse_hmm_tblout(file_path):
    with open(file_path, 'r') as f:
        text_data = f.read()
    lines = [l.strip() for l in text_data.split('\n') if l.strip() and not l.startswith("#")]
    parsed_rows = [line.split(maxsplit=22) for line in lines]
    columns = [
        "target_name", "target_accession", "tlen", 
        "query_name", "query_accession", "qlen",
        "full_e_value", "full_score", "full_bias",
        "dom_idx", "dom_total", 
        "dom_c_evalue", "dom_i_evalue", "dom_score", "dom_bias",
        "hmm_from", "hmm_to", 
        "ali_from", "ali_to", 
        "env_from", "env_to",
        "acc",
        "description"
    ]
    df = pl.DataFrame(parsed_rows, schema=columns)
    numeric_cols = ["tlen", "qlen", "hmm_from", "hmm_to", "ali_from", "ali_to"]
    float_cols = ["full_e_value", "full_score", "dom_score"]


    df = df.with_columns([
        pl.col(numeric_cols).cast(pl.Int32),
        pl.col(float_cols).cast(pl.Float64)
    ])
    return df


df = parse_hmm_tblout(domtbl)



df = df.with_columns(
    pl.col("query_name").str.split("_").list.get(-1).alias("stop")
)

df = df.with_columns(
    pl.col("query_name").str.split("_").list.get(-2).alias("start")
)

df = df.with_columns(
    pl.col("query_name")
    .str.split("_")
    .list.slice(0, pl.col("query_name").str.split("_").list.len() - 2)
    .list.join("_")
    .alias("plasmid")
)

headers = []
oris = []

i = 0
with open('hmminput_allplasmid_proteins_strandorientation.fa', 'r') as f:
    for line in f:
        if i % 2 == 0:
            headers.append(line[1:])
        else:    
            oris.append(line)
        i += 1


orientation_dict = dict(zip(headers, oris))
orientation_dict = {k.strip():v.strip() for k,v in orientation_dict.items()}

df = df.with_columns(
    pl.col("query_name").replace(orientation_dict).alias("strand")
)



df.write_parquet(outpath)