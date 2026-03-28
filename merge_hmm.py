import polars as pl
from pathlib import Path

data_dir = Path('/home/kd541/rds/hpc-work/plasmid_motif_network/intermediate')
files = sorted(data_dir.glob("parsed_*.parquet"))

df_merged = pl.concat([pl.read_parquet(f) for f in files]).rechunk()

output_file = data_dir.parent / "hmm_env_full_merged.parquet"
df_merged.write_parquet(output_file)
