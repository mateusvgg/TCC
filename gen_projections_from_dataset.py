import os
import pandas as pd
from tqdm import tqdm

from gen_projections_from_ply import gen_projections


projections_ref_path = r'/mnt/d/Arquivos/Desktop/APSIPA___M-PCCD/ref_projections'
projections_deg_path = r'/mnt/d/Arquivos/Desktop/APSIPA___M-PCCD/deg_projections'

df_paths = pd.read_csv('apsipa.csv')

df_paths['SIGNAL'] = df_paths['SIGNAL'].apply(lambda x: x.strip())
df_paths['REF'] = df_paths['REF'].apply(lambda x: x.strip())
df_paths['LOCATION'] = df_paths['LOCATION'].apply(lambda x: x.strip())
df_paths['REFLOCATION'] = df_paths['REFLOCATION'].apply(lambda x: x.strip())

df_refs = df_paths.drop_duplicates(subset='REF')

for _, row in tqdm(df_refs.iterrows()):
    pc_path = os.path.join(*[row['REFLOCATION'], row['REF']])
    ref = row['REF'].replace('.ply', '')
    path_to_save = os.path.join(*[projections_ref_path, ref])
    gen_projections(pc_path, path_to_save)

for _, row in tqdm(df_paths.iterrows()):
    pc_path = os.path.join(*[row['LOCATION'], row['SIGNAL']])
    deg = row['SIGNAL'].replace('.ply', '')
    path_to_save = os.path.join(*[projections_deg_path, deg])
    gen_projections(pc_path, path_to_save)