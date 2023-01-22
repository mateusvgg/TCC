import os
import pandas as pd
from tqdm import tqdm

from gen_projections_from_ply import gen_projections


DBS_METADATA = {
    # 'APSIPA': {
    #     'ref_base': '/mnt/d/Arquivos/Desktop/APSIPA___M-PCCD/ref_projections/',
    #     'deg_base': '/mnt/d/Arquivos/Desktop/APSIPA___M-PCCD/deg_projections/',
    #     'df_path': '/mnt/d/Arquivos/Desktop/APSIPA___M-PCCD/APSIPA.csv'
    # },
    'WPC': {
        'ref_base': '/mnt/d/Arquivos/Desktop/WPC/ref_projections/',
        'deg_base': '/mnt/d/Arquivos/Desktop/WPC/deg_projections/',
        'df_path': '/mnt/d/Arquivos/Desktop/WPC/WPC.csv'
    }
}


def run_projs_dataset(db_info):
    df_paths = pd.read_csv(db_info['df_path'])

    df_paths['SIGNAL'] = df_paths['SIGNAL'].apply(lambda x: x.strip())
    df_paths['REF'] = df_paths['REF'].apply(lambda x: x.strip())
    df_paths['LOCATION'] = df_paths['LOCATION'].apply(lambda x: x.strip())
    df_paths['REFLOCATION'] = df_paths['REFLOCATION'].apply(lambda x: x.strip())

    df_refs = df_paths.drop_duplicates(subset='REF')

    print(f"Ref Projections for {db_info['df_path']}:")
    for _, row in tqdm(df_refs.iterrows()):
        pc_path = os.path.join(*[row['REFLOCATION'], row['REF']])
        ref = row['REF'].replace('.ply', '')
        path_to_save = os.path.join(*[db_info['ref_base'], ref])
        gen_projections(pc_path, path_to_save)

    print(f"Deg Projections for {db_info['df_path']}:")
    for _, row in tqdm(df_paths.iterrows()):
        pc_path = os.path.join(*[row['LOCATION'], row['SIGNAL']])
        deg = row['SIGNAL'].replace('.ply', '')
        path_to_save = os.path.join(*[db_info['deg_base'], deg])
        gen_projections(pc_path, path_to_save)


if __name__=="__main__":
    for dataset in DBS_METADATA.keys():
        run_projs_dataset(DBS_METADATA[dataset])
        