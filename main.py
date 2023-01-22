import torch
import pandas as pd
from tqdm import tqdm
import joblib

from src.data_loader import LoadProjectionsData
from src.model import DISTS


DBS_METADATA = {
    'APSIPA': {
        'ref_base': r'./APSIPA_projections/ref_projections/',
        'deg_base': r'./APSIPA_projections/deg_projections/',
        'df_path': r'./APSIPA.csv'
    },
    'QOMEX': {
        'ref_base': r'./qomex_projections/ref_projections/',
        'deg_base': r'./qomex_projections/deg_projections/',
        'df_path': r'./QOMEX.csv'
    },
    'WPC': {
        'ref_base': r'./wpc_projections/ref_projections/',
        'deg_base': r'./wpc_projections/deg_projections/',
        'df_path': r'./WPC.csv'
    },
    'UNB-PCQA': {
        'ref_base': r'./unb_projections/ref_projections/',
        'deg_base': r'./unb_projections/deg_projections/',
        'df_path': r'./WPC.csv'
    },
    'SJTU': {
        'ref_base': r'./sjtu_projections/ref_projections/',
        'deg_base': r'./sjtu_projections/deg_projections/',
        'df_path': r'./WPC.csv'
    }
}


def gen_x_and_y(
    db_metadata: dict[str, str],
    model: DISTS
) -> tuple[list[list[float]], list[float], list[str]]:
    df_data = pd.read_csv(db_metadata['df_path'])

    generator = LoadProjectionsData(
        ref_base=db_metadata['ref_base'],
        deg_base=db_metadata['deg_base'],
        df_data=df_data
    )

    device = torch.device(
        "mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    print(f"model is using {device}")
    data_gen = generator.data_generator()

    X, y, ref_names = [], [], []
    for pair_proj in tqdm(data_gen):
        X.append(model(pair_proj))
        y.append(pair_proj.score)
        ref_names.append(pair_proj.ref.name)

    return X, y, ref_names


if __name__ == '__main__':
    for db in DBS_METADATA.keys():
        model = DISTS()
        X, y, ref_names = gen_x_and_y(DBS_METADATA[db], model)
        torch.save(X, f'X_tensor_{db}.pt')
        torch.save(y, f'y_tensor_{db}.pt')
        joblib.dump(ref_names, f'ref_names_{db}.pkl')
