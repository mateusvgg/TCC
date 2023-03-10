import torch
import pandas as pd
from tqdm import tqdm
import joblib

from perceptual_driven_pcqm.data_loader import LoadProjectionsData
from perceptual_driven_pcqm.model import DISTS


DBS_METADATA = {
    'APSIPA': {
        'ref_base': '/Volumes/mnt/APSIPA_projections/ref_projections/',
        'deg_base': '/Volumes/mnt/APSIPA_projections/deg_projections/',
        'df_path': '/Volumes/mnt/APSIPA_projections/APSIPA.csv'
    },
    'WPC': {
        'ref_base': r'/Volumes/mnt/WPC/ref_projections/',
        'deg_base': r'/Volumes/mnt/WPC/deg_projections/',
        'df_path': r'/Volumes/mnt/WPC/WPC_BACKUP.csv'
    }
}


def gen_x_and_y(
    db_metadata: dict[str, str],
    model: DISTS,
    db_name: str
) -> tuple[list[list[float]], list[float], list[str], list[str]]:
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

    if db_name == 'WPC':
        degradations = list(df_data['Distortion_Type'])
    elif db_name == 'APSIPA':
        degradations = list(df_data['ATTACK'])

    return X, y, ref_names, degradations


if __name__ == '__main__':
    for db in DBS_METADATA.keys():
        model = DISTS()
        X, y, ref_names, codecs = gen_x_and_y(DBS_METADATA[db], model, db)
        torch.save(X, f'./data/X_tensor_{db}.pt')
        torch.save(y, f'./data/y_tensor_{db}.pt')
        joblib.dump(ref_names, f'./data/ref_names_{db}.pkl')
        joblib.dump(codecs, f'./data/degradations_{db}.pkl')
