import pandas as pd
from tqdm import tqdm

from data_loader import LoadProjectionsData
from model import DISTS


REF_BASE = r'/mnt/d/Arquivos/Desktop/APSIPA___M-PCCD/ref_projections'
DEG_BASE = r'/mnt/d/Arquivos/Desktop/APSIPA___M-PCCD/deg_projections'
DF_DATA_PATH = r'/mnt/d/Arquivos/Desktop/APSIPA___M-PCCD/apsipa_wsl.csv'


def get_projections_data():
    df_data = pd.read_csv(DF_DATA_PATH)
    df_data = df_data.sample(frac=0.1)
    generator = LoadProjectionsData(
        ref_base=REF_BASE,
        deg_base=DEG_BASE,
        df_data=df_data
    )
    return generator.prepare_data()


def forward_test():
    data = get_projections_data()
    model = DISTS(use_pooling=True)
    outputs = [model(sample) for sample in tqdm(data)]
    real = [sample.score for sample in data]
    _ = [print(f'out = {o} - true = {r}') for o, r in zip(outputs, real)]
    # print('\n\n')
    # _ = [print(f'out = {(1-o)*5} - true = {r}') for o, r in zip(outputs, real)]


def gen_x_and_y() -> tuple[list[list[float]], list[float]]:
    df_data = pd.read_csv(DF_DATA_PATH)
    
    generator = LoadProjectionsData(
        ref_base=REF_BASE,
        deg_base=DEG_BASE,
        df_data=df_data
    )

    model = DISTS()
    device = torch.device(
        "mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    print(f"model is using {device}")
    data_gen = generator.data_generator()

    X, y = [], []
    for pair_proj in data_gen:
        X.append(model(pair_proj))
        y.append(pair_proj.score)

    return X, y


if __name__ == '__main__':
    X, y = gen_x_and_y()
    torch.save(X, 'X_tensor.pt')
    torch.save(y, 'y_tensor.pt')
