import torch
import pandas as pd
from src.data_loader import LoadProjectionsData
from src.model import DISTS
from tqdm import tqdm


def get_projections_data(df_data_path, ref_base, deg_base):
    df_data = pd.read_csv(df_data_path)
    df_data = df_data.sample(frac=0.1)
    generator = LoadProjectionsData(
        ref_base=ref_base,
        deg_base=deg_base,
        df_data=df_data
    )
    return generator.prepare_data()


def forward_test(df_data_path, ref_base, deg_base):
    data = get_projections_data(df_data_path, ref_base, deg_base)
    model = DISTS(use_pooling=True)
    outputs = [model(sample) for sample in tqdm(data)]
    real = [sample.score for sample in data]
    _ = [print(f'out = {o} - true = {r}') for o, r in zip(outputs, real)]
    # print('\n\n')
    # _ = [print(f'out = {(1-o)*5} - true = {r}') for o, r in zip(outputs, real)]


def gen_x_and_y(df_data_path, ref_base, deg_base) -> tuple[list[list[float]], list[float]]:
    df_data = pd.read_csv(df_data_path)

    generator = LoadProjectionsData(
        ref_base=ref_base,
        deg_base=deg_base,
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
