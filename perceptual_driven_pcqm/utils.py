import pandas as pd
from perceptual_driven_pcqm.data_loader import LoadProjectionsData
from perceptual_driven_pcqm.model import DISTS
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
