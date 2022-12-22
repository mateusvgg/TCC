import pandas as pd

from data_loader import LoadProjectionsData


REF_BASE = r'D:\Arquivos\Desktop\APSIPA___M-PCCD\ref_projections'
DEG_BASE = r'D:\Arquivos\Desktop\APSIPA___M-PCCD\deg_projections'
DF_DATA_PATH = r'D:\Arquivos\Desktop\APSIPA___M-PCCD\apsipa.csv'


def get_projections_data():
    df_data = pd.read_csv(DF_DATA_PATH)
    generator = LoadProjectionsData(
        ref_base=REF_BASE,
        deg_base=DEG_BASE,
        df_data=df_data
    )
    return generator.prepare_data()


if __name__=='__main__':
    data = get_projections_data()