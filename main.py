import torch
from src.utils import gen_x_and_y


REF_BASE = r'./APSIPA_projections/ref_projections/'
DEG_BASE = r'./APSIPA_projections/deg_projections/'
DF_DATA_PATH = r'./apsipa.csv'


if __name__ == '__main__':
    X, y = gen_x_and_y(DF_DATA_PATH, REF_BASE, DEG_BASE)
    torch.save(X, 'X_tensor.pt')
    torch.save(y, 'y_tensor.pt')
