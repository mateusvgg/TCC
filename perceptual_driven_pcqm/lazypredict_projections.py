from sklearn.model_selection import train_test_split
from Supervised import LazyRegressor
import torch
from joblib import Memory
import numpy as np
from typing import NewType
from numpy.typing import ArrayLike
import pandas as pd

ModelData = NewType("ModelData", tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike])

cache_dir = "cache"
memory = Memory(cache_dir, verbose=0)


@memory.cache
def load_and_preprocess_data(dataset: str) -> ModelData:
    print("Preprocessing data...")
    X = torch.load(f"data/X_tensor_{dataset}.pt")
    X = [[v.cpu().detach().numpy() for v in x] for x in X]
    y = torch.load(f"data/y_tensor_{dataset}.pt")

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=56
    )

    return X_train, X_test, y_train, y_test


# @memory.cache
def run_lazypredict(data: ModelData, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train, x_test, y_train, y_test = data

    print("Running LazyPredict...")
    reg = LazyRegressor()
    models, predictions = reg.fit(x_train, x_test, y_train, y_test)

    return models, predictions


# @memory.cache
def save_results(models: pd.DataFrame, predictions: pd.DataFrame, dataset: str):
    print("Saving results...")
    models.to_markdown(f"results/lazypredict_{dataset}_projections.md")
    predictions.to_markdown(f"results/lazypredict_{dataset}_projections_predictions.md")
    models.to_csv(f"results/lazypredict_{dataset}_projections.csv")
    predictions.to_csv(f"results/lazypredict_{dataset}_projections_predictions.csv")
    print("Results saved")


def main():
    for dataset in ["APSIPA", "WPC"]:
        print(f"Running for {dataset}...")
        data = load_and_preprocess_data(dataset)
        models, predictions = run_lazypredict(data, dataset)
        save_results(models, predictions, dataset)


if __name__ == "__main__":
    main()
