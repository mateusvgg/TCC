import torch
import numpy as np
import joblib
from typing import NewType
from numpy.typing import ArrayLike
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import SGDRegressor, PoissonRegressor, HuberRegressor
from sklearn.svm import SVR, NuSVR


"""
LazyPredict results:
| Model                         |    RMSE |    Pearson |   Spearman |   Time Taken |
|:------------------------------|--------:|-----------:|-----------:|-------------:|
| SVR                           | 12.9257 |   0.821104 |   0.840326 |   0.0114079  |
| NuSVR                         | 13.2125 |   0.819732 |   0.840948 |   0.0106661  |
| PoissonRegressor              | 13.1844 |   0.806891 |   0.838016 |   0.027118   |
| HuberRegressor                | 13.2998 |   0.80555  |   0.834461 |   0.00759292 |
| SGDRegressor                  | 13.4321 |   0.804411 |   0.834669 |   0.00345707 |
"""

ModelData = NewType("ModelData", tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike])

cache_dir = "cache"
memory = joblib.Memory(cache_dir, verbose=0)

SCORING = "neg_root_mean_squared_error"
N_ITER = 800


@memory.cache
def load_and_preprocess_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = torch.load("data/X_tensor_WPC.pt")
    y = torch.load("data/y_tensor_WPC.pt")

    x = np.array([[v.cpu().detach().numpy() for v in t] for t in x])
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    return x_train, x_test, y_train, y_test


def run_random_search_SVR(data: ModelData):
    x_train, x_test, y_train, y_test = data
    svr = SVR()
    PARAM_GRID = {
        "C": [0.1, 1, 5, 10, 50, 100],
        "gamma": [0.01, 0.1, 1, 10],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [1, 2, 3, 4],
        "epsilon": [0.01, 0.1, 1],
    }

    random_search = RandomizedSearchCV(
        svr,
        param_distributions=PARAM_GRID,
        n_iter=N_ITER,
        cv=5,
        n_jobs=-1,
        scoring=SCORING,
        verbose=1,
    )
    random_search.fit(x_train, y_train)
    print("\nBest params for SVR")
    print(random_search.best_params_)
    print(random_search.best_estimator_)
    return random_search.best_estimator_, random_search.best_params_


def run_random_search_NuSVR(data: ModelData):
    x_train, x_test, y_train, y_test = data
    nu_svr = NuSVR()
    PARAM_GRID = {
        "nu": [0.1, 0.3, 0.5, 0.7, 0.9],
        "C": [0.1, 1, 5, 10, 50, 100],
        "gamma": [0.001, 0.01, 0.1, 1, 10, 100],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [1, 2, 3, 4],
    }

    random_search = RandomizedSearchCV(
        nu_svr,
        param_distributions=PARAM_GRID,
        n_iter=N_ITER,
        cv=5,
        n_jobs=-1,
        scoring=SCORING,
        verbose=1,
    )
    random_search.fit(x_train, y_train)
    print("\nBest params for NuSVR")
    print(random_search.best_params_)
    print(random_search.best_estimator_)
    return random_search.best_estimator_, random_search.best_params_


def run_random_search_PoissonRegressor(data: ModelData):
    x_train, x_test, y_train, y_test = data
    poisson_regressor = PoissonRegressor()
    PARAM_GRID = {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        "max_iter": np.arange(100, 500, 100),
    }

    random_search = RandomizedSearchCV(
        poisson_regressor,
        param_distributions=PARAM_GRID,
        n_iter=N_ITER,
        cv=5,
        n_jobs=-1,
        scoring=SCORING,
        verbose=1,
    )
    random_search.fit(x_train, y_train)
    print("\nBest params for PoissonRegressor")
    print(random_search.best_params_)
    print(random_search.best_estimator_)
    return random_search.best_estimator_, random_search.best_params_


def run_random_search_HuberRegressor(data: ModelData):
    x_train, x_test, y_train, y_test = data
    huber_regressor = HuberRegressor()
    PARAM_GRID = {
        "epsilon": np.linspace(1.0, 2.0),
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        "max_iter": np.arange(100, 500, 100),
    }

    random_search = RandomizedSearchCV(
        huber_regressor,
        param_distributions=PARAM_GRID,
        n_iter=N_ITER,
        cv=5,
        n_jobs=-1,
        scoring=SCORING,
        verbose=1,
    )
    random_search.fit(x_train, y_train)
    print("\nBest params for HuberRegressor")
    print(random_search.best_params_)
    print(random_search.best_estimator_)
    return random_search.best_estimator_, random_search.best_params_


def run_random_search_SGDRegressor(data: ModelData):
    x_train, x_test, y_train, y_test = data
    sgd_regressor = SGDRegressor()
    PARAM_GRID = {"max_iter": [900, 1000, 1500, 2000]}

    random_search = RandomizedSearchCV(
        sgd_regressor,
        param_distributions=PARAM_GRID,
        n_iter=N_ITER,
        cv=5,
        n_jobs=-1,
        scoring=SCORING,
        verbose=1,
    )
    random_search.fit(x_train, y_train)
    print("\nBest params for SGDRegressor")
    print(random_search.best_params_)
    print(random_search.best_estimator_)
    return random_search.best_estimator_, random_search.best_params_


if __name__ == "__main__":
    data = load_and_preprocess_data()
    print("Performing random search for SVR...")
    run_random_search_SVR(data)
    print("Performing random search for NuSVR...")
    run_random_search_NuSVR(data)
    print("Performing random search for HuberRegressor...")
    run_random_search_PoissonRegressor(data)
    print("Performing random search for HuberRegressor...")
    run_random_search_HuberRegressor(data)
    print("Performing random search for SGDRegressor...")
    run_random_search_SGDRegressor(data)
