| Model                         |   Adjusted R-Squared |    R-Squared |     RMSE |    Pearson |   Spearman |   Time Taken |
|:------------------------------|---------------------:|-------------:|---------:|-----------:|-----------:|-------------:|
| NuSVR                         |            0.786916  |   0.798033   | 0.567515 |   0.894148 |   0.906114 |   0.00348401 |
| SVR                           |            0.783115  |   0.794431   | 0.572554 |   0.896619 |   0.907729 |   0.00327492 |
| MLPRegressor                  |            0.77647   |   0.788133   | 0.581258 |   0.893784 |   0.904799 |   0.143475   |
| ExtraTreesRegressor           |            0.76398   |   0.776294   | 0.597277 |   0.885565 |   0.894283 |   0.0437362  |
| KNeighborsRegressor           |            0.762752  |   0.77513    | 0.598829 |   0.886687 |   0.89786  |   0.00296116 |
| HistGradientBoostingRegressor |            0.745627  |   0.758899   | 0.620064 |   0.875881 |   0.893443 |   0.782669   |
| AdaBoostRegressor             |            0.739279  |   0.752882   | 0.627753 |   0.869075 |   0.901433 |   0.031343   |
| GradientBoostingRegressor     |            0.736124  |   0.749891   | 0.631541 |   0.873413 |   0.884702 |   0.047848   |
| PoissonRegressor              |            0.732461  |   0.746419   | 0.635909 |   0.878719 |   0.907099 |   0.0189822  |
| RandomForestRegressor         |            0.731034  |   0.745067   | 0.637602 |   0.871372 |   0.885973 |   0.119957   |
| LGBMRegressor                 |            0.72866   |   0.742817   | 0.64041  |   0.86696  |   0.887056 |   0.0821159  |
| XGBRegressor                  |            0.713034  |   0.728006   | 0.658592 |   0.862434 |   0.873238 |   0.461734   |
| BaggingRegressor              |            0.71186   |   0.726894   | 0.659937 |   0.863225 |   0.88548  |   0.0093081  |
| GammaRegressor                |            0.642579  |   0.661227   | 0.735007 |   0.867849 |   0.906256 |   0.00734472 |
| OrthogonalMatchingPursuitCV   |            0.610806  |   0.631112   | 0.76698  |   0.799006 |   0.916487 |   0.00410485 |
| LassoLarsCV                   |            0.609502  |   0.629876   | 0.768264 |   0.802221 |   0.921222 |   0.0134628  |
| Lars                          |            0.609251  |   0.629638   | 0.768511 |   0.796984 |   0.913691 |   0.00647998 |
| LassoCV                       |            0.609117  |   0.629511   | 0.768643 |   0.80164  |   0.919245 |   0.0453422  |
| LinearSVR                     |            0.608957  |   0.629359   | 0.7688   |   0.793331 |   0.904749 |   0.0177081  |
| LassoLarsIC                   |            0.607922  |   0.628378   | 0.769817 |   0.800324 |   0.918495 |   0.00385404 |
| RidgeCV                       |            0.605877  |   0.62644    | 0.771822 |   0.798853 |   0.915349 |   0.00286317 |
| ElasticNetCV                  |            0.605818  |   0.626384   | 0.77188  |   0.800873 |   0.918572 |   0.022285   |
| TransformedTargetRegressor    |            0.605626  |   0.626202   | 0.772068 |   0.797346 |   0.913195 |   0.00298476 |
| LinearRegression              |            0.605626  |   0.626202   | 0.772068 |   0.797346 |   0.913195 |   0.00281501 |
| Ridge                         |            0.596748  |   0.617787   | 0.780709 |   0.798496 |   0.913599 |   0.00290322 |
| HuberRegressor                |            0.595974  |   0.617054   | 0.781458 |   0.793478 |   0.909487 |   0.00530791 |
| SGDRegressor                  |            0.594258  |   0.615427   | 0.783116 |   0.796039 |   0.906045 |   0.00288892 |
| BayesianRidge                 |            0.591008  |   0.612347   | 0.786246 |   0.796604 |   0.909737 |   0.00321889 |
| OrthogonalMatchingPursuit     |            0.578052  |   0.600067   | 0.798603 |   0.7882   |   0.902183 |   0.00285006 |
| DecisionTreeRegressor         |            0.575925  |   0.598051   | 0.800613 |   0.817046 |   0.853197 |   0.00282001 |
| TweedieRegressor              |            0.568602  |   0.59111    | 0.807496 |   0.796045 |   0.90596  |   0.0579042  |
| ExtraTreeRegressor            |            0.563905  |   0.586658   | 0.81188  |   0.812368 |   0.831617 |   0.0028162  |
| LarsCV                        |            0.561421  |   0.584304   | 0.814189 |   0.797528 |   0.909887 |   0.018337   |
| ElasticNet                    |            0.417307  |   0.447709   | 0.938471 |   0.796031 |   0.906968 |   0.00278115 |
| LassoLars                     |            0.0909239 |   0.138354   | 1.1722   |   0.7882   |   0.902183 |   0.0256493  |
| Lasso                         |            0.0909237 |   0.138354   | 1.1722   |   0.7882   |   0.902183 |   0.00302267 |
| RANSACRegressor               |           -0.0508362 |   0.00399003 | 1.26028  |   0.688229 |   0.757089 |   0.048449   |
| DummyRegressor                |           -0.0606226 |  -0.00528578 | 1.26614  | nan        | nan        |   0.00224113 |
| QuantileRegressor             |           -0.266956  |  -0.200854   | 1.38383  |   0.789571 |   0.896009 |  13.1432     |
| PassiveAggressiveRegressor    |           -0.446927  |  -0.371435   | 1.47885  |   0.796735 |   0.909664 |   0.00302982 |
| KernelRidge                   |           -3.0938    |  -2.88021    | 2.48751  |   0.798495 |   0.913599 |   0.0258622  |
| GaussianProcessRegressor      |          -13.8244    | -13.051      | 4.73359  |   0.318604 |   0.470627 |   0.0428648  |