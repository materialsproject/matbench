from catboost import CatBoostRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def candidates():
    result = []
    for features in ("fractions", "magpie", "combined"):
        for leaf in (1, 2):
            result.append((f"et-{features}-leaf{leaf}", features,
                           ExtraTreesRegressor(n_estimators=400, max_features=0.8,
                               min_samples_leaf=leaf, random_state=1701, n_jobs=4)))
        result.append((f"rf-{features}", features,
                       RandomForestRegressor(n_estimators=400, max_features=0.8,
                           min_samples_leaf=1, random_state=1701, n_jobs=4)))
        for depth in (3, 5, 7):
            for loss in ("MAE", "RMSE"):
                result.append((f"cat-{features}-depth{depth}-{loss}", features,
                               CatBoostRegressor(iterations=800, depth=depth,
                                   learning_rate=0.035, loss_function=loss,
                                   l2_leaf_reg=3, random_seed=1701, thread_count=4,
                                   verbose=False, allow_writing_files=False)))
        for loss in ("absolute_error", "squared_error"):
            result.append((f"hgb-{features}-{loss}", features,
                           HistGradientBoostingRegressor(loss=loss, max_iter=400,
                               max_leaf_nodes=7, min_samples_leaf=5, learning_rate=0.04,
                               l2_regularization=1, early_stopping=False, random_state=1701)))
    for alpha in (0.1, 1, 10):
        for gamma in (0.001, 0.01, 0.1):
            result.append((f"krr-fractions-a{alpha}-g{gamma}", "fractions",
                TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),
                    KernelRidge(alpha=alpha, gamma=gamma, kernel="rbf")),
                    transformer=StandardScaler())))
    return [(name, features, make_pipeline(SimpleImputer(keep_empty_features=True), model))
            for name, features, model in result]

