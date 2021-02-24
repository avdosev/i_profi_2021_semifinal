from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor

TEST_TRAIN = False
# TEST_TRAIN = True

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
test['NTG'] = np.zeros(len(test))


if TEST_TRAIN:
    X, y = train.drop(['Well', 'NTG'], axis=1), train['NTG']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
else:
    X, y = train.drop(['Well', 'NTG'], axis=1), train['NTG']
    X_train, X_test, y_train, y_test = X,X,y,y


# model = RandomForestRegressor(max_depth=30)
models = [
    (LinearRegression(), 'linear'),
    (make_pipeline(PolynomialFeatures(2), LinearRegression()), 'polinom 2 linear'),
    (make_pipeline(PolynomialFeatures(3), LinearRegression()), 'polinom 3 linear'),
    (make_pipeline(PolynomialFeatures(4), LinearRegression()), 'polinom 4 linear'), # 4
    (make_pipeline(PolynomialFeatures(5), LinearRegression()), 'polinom 5 linear'),
    (make_pipeline(PolynomialFeatures(6), LinearRegression()), 'polinom 6 linear'),
    (make_pipeline(PolynomialFeatures(7), LinearRegression()), 'polinom 7 linear'),
    (make_pipeline(PolynomialFeatures(8), LinearRegression()), 'polinom 8 linear'),
    (make_pipeline(PolynomialFeatures(9), LinearRegression()), 'polinom 9 linear'),
    # (BaggingRegressor(n_jobs=-1), 'bagging'),
    (ExtraTreesRegressor(n_jobs=-1,), 'extra_trees'),
    (make_pipeline(PolynomialFeatures(2), ExtraTreesRegressor(n_jobs=-1,)), 'polinom 2 extra_trees'), # 7 16:21
    (make_pipeline(PolynomialFeatures(3), ExtraTreesRegressor(n_jobs=-1,)), 'polinom 3 extra_trees'),
    (make_pipeline(PolynomialFeatures(4), ExtraTreesRegressor(n_jobs=-1,)), 'polinom 4 extra_trees'), 
    (make_pipeline(PolynomialFeatures(5), ExtraTreesRegressor(n_jobs=-1,)), 'polinom 5 extra_trees'),
    (make_pipeline(PolynomialFeatures(6), ExtraTreesRegressor(n_jobs=-1,)), 'polinom 6 extra_trees'),
    (make_pipeline(PolynomialFeatures(8), ExtraTreesRegressor(n_jobs=-1,)), 'polinom 8 extra_trees'),
    (make_pipeline(PolynomialFeatures(10), ExtraTreesRegressor(n_jobs=-1,)), 'polinom 10 extra_trees'),
    (ExtraTreesRegressor(n_jobs=-1, max_depth=5), 'extra_trees max_d=5'), # 3
    (ExtraTreesRegressor(n_jobs=-1, max_depth=10), 'extra_trees max_d=10'),
    (GradientBoostingRegressor(), 'grad_boosting'),
    (make_pipeline(PolynomialFeatures(2), GradientBoostingRegressor()), 'polinom 2 grad_boosting'),
    (make_pipeline(PolynomialFeatures(4), GradientBoostingRegressor()), 'polinom 4 grad_boosting'),
    (make_pipeline(PolynomialFeatures(5), GradientBoostingRegressor()), 'polinom 5 grad_boosting'),
    (make_pipeline(PolynomialFeatures(6), GradientBoostingRegressor()), 'polinom 6 grad_boosting'),
    (make_pipeline(PolynomialFeatures(8), GradientBoostingRegressor()), 'polinom 8 grad_boosting'),
    (RandomForestRegressor(n_estimators=25, n_jobs=-1), 'rand_forset n_est=25'),
    (RandomForestRegressor(n_estimators=10, n_jobs=-1), 'rand_forset n_est=10'),
    (RandomForestRegressor(n_estimators=5, n_jobs=-1), 'rand_forset n_est=5'),
    (RandomForestRegressor(n_estimators=2, n_jobs=-1), 'rand_forset n_est=2'),
    (DecisionTreeRegressor(), 'tree'),
    (make_pipeline(PolynomialFeatures(4), DecisionTreeRegressor()), 'polinom 4 tree'),
    (KNeighborsRegressor(n_neighbors=2, weights='distance'), 'neighbors 2 distance'), # 2 
    (KNeighborsRegressor(n_neighbors=5, weights='distance'), 'neighbors 5 distance'),
    (KNeighborsRegressor(n_neighbors=6, weights='distance'), 'neighbors 6 distance'),
    (KNeighborsRegressor(n_neighbors=7, weights='distance'), 'neighbors 7 distance'),
    (KNeighborsRegressor(n_neighbors=8, weights='distance'), 'neighbors 8 distance'), # 1
    (KNeighborsRegressor(n_neighbors=9, weights='distance'), 'neighbors 9 distance'),
    (KNeighborsRegressor(n_neighbors=10, weights='distance'), 'neighbors 10 distance'), # 5 15:26
    (KNeighborsRegressor(n_neighbors=15, weights='distance'), 'neighbors 15 distance'),
    (KNeighborsRegressor(n_neighbors=20, weights='distance'), 'neighbors 20 distance'),
    (CatBoostRegressor(silent=True, iterations=900), 'cat boost reg'), # 6 15:27
]

for model, name  in models:
    model.fit(X_train, y_train)

    print(name.ljust(21, ' '), 'score: ', "{:.20f}".format(mean_squared_error(y_test, model.predict(X_test))))
    X_predict = test.drop(['Well', 'NTG'], axis=1)
    out = model.predict(X_predict)
    test['NTG'] = out
    test.to_csv(f'./output/{name}.csv', index=False)