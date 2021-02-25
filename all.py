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
    (make_pipeline(PolynomialFeatures(4), LinearRegression()), 'polinom 4 linear'), # 4 15:26
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
    (ExtraTreesRegressor(n_jobs=-1, max_depth=5), 'extra_trees max_d=5'), # 3 15:23
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
    (KNeighborsRegressor(n_neighbors=2, weights='distance'), 'neighbors 2 distance'), # 2 13:54
    (KNeighborsRegressor(n_neighbors=5, weights='distance'), 'neighbors 5 distance'),
    (KNeighborsRegressor(n_neighbors=6, weights='distance'), 'neighbors 6 distance'),
    (KNeighborsRegressor(n_neighbors=7, weights='distance'), 'neighbors 7 distance'),
    (KNeighborsRegressor(n_neighbors=8, weights='distance'), 'neighbors 8 distance'), # 1  13:53:
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

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy as sp

x_left = 201
size_x = 247 - 200
y_left = 901
size_y = 930 - 900

arr = np.full((size_x, size_y), -1)

TEST_TRAIN = False
TEST_TRAIN = True

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

def save_res(rs, name):
    arr = np.zeros(len(test))
    for i, point in test.iterrows():
        arr[i] = rs[point.Y-y_left, point.X-x_left]
    test['NTG'] = arr
    test.to_csv(f'./output/{name}.csv', index=False)

X, y = train.drop(['Well', 'NTG'], axis=1), train['NTG']

x_grid = np.linspace(x_left, x_left+size_x, size_x)
y_grid = np.linspace(y_left, y_left+size_y, size_y)
B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')

# 25 фев 2021, 14:23:28
spline = sp.interpolate.Rbf(X.X,X.Y,y,function='linear',smooth=5, episilon=5)
res = spline(B1, B2)
plt.imshow(res)
plt.savefig('data_visual_2.png')
save_res(res, 'rbf_linear')

# 25 фев 2021, 14:24:34
spline = sp.interpolate.Rbf(X.X,X.Y,y,function='multiquadric',smooth=5, episilon=5)
res = spline(B1, B2)
plt.imshow(res)
plt.savefig('data_visual_2_multiquadric.png')
save_res(res, 'rbf_multiquadric')

# 25 фев 2021, 14:15:50
interp = sp.interpolate.NearestNDInterpolator(list(zip(X.X,X.Y)), y)
res = interp(B1, B2)
plt.imshow(res)
plt.savefig('data_visual_3.png')
save_res(res, 'nearest_interpolator')

interp = sp.interpolate.LinearNDInterpolator(list(zip(X.X,X.Y)), y)
res = interp(B1, B2)
plt.imshow(res)
plt.savefig('data_visual_4.png')
save_res(res, 'linear_interpolator')

interp = sp.interpolate.CloughTocher2DInterpolator(list(zip(X.X,X.Y)), y)
res = interp(B1, B2)
plt.imshow(res)
plt.savefig('data_visual_5.png')
save_res(res, 'CloughTocher_interpolator')

# 25 фев 2021, 13:46:43
# metod3_2 14:06
# metod3_3 14:07
# metod3_4 14:08

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy as sp

x_left = 201
size_x = 247 - 200
y_left = 901
size_y = 930 - 900

arr = np.full((size_x, size_y), -1.)

TEST_TRAIN = False
TEST_TRAIN = True

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

def save_res(rs, name):
    arr = np.zeros(len(test))
    for i, point in test.iterrows():
        arr[i] = rs[point.X-x_left, point.Y-y_left]
    test['NTG'] = arr
    test.to_csv(f'./output/{name}.csv', index=False)

for i, point in train.iterrows():
    x, y = point.X-x_left, point.Y-y_left
    arr[x,y] = point.NTG

def normal_p(p, p_s):
    return p >= 0 and p < p_s

def iter_radius(x, y, r):
    for x_i in range(-r, r+1):
        for y_i in range(-r, r+1):
            if normal_p(x_i+x, size_x) and normal_p(y_i+y, size_y):
                yield x_i+x, y_i+y

res = np.full((size_x, size_y), -1.)
for i, point in train.iterrows():
    x, y = point.X-x_left, point.Y-y_left
    for x_r, y_r in iter_radius(x, y, 1):
        if res[x_r, y_r] != -1:
            continue
        points = [arr[x_p, y_p] for x_p, y_p in iter_radius(x_r, y_r, 4) if arr[x_p, y_p] != -1]
        # print(points)
        avg = sum(points) / len(points)
        res[x_r, y_r] = avg
    res[x, y] = point.NTG

plt.imshow(res)
plt.show()

res2 = np.copy(res)
for y in range(size_y):
    for x in range(size_x):
        if res[x, y] == -1:
            points = [res[x_p, y_p] for x_p, y_p in iter_radius(x, y, 1) if res[x_p, y_p] != -1]
            avg = sum(points) / len(points)
            res[x, y] = avg

plt.imshow(res)
plt.savefig('data_visual_6.png')
# plt.show()

for x in range(size_x):
    for y in range(size_y):
        if res2[x, y] == -1:
            points = [res2[x_p, y_p] for x_p, y_p in iter_radius(x, y, 1) if res2[x_p, y_p] != -1]
            avg = sum(points) / len(points)
            res2[x, y] = avg

plt.imshow(res2)
plt.savefig('data_visual_7.png')
# plt.show()

save_res(res, 'metod3')
save_res(res2, 'metod3.2')

res3 = (res + res2) / 2
plt.imshow(res3)
plt.savefig('data_visual_8.png')

save_res(res3, 'metod3.3_4')