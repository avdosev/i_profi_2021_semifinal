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


spline = sp.interpolate.Rbf(X.X,X.Y,y,function='thin_plate',smooth=5, episilon=5)
res = spline(B1, B2)
plt.imshow(res)
plt.savefig('data_visual_2.png')
save_res(res, 'rbf')

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