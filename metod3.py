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
        points = [arr[x_p, y_p] for x_p, y_p in iter_radius(x_r, y_r, 3) if arr[x_p, y_p] != -1]
        # print(points)
        avg = sum(points) / len(points)
        res[x_r, y_r] = avg
    res[x, y] = point.NTG

plt.imshow(res)
plt.show()

for y in range(size_y):
    for x in range(size_x):
        if res[x, y] == -1:
            points = [res[x_p, y_p] for x_p, y_p in iter_radius(x, y, 1) if res[x_p, y_p] != -1]
            avg = sum(points) / len(points)
            res[x, y] = avg

plt.imshow(res)
plt.savefig('data_visual_6.png')
plt.show()        
save_res(res, 'metod3')