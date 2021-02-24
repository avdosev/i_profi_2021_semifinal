import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

p = np.zeros((300, 1000))

for i, point in train.iterrows():
    p[point.X, point.Y] = 1

for i, point in test.iterrows():
    p[point.X, point.Y] = 3

plt.imshow(p[201:247, 901:930])
plt.savefig('data_visual.png')