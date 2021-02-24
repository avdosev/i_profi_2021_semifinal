import pandas as pd

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

print('X uniq:')
t1 = train.X.unique()
t2 = test.X.unique()
print(set(t1) - set(t2))

print('Y uniq:')
t1 = train.Y.unique()
t2 = test.Y.unique()
print(set(t1) - set(t2))