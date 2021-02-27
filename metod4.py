import pandas as pd

def stack_results(names):
    data = list(map(pd.read_csv, names))
    s = data[0]['NTG']
    for c in data[1:]:
        s += c['NTG']
    s /= len(data)
    res = pd.read_csv(names[0])
    res['NTG'] = s
    return res

def save_res(df, name):
    df.to_csv(f'./output/{name}.csv', index=False)

# 27 фев 2021, 16:27:04
save_res(stack_results([
    'output/neighbors 2 distance.csv',
    'output/neighbors 8 distance.csv',
    'output/neighbors 10 distance.csv',
]), 'metod4_1')

# 27 фев 2021, 16:28:21
save_res(stack_results([
    'output/neighbors 2 distance.csv',
    'output/polinom 4 linear.csv',
]), 'metod4_2')

# 27 фев 2021, 16:29:33
save_res(stack_results([
    'output/neighbors 2 distance.csv',
    'output/neighbors 8 distance.csv',
    'output/neighbors 10 distance.csv',
    'output/rbf_linear.csv',
    'output/rbf_multiquadric.csv',
    'output/nearest_interpolator.csv',
    'output/metod3.3_4.csv'
]), 'metod4_3')

# 27 фев 2021, 16:30:39
save_res(stack_results([
    'output/neighbors 2 distance.csv',
    'output/rbf_linear.csv',
    'output/rbf_multiquadric.csv',
    'output/nearest_interpolator.csv',
]), 'metod4_4')