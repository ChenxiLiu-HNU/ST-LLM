import pandas as pd

# 读取.pkl文件
pkl_file_path = './data/PEMS08/adj_PEMS08.pkl'
data = pd.read_pickle(pkl_file_path)

# 将数据保存为.csv文件
csv_file_path = './data/PEMS08/adj_PEMS08.csv'
data.to_csv(csv_file_path, index=False)

print(f'Conversion complete. Data saved to {csv_file_path}')
