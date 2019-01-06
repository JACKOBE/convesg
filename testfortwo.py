import pandas as pd
import numpy as np
import seaborn as sns

# 数据读取
income = pd.read_excel(r'C:\Users\Lenovo\Desktop\income.xlsx')

# 查看数据集是否存在缺失值
print(income.apply(lambda x:np.sum(x.isnull())))