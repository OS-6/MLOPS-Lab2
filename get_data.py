import pandas as pd
import os

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

os.makedirs('data', exist_ok=True)

df.to_csv('data/data_raw.csv', index=False)