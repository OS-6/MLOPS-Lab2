import pandas as pd
import os

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

os.makedirs('data', exist_ok=True)

df.to_csv('data/data_raw.csv', index=False)