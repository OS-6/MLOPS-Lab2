import pandas as pd

def get_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df.to_csv("data/titanic_raw.csv", index=False)
    print("✅ Data downloaded and saved to data/titanic_raw.csv")

if __name__ == "__main__":
    get_data()
