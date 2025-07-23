import pandas as pd

data = pd.read_csv("dataset.csv")
data["ticker"].unique()
for ticker in data["ticker"].unique():
    data[data["ticker"] == ticker].to_csv(f"dataset/{ticker}.csv", index=False)
