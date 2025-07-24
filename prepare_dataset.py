import pandas as pd
import numpy as np
import torch

df = pd.read_csv("dataset.csv")
df = df[["ticker" ,"date", "open", "high", "low", "close"]].copy()

# convert ticker to integer

ticker_id, unique_tickers = pd.factorize(df["ticker"])
df['ticker'] = ticker_id

# convert dates to integer(ordinal)

df["date"] = pd.to_datetime(df["date"])
df["date"] = df["date"].apply(lambda x: x.toordinal()).values

df_array = df.to_numpy()

grouped = [group.iloc[:227].to_numpy() for _, group in df.groupby("ticker")]
df_div = np.stack(grouped)

X = [[[] for _ in range(227)] for _ in range(47)]
Y = [[[] for _ in range(227)] for _ in range(47)]

for i in range(47):
    for j in range(227):
        row = df_div[i][j]
        X[i][j].append([row[0], row[1]])
        Y[i][j].append([row[2], row[3], row[4], row[5]])
X = np.array(X).squeeze()
Y = np.array(Y).squeeze()

x_train, y_train = [],[]
x_test, y_test = [],[]

for i in range(47):
  split = int(len(X[i])*0.8)
  x_train.append(X[i][:split])
  y_train.append(Y[i][:split])
  x_test.append(X[i][split:])
  y_test.append(Y[i][split:])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).float()


