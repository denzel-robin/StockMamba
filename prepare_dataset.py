import torch.utils.data as utils
import numpy as np
import torch

def PrepareDataset(historical_data, BATCH_SIZE=48, seq_len=12, pred_len=12, train_propotion=0.7, valid_propotion=0.1):
    time_len = historical_data.shape[0]

    # MinMax Normalization Method.
    for col in historical_data.columns:
      col_max = historical_data[col].max()
      col_min = historical_data[col].min()
      historical_data[col] = (historical_data[col] - col_min) / (col_max - col_min)


    price_sequences, price_labels = [], []
    for i in range(time_len - seq_len - pred_len):
        price_sequences.append(historical_data.iloc[i:i + seq_len].values)
        price_labels.append(historical_data.iloc[i + seq_len:i + seq_len + pred_len].values)
    price_sequences, price_labels = np.asarray(price_sequences), np.asarray(price_labels)

    # Reshape labels to have the same second dimension as the sequences
    price_labels = price_labels.reshape(price_labels.shape[0], seq_len, -1)

    # shuffle & split the dataset to training and testing sets
    sample_size = price_sequences.shape[0]
    index = np.arange(sample_size, dtype=int)
    np.random.shuffle(index)

    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))

    train_data, train_label = price_sequences[:train_index], price_labels[:train_index]
    valid_data, valid_label = price_sequences[train_index:valid_index], price_labels[train_index:valid_index]
    test_data, test_label = price_sequences[valid_index:], price_labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader
