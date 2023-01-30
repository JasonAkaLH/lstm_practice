import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, item):
        return self.data[item]
    
    def __len__(self):
        return len(self.data)


def load_data(file_path):
    df = pd.read_csv(file_path, encoding='gbk')
    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    return df


def split_dataset(dataset, train_size, test_size):
    train_data = dataset[:int(len(dataset) * train_size)]
    valid_data = dataset[int(len(dataset) * train_size):int(len(dataset) * (train_size + test_size))]
    test_data = dataset[int(len(dataset) * (train_size + test_size)):]
    return train_data, valid_data, test_data


def get_difference(y_data):
    res = np.zeros(len(y_data) - 24)
    for i in range(24, len(y_data)):
        res[i - 24] = y_data[i] - y_data[i - 1]
    return res


def process(x_data, y_data, shuffle, x_min, x_max, y_min, y_max, batch_size, features_size):
    x_data = (x_data - x_min) / (x_max - x_min)
    target = (y_data - y_min) / (y_max - y_min)
    x_data = x_data.values
    seq = []
    for i in range(len(x_data) - features_size):
        train_seq = []
        train_label = []
        for j in range(i, i + features_size):
            x = x_data[j]
            train_seq.append(x)
        train_label.append(target[i])
        train_seq = torch.FloatTensor(train_seq)
        train_label = torch.FloatTensor(train_label).view(-1)
        seq.append((train_seq, train_label))
    
    seq = MyDataset(seq)
    seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
    return seq


def nn_seq_us(dataset: pd.DataFrame, batch_size, target_col, train_size=0.7, test_size=0.15, features_size=7):
    print('data processing start......')
    
    max_values, min_values = dataset.max(axis=0).values, dataset.min(axis=0).values
    target_max, target_min = dataset[target_col].max(), dataset[target_col].min()
    train_data, valid_data, test_data = split_dataset(dataset, train_size, test_size)
    
    train_set = process(train_data, train_data[target_col].values, True, min_values, max_values, target_min, target_max,
                        batch_size, features_size)
    valid_set = process(valid_data, valid_data[target_col].values, True, min_values, max_values, target_min, target_max,
                        batch_size, features_size)
    test_set = process(test_data, test_data[target_col].values, False, min_values, max_values, target_min, target_max,
                       batch_size, features_size)
    return train_set, valid_set, test_set, max_values, min_values, target_max, target_min
