import pandas as pd
from torch.utils.data import Dataset


def get_data(df, column, train_end=-250, days_before=7, return_all=True, generate_index=False):
    series = df[column].copy()
    
    # split data
    train_series, test_series = series[:train_end], series[train_end - days_before:]
    
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for i in range(days_before):
        train_data[f't_{i}'] = train_series.tolist()[i:-days_before + i]
        test_data[f't_{i}'] = test_series.tolist()[i:-days_before + i]
    
    train_data['label'] = train_series.tolist()[days_before:]
    test_data['label'] = test_series.tolist()[days_before:]
    
    if generate_index:
        train_data.index = train_series.index[days_before:]
    
    if return_all:
        return train_data, test_data, series, df.index.tolist()
    
    return train_data, test_data


# build dataloader
class TrainSet(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.data)
