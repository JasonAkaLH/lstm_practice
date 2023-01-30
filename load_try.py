import numpy as np
import pandas as pd
import torch

from utils.data_process import MyDataset, nn_seq_us, get_difference
from train import test

if __name__ == '__main__':
    model = torch.load('./saved_model/test.pth')
    print(model)
    model.eval()
    # seq = MyDataset(np.random.random([24, 3]))
    data = pd.read_csv('./data/DailyDelhiClimateTest.csv')
    _, _, test_dt, _, _, target_max, target_min = nn_seq_us(data, 64, 'LT001', features_size=24)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(test_dt=test_dt, path='./saved_model/test.pth', target_max=target_max, target_min=target_min,
         compute_device=device)
