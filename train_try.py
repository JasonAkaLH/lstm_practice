import pandas as pd
import torch
import os

from utils.data_process import nn_seq_us
from train import train, test

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # data = pd.read_csv('./data/DailyDelhiClimateTrain.csv')
    # # print(data)
    # data = data[['meantemp', 'humidity', 'wind_speed', 'meanpressure']]
    stock_name = '000004.SZ'
    data = pd.read_csv(f'../fishing_data/{stock_name}.csv')
    data.set_index('Date', inplace=True)
    train_dt, valid_dt, test_dt, _, _, y_max, y_min = nn_seq_us(data, 64, 'Close', features_size=100)
    print(train_dt.dataset.__len__())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model_path = './saved_model/fishing_path.pth'
    
    train(input_size=len(data.columns), hidden_size=32, num_layers=1, output_size=1, batch_size=64, lr=0.001,
          weight_decay=0.01, step_size=30, gamma=0.1, optimizer='Adam', epochs=300, train_dt=train_dt, val_dt=valid_dt,
          path=model_path, compute_device=device, save_state=False, bidirectional=True)
    
    test(test_dt=test_dt, path=model_path, target_max=y_max, target_min=y_min, compute_device=device)
