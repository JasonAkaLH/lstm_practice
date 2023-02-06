import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from load_data_v2 import get_data, TrainSet
from models.lstm import LSTM_v2
from tqdm import tqdm


def train(_train_loader, _test_loader, input_size, hidden_size, num_layers, output_size, step_size, gamma, epochs):
    model = LSTM_v2(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    loss_function = nn.MSELoss()
    is_cuda = torch.cuda.is_available()
    
    if is_cuda:
        model = model.cuda()
        loss_function = loss_function.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    step_lr = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    for epoch in tqdm(range(epochs)):
        model.train()
        for tx, ty in _train_loader:
            if is_cuda:
                tx = tx.cuda()
                ty = ty.cuda()
            
            output = model(torch.unsqueeze(tx, dim=2))
            loss = loss_function(torch.squeeze(output), ty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        step_lr.step()
        print(
            f"epoch: {epoch + 1}, train_loss: {loss.cpu()}")
    
    return model


if __name__ == '__main__':
    stock_name = '000004.SZ'
    data = pd.read_csv(f'../fishing_data/{stock_name}.csv')
    
    train_end = -500
    train_data, test_data = get_data(data, 'Close', train_end=train_end, return_all=False)
    print(train_data)
    print(test_data)
    
    train_data_np = np.array(train_data)
    train_mean = np.mean(train_data_np)
    train_std = np.std(train_data_np)
    train_data_np = (train_data_np - train_mean) / train_std
    test_data_np = (np.array(test_data) - train_mean) / train_std
    
    train_tensor = torch.Tensor(train_data_np)
    test_tensor = torch.Tensor(test_data_np)
    
    train_loader = DataLoader(TrainSet(train_tensor), batch_size=32, shuffle=True)
    test_loader = DataLoader(TrainSet(test_tensor), batch_size=32, shuffle=False)
    
    for x, y in test_loader:
        print(x)
        print(y)
    # final_model = train(train_loader, test_loader, input_size=1, hidden_size=64, num_layers=1, output_size=1,
    #                     step_size=50, gamma=0.1, epochs=500)
