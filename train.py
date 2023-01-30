import copy
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import make_interp_spline
from torch import nn
from torch.optim.lr_scheduler import StepLR

import utils.data_process
from utils.data_process import MyDataset
from models.lstm import LSTM
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller

from utils.indicators_culculate import get_val_loss, get_mape


def train(input_size, hidden_size, num_layers, output_size, batch_size, lr, weight_decay, step_size, gamma, optimizer,
          epochs, train_dt, val_dt, path, compute_device, save_state=True):
    input_size, hidden_size, num_layers = input_size, hidden_size, num_layers
    output_size = output_size
    
    model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                 batch_size=batch_size, device=compute_device).to(compute_device)
    
    loss_function = nn.MSELoss().to(compute_device)
    
    if optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    scheduler = StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    
    for epoch in tqdm(range(epochs)):
        train_loss = []
        model.train()
        for (seq, label) in train_dt:
            seq = seq.to(compute_device)
            label = label.to(compute_device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        model.eval()
        val_loss = get_val_loss(model, val_dt, compute_device)
        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
        
        print(f'epoch: {epoch + 1}, train_loss: {np.mean(train_loss)}, valid_loss: {val_loss}')
    
    state = {f'models: {best_model.state_dict()}'}
    if save_state:
        torch.save(state, path)
    else:
        torch.save(model, path)


def test(test_dt, path, target_max, target_min, compute_device):
    pred = []
    y = []
    print('loading model....')
    model = torch.load(path)
    model.eval()
    
    print('predicting...')
    for (seq, target) in tqdm(test_dt):
        # target = np.array(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(compute_device)
        with torch.no_grad():
            y_pred = model(seq)
            print(y_pred)
            # y_pred = np.array(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)
    
    # y, pred = np.array(y), np.array(pred)
    y = np.array(list(map(lambda res: res.item(), y)))
    pred = np.array(list(map(lambda res: res.item(), pred)))
    y = (target_max - target_min) * y + target_min
    pred = (target_max - target_min) * pred + target_min
    
    print(f"mape: {get_mape(y, pred)}")
    
    plt.plot(y, c='blue', ms=1, alpha=0.75, label='ground truth')
    plt.plot(pred, c='red', ms=1, alpha=0.75, label='prediction')
    plt.grid(axis='y')
    plt.legend()
    plt.show()
