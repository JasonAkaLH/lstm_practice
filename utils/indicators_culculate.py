import torch
from torch import nn
import numpy as np


def get_val_loss(model, val_dt, device):
    loss_function = nn.MSELoss().to(device)
    total_loss = 0
    length = val_dt.__len__()
    for (seq, label) in val_dt:
        seq = seq.to(device)
        label = label.to(device)
        with torch.no_grad():
            y_pred = model(seq)
        loss = loss_function(y_pred, label)
        total_loss += loss
    
    return total_loss / length


def get_mape(y, pred):
    res = 0
    for i in range(len(y)):
        mape = np.abs((y - pred) / y) * 100 if y is not 0 else 0
        res += mape
    
    return res / len(y)
