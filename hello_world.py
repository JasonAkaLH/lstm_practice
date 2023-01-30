import torch
import pandas as pd
import os

print("hello world")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
data = pd.read_csv(f'../fishing_data/000004.SZ.csv')
print(data)