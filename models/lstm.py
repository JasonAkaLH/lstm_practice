import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, device, num_layers=1, num_direction=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_direction = num_direction
        self.device = device
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, input_seq):
        batch_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]
        # h_0 = torch.randn(self.num_direction * self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        # c_0 = torch.randn(self.num_direction * self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        # output, _ = self.lstm(input_seq, (h_0, c_0))
        output, _ = self.lstm(input_seq)
        pred = self.linear(output)
        pred = pred[:, -1, :]
        return pred
    
