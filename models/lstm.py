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
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True).to(self.device)
        self.linear = nn.Linear(self.hidden_size, self.output_size).to(self.device)
    
    def forward(self, input_seq):
        batch_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]
        h_0 = torch.randn(self.num_direction * self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_direction * self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        # output, _ = self.lstm(input_seq, (h_0, c_0))
        output, (hn, cn) = self.lstm(input_seq)
        pred = self.linear(output)
        pred = pred[:, -1, :]
        return pred


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, device, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = device
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * self.hidden_size, self.output_size)
    
    def forward(self, input_seq):
        h_0 = torch.randn(2 * self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(2 * self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        # print(input_seq.size())
        seq_len = input_seq.shape[1]
        # input(batch_size, seq_len, input_size)
        input_seq = input_seq.view(self.batch_size, seq_len, self.input_size)
        # print(f'input size: {input_seq.shape}')
        # print(f'seq len: {seq_len}, batch size: {self.batch_size}, embedding size: {self.input_size}')
        # output(batch_size, seq_len, num_directions * hidden_size)
        # output, _ = self.lstm(input_seq, (h_0, c_0))
        output, _ = self.lstm(input_seq)
        # output = output.contiguous().view(self.batch_size, seq_len, 2, self.hidden_size)
        # output = torch.mean(output, dim=2)
        pred = self.linear(output)
        # print('pred=', pred.shape)
        pred = pred[:, -1, :]
        
        return pred


class LSTM_v2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_v2, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        self.out = nn.Sequential(nn.Linear(self.hidden_size, self.output_size))
    
    def forward(self, x):
        r_out, (h_n, c_n) = self.lstm(x, None)
        out = self.out(r_out[:, -1, :])
        
        return out
