import torch
import torch.nn as nn

def model_load(input_size=1, hidden_size=64, num_layers=2):
# actually fetch from drive
    class PriceLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, (hidden, cell) = self.lstm(x)

            last_time_step = out[:, -1, :]

            prediction = self.fc(last_time_step)

            return prediction
    model = PriceLSTM(input_size, hidden_size, num_layers)
    return model

