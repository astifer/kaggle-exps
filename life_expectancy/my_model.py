import torch
from torch import nn, Tensor

class BaseBlock(nn.Module):
    def __init__(self, hidden_size: int, drop_p: float):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_size)
        self.do = nn.Dropout1d(p=drop_p)
        self.linear_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.act = nn.LeakyReLU()
        self.linear_2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.do(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.act(x)

        return x


class MyModel(nn.Module):
    def __init__(self, hidden_size: int, drop_p: float):
        super().__init__()
        self.numeric_linear = nn.Linear(19, hidden_size)

        self.block1 = BaseBlock(hidden_size, drop_p)
        self.block2 = BaseBlock(hidden_size, drop_p)
        self.block3 = BaseBlock(hidden_size, drop_p)

        self.linear_out = nn.Linear(hidden_size, 1)
        self.prediction_mode = False

    def forward(self, numeric_features: Tensor) -> Tensor:
    
        x_total = self.numeric_linear(numeric_features)
        
        x_total = self.block1(x_total) + x_total
        x_total = self.block2(x_total) + x_total
        x_total = self.block3(x_total) + x_total

        result = self.linear_out(x_total)

        result = result.view(-1)

        if self.prediction_mode:
            result = torch.clip(result, 20, 100)

        return result