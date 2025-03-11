import torch
from torch import nn, Tensor

class BaseBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.act = nn.LeakyReLU()
        self.linear_2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class MyModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.emb_person_home_ownership = nn.Embedding(4, embedding_dim=hidden_size)
        self.emb_loan_intent = nn.Embedding(6, embedding_dim=hidden_size)
        self.emb_loan_grade = nn.Embedding(7, embedding_dim=hidden_size)
        self.emb_cb_person_default_on_file = nn.Embedding(2, embedding_dim=hidden_size)

        self.numeric_linear = nn.Linear(7, hidden_size)

        self.block = BaseBlock(hidden_size)

        self.linear_out = nn.Linear(hidden_size, 1)

    def forward(self, cat_features: dict[str, Tensor], numeric_features: dict[str, Tensor]) -> Tensor:
        
        x_home = self.emb_person_home_ownership(cat_features['person_home_ownership'])
        x_intent = self.emb_loan_intent(cat_features['loan_intent'])
        x_grade = self.emb_loan_grade(cat_features['loan_grade'])
        x_on_file = self.emb_cb_person_default_on_file(cat_features['cb_person_default_on_file'])

        stacked_numeric = torch.stack([ 
            (numeric_features['person_age'] - 20) / 100, 
            (numeric_features['person_income'] - 4e3) / 1.2e6, 
            numeric_features['person_emp_length'] / 123,
            (numeric_features['loan_amnt'] - 500) / 35000,
            (numeric_features['loan_int_rate'] - 5.4) / 23.3,
            numeric_features['loan_percent_income'],
            (numeric_features['cb_person_cred_hist_length'] - 2)/ 28
            ], dim=-1)
        
        x_numeric = self.numeric_linear(stacked_numeric)

        x_total = x_home + x_intent + x_grade + x_on_file + x_numeric

        x_total = self.block(x_total) 

        result = self.linear_out(x_total)

        result = result.view(-1)

        return result