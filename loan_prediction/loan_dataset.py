from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

_PERSON_HOME_OWNERSHIP_MAP = {
    "RENT": 0,
    "MORTGAGE": 1,
    "OWN": 2,
    "OTHER": 3
}

_LOAN_INTENT_MAP = {
    "EDUCATION": 0,
    "MEDICAL": 1,
    "VENTURE": 2,
    "PERSONAL": 3,
    "DEBTCONSOLIDATION": 4,
    "HOMEIMPROVEMENT": 5
}

_LOAN_GRADE_MAP = {
    "G": 0,
    "F": 1,
    "E": 2,
    "D": 3,
    "C": 4,
    "B": 5,
    "A": 6
}

_CB_PERSON_DEFAULT_ON_FILE_MAP = {
    "N": 0,
    "Y": 1
}

class LoanDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item: int) -> dict[str, dict[str | Tensor] | Tensor]:
        item = self._data.iloc[item]
        return {
            'target': torch.scalar_tensor(item['loan_status'], dtype=torch.float32),
            'cat_features': {
                'person_home_ownership': torch.scalar_tensor(_PERSON_HOME_OWNERSHIP_MAP[item['person_home_ownership']], dtype=torch.long),
                'loan_intent': torch.scalar_tensor(_LOAN_INTENT_MAP[item['loan_intent']], dtype=torch.long),
                'loan_grade': torch.scalar_tensor(_LOAN_GRADE_MAP[item['loan_grade']], dtype=torch.long),
                'cb_person_default_on_file': torch.scalar_tensor(_CB_PERSON_DEFAULT_ON_FILE_MAP[item['cb_person_default_on_file']], dtype=torch.long),
            },
            'numeric_features': {
                'person_age': torch.scalar_tensor(item['person_age'], dtype=torch.float32),
                'person_income': torch.scalar_tensor(item['person_income'], dtype=torch.float32),
                'person_emp_length': torch.scalar_tensor(item['person_emp_length'], dtype=torch.float32),
                'loan_amnt': torch.scalar_tensor(item['loan_amnt'], dtype=torch.float32),
                'loan_int_rate': torch.scalar_tensor(item['loan_int_rate'], dtype=torch.float32),
                'loan_percent_income': torch.scalar_tensor(item['loan_percent_income'], dtype=torch.float32),
                'cb_person_cred_hist_length': torch.scalar_tensor(item['cb_person_cred_hist_length'], dtype=torch.float32)
            }
        }


class LoanCollator:
    def __call__(self, items: list[dict[str, dict[str | Tensor] | Tensor]]) -> dict[str, dict[str | Tensor] | Tensor]:
        return {
            'target': torch.stack([x['target'] for x in items]),
            'cat_features': {
                'person_home_ownership': torch.stack([x['cat_features']['person_home_ownership'] for x in items]),
                'loan_intent': torch.stack([x['cat_features']['loan_intent'] for x in items]),
                'loan_grade': torch.stack([x['cat_features']['loan_grade'] for x in items]),
                'cb_person_default_on_file': torch.stack([x['cat_features']['cb_person_default_on_file'] for x in items])
            },
            'numeric_features': {
                'person_age': torch.stack([x['numeric_features']['person_age'] for x in items]),
                'person_income': torch.stack([x['numeric_features']['person_income'] for x in items]),
                'person_emp_length': torch.stack([x['numeric_features']['person_emp_length'] for x in items]),
                'loan_amnt': torch.stack([x['numeric_features']['loan_amnt'] for x in items]),
                'loan_int_rate': torch.stack([x['numeric_features']['loan_int_rate'] for x in items]),
                'loan_percent_income': torch.stack([x['numeric_features']['loan_percent_income'] for x in items]),
                'cb_person_cred_hist_length': torch.stack([x['numeric_features']['cb_person_cred_hist_length'] for x in items])
            }
        }


def load_loan() -> tuple[LoanDataset, LoanDataset]:
    train_data = pd.read_csv("data/loan_train.csv")
    val_data = pd.read_csv("data/loan_test.csv")

    t_dtst = LoanDataset(train_data)
    v_dtst = LoanDataset(val_data)
    return t_dtst, v_dtst