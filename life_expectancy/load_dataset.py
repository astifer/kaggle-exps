from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class LifeDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        item = self._data.iloc[item]

        return {
            'target': torch.scalar_tensor(item.get('Life expectancy ', -1), dtype=torch.float32),
            'numeric_features': 
                torch.cat(
                [torch.tensor([item['Year']], dtype=torch.float32),
                torch.tensor([item['Adult Mortality']], dtype=torch.float32),
                torch.tensor([item['infant deaths']], dtype=torch.float32),
                torch.tensor([item['Alcohol']], dtype=torch.float32),
                torch.tensor([item['percentage expenditure']], dtype=torch.float32),
                torch.tensor([item['Hepatitis B']], dtype=torch.float32),
                torch.tensor([item['Measles ']], dtype=torch.float32),
                torch.tensor([item[' BMI ']], dtype=torch.float32),
                torch.tensor([item['under-five deaths ']], dtype=torch.float32),
                torch.tensor([item['Polio']], dtype=torch.float32),
                torch.tensor([item['Total expenditure']], dtype=torch.float32),
                torch.tensor([item['Diphtheria ']], dtype=torch.float32),
                torch.tensor([item[' HIV/AIDS']], dtype=torch.float32),
                torch.tensor([item['GDP']], dtype=torch.float32),
                torch.tensor([item['Population']], dtype=torch.float32),
                torch.tensor([item[' thinness  1-19 years']], dtype=torch.float32),
                torch.tensor([item[' thinness 5-9 years']], dtype=torch.float32),
                torch.tensor([item['Income composition of resources']], dtype=torch.float32),
                torch.tensor([item['Schooling']], dtype=torch.float32)]
                )
        }


class MyCollator:
    def __call__(self, items) -> dict[str, Tensor]:
        return {
            'target': torch.stack([x.get('target', -1) for x in items]),
            'numeric_features': torch.stack([x['numeric_features'] for x in items])
        }


def load() -> tuple[LifeDataset, LifeDataset, LifeDataset]:
    numeric_features = ['Year', 'Adult Mortality', 'infant deaths',
       'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
       'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
       ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
       ' thinness 5-9 years', 'Income composition of resources', 'Schooling']
    
    data = pd.read_csv("data/Life_train.csv")
    stat = data.loc[:, numeric_features].describe()

    for col in numeric_features:
        data[col] = data[col].apply(lambda x: (x - stat[col]['min'])/(stat[col]['max'] - stat[col]['min']) )

    data = data.fillna(0)

    train_data, val_data = train_test_split(data, test_size=0.01)
    test_data = pd.read_csv("data/Life_test.csv")

    for col in numeric_features:
        test_data[col] = test_data[col].apply(lambda x: (x - stat[col]['min'])/(stat[col]['max'] - stat[col]['min']) )

    test_data = test_data.fillna(0)

    
    train_dtst = LifeDataset(train_data)
    val_dtst = LifeDataset(val_data)
    test_dtst = LifeDataset(test_data)

    return train_dtst, val_dtst, test_dtst