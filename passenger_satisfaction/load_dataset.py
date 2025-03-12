from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

_MARK_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5
}

_CLASS_MAP = {
    "Business": 0,
    "Economy": 1,
    "Economy Plus": 2
}

_TRAVEL_TYPE_MAP = {
    "Business": 0,
    "Personal": 1
}

_CUSTOMER_TYPE_MAP = {
    "Returning": 0,
    "First-time": 1
}

_GENDER_MAP = {
    "Female": 0,
    "Male": 1
}

_TARGET_MAP = {
    "Satisfied": 1,
    "Neutral or Dissatisfied": 0,
    "None": -1
}

class SatisfactionDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item: int) -> dict[str, dict[str | Tensor] | Tensor]:
        item = self._data.iloc[item]
        return {
            'target': torch.scalar_tensor(_TARGET_MAP[item.get('Satisfaction', "None")], dtype=torch.float32),
            'cat_features': {
                'Gender': torch.scalar_tensor(_GENDER_MAP[item['Gender']], dtype=torch.long),
                'Customer Type': torch.scalar_tensor(_CUSTOMER_TYPE_MAP[item['Customer Type']], dtype=torch.long),
                'Type of Travel': torch.scalar_tensor(_TRAVEL_TYPE_MAP[item['Type of Travel']], dtype=torch.long),
                'Class': torch.scalar_tensor(_CLASS_MAP[item['Class']], dtype=torch.long),
                'Departure and Arrival Time Convenience': torch.scalar_tensor(_MARK_MAP[item['Departure and Arrival Time Convenience']], dtype=torch.long),
                'Ease of Online Booking': torch.scalar_tensor(_MARK_MAP[item['Ease of Online Booking']], dtype=torch.long),
                'Check-in Service': torch.scalar_tensor(_MARK_MAP[item['Check-in Service']], dtype=torch.long),
                'Online Boarding': torch.scalar_tensor(_MARK_MAP[item['Online Boarding']], dtype=torch.long),
                'Gate Location': torch.scalar_tensor(_MARK_MAP[item['Gate Location']], dtype=torch.long),
                'On-board Service': torch.scalar_tensor(_MARK_MAP[item['On-board Service']], dtype=torch.long),
                'Seat Comfort': torch.scalar_tensor(_MARK_MAP[item['Seat Comfort']], dtype=torch.long),
                'Leg Room Service': torch.scalar_tensor(_MARK_MAP[item['Leg Room Service']], dtype=torch.long),
                'Cleanliness': torch.scalar_tensor(_MARK_MAP[item['Cleanliness']], dtype=torch.long),
                'Food and Drink': torch.scalar_tensor(_MARK_MAP[item['Food and Drink']], dtype=torch.long),
                'In-flight Service': torch.scalar_tensor(_MARK_MAP[item['In-flight Service']], dtype=torch.long),
                'In-flight Wifi Service': torch.scalar_tensor(_MARK_MAP[item['In-flight Wifi Service']], dtype=torch.long),
                'In-flight Entertainment': torch.scalar_tensor(_MARK_MAP[item['In-flight Entertainment']], dtype=torch.long),
                'Baggage Handling': torch.scalar_tensor(_MARK_MAP[item['Baggage Handling']], dtype=torch.long)
            },
            'numeric_features': {
                'Age': torch.scalar_tensor(item['Age'], dtype=torch.float32),
                'Flight Distance': torch.scalar_tensor(item['Flight Distance'], dtype=torch.float32),
                'Departure Delay': torch.scalar_tensor(item['Departure Delay'], dtype=torch.float32),
                'Arrival Delay': torch.scalar_tensor(item['Arrival Delay'], dtype=torch.float32)
            }
        }
    def save_prediction(self, labels):
        pd_result = pd.DataFrame(
            { 
                "ID": self._data.loc[:, "ID"],
                "Satisfaction": labels
            }
        )
        pd_result.to_csv("submission.csv", index=False)

class MyCollator:
    def __call__(self, items: list[dict[str, dict[str | Tensor] | Tensor]]) -> dict[str, dict[str | Tensor] | Tensor]:
        return {
            'target': torch.stack([x.get('target', None) for x in items]),
            'cat_features': {
                'Gender': torch.stack([x['cat_features']['Gender'] for x in items]),
                'Customer Type': torch.stack([x['cat_features']['Customer Type'] for x in items]),
                'Type of Travel': torch.stack([x['cat_features']['Type of Travel'] for x in items]),
                'Class': torch.stack([x['cat_features']['Class'] for x in items]),
                'Departure and Arrival Time Convenience': torch.stack([x['cat_features']['Departure and Arrival Time Convenience'] for x in items]),
                'Ease of Online Booking': torch.stack([x['cat_features']['Ease of Online Booking'] for x in items]),
                'Check-in Service': torch.stack([x['cat_features']['Check-in Service'] for x in items]),
                'Online Boarding': torch.stack([x['cat_features']['Online Boarding'] for x in items]),
                'Gate Location': torch.stack([x['cat_features']['Gate Location'] for x in items]),
                'On-board Service': torch.stack([x['cat_features']['On-board Service'] for x in items]),
                'Seat Comfort': torch.stack([x['cat_features']['Seat Comfort'] for x in items]),
                'Leg Room Service': torch.stack([x['cat_features']['Leg Room Service'] for x in items]),
                'Cleanliness': torch.stack([x['cat_features']['Cleanliness'] for x in items]),
                'Food and Drink': torch.stack([x['cat_features']['Food and Drink'] for x in items]),
                'In-flight Service': torch.stack([x['cat_features']['In-flight Service'] for x in items]),
                'In-flight Wifi Service': torch.stack([x['cat_features']['In-flight Wifi Service'] for x in items]),
                'In-flight Entertainment': torch.stack([x['cat_features']['In-flight Entertainment'] for x in items]),
                'Baggage Handling': torch.stack([x['cat_features']['Baggage Handling'] for x in items])
            },
            'numeric_features': {
                'Age': torch.stack([x['numeric_features']['Age'] for x in items]),
                'Flight Distance': torch.stack([x['numeric_features']['Flight Distance'] for x in items]),
                'Departure Delay': torch.stack([x['numeric_features']['Departure Delay'] for x in items]),
                'Arrival Delay': torch.stack([x['numeric_features']['Arrival Delay'] for x in items])
            }
        }


def load() -> tuple[SatisfactionDataset, SatisfactionDataset, SatisfactionDataset]:
    numeric_features = [
        "Age",
        "Flight Distance",
        "Departure Delay",
        "Arrival Delay",
    ]
    
    data = pd.read_csv("data/train.csv")
    stat = data.loc[:, numeric_features].describe()

    for col in numeric_features:
        data[col] = data[col].apply(lambda x: (x - stat[col]['mean'])/ stat[col]['std'])

    data = data.fillna(0)

    train_data, val_data = train_test_split(data, test_size=0.05)
    test_data = pd.read_csv("data/test.csv")

    for col in numeric_features:
        test_data[col] = test_data[col].apply(lambda x: (x - stat[col]['mean'])/ stat[col]['std'])

    test_data = test_data.fillna(0)

    
    train_dtst = SatisfactionDataset(train_data)
    val_dtst = SatisfactionDataset(val_data)
    test_dtst = SatisfactionDataset(test_data)

    return train_dtst, val_dtst, test_dtst