import pandas as pd
import numpy as np
from arch import MyModel
import yaml
import torch

from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from tqdm import tqdm

from loan_dataset import LoanCollator, LoanDataset, load_loan

from matplotlib import pyplot as plt
from hashlib import md5

cat_features = [ 
        "person_home_ownership", 
        "loan_intent", 
        "loan_grade", 
        "cb_person_default_on_file"
        ]
    
num_features = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        params = config.get("hyperparameters", {})
        if not params:
            print(f"Fail to load config")
            return {}
        else:
            print(f"Train model with following config:\n{params}")
    return params

def create_plot(train_loss_history, val_loss_history, params):
    plt.figure(figsize=(10, 6))

    plt.plot(train_loss_history, label='Training Loss', color='blue', linestyle='--', marker='o')

    plt.plot(val_loss_history, label='Validation Loss', color='red', linestyle='-', marker='x')

    plt.title('Training and Validation Loss Over Epochs', fontsize=16)
    plt.xlabel(f'Config: {params}', fontsize=8)
    plt.ylabel('Loss', fontsize=14)

    _hash = md5(f"{params}".encode()).hexdigest()
    plt.legend(fontsize=12)

    plt.grid(True)

    plt.savefig(f'experiments/loss_history_{_hash}.png', bbox_inches='tight')
    plt.show()


def run():
    '''
    Train, valid, test loop
    '''

    # load config with parameters
    params = load_config()

    torch.random.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    train_dataset, test_dataset = load_loan()

    collator = LoanCollator()

    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], num_workers=1, collate_fn=collator)
    test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], num_workers=1, collate_fn=collator)
    model = MyModel(hidden_size=params["hidden_size"])

    loss_bce = BCEWithLogitsLoss()
    optimizer = SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    train_loss_history = []
    val_loss_history = []

    for i_epoch in range(params['num_epochs']):
        print(f"Epoch {i_epoch}")

        model.train()
        epoch_train_loss = 0

        with tqdm(total=len(train_dataloader)) as pbar:

            for i, train_batch in enumerate(train_dataloader):
                model_result = model(cat_features=train_batch['cat_features'], numeric_features=train_batch['numeric_features'])
                optimizer.zero_grad()
                loss_value = loss_bce(model_result, train_batch['target'])
                loss_value.backward()
                optimizer.step()

                epoch_train_loss += loss_value.item() / params['batch_size']
                pbar.update(1)
            
            train_loss_history.append(epoch_train_loss)
            print(f"Train loss: {epoch_train_loss}")

        model.eval()
        epoch_eval_loss = 0
    
        with torch.no_grad():
            with tqdm(total=len(test_dataloader)) as pbar:
                for i, eval_batch in enumerate(test_dataloader):
                    model_result = model(cat_features=eval_batch['cat_features'], numeric_features=eval_batch['numeric_features'])
                    loss_value = loss_bce(model_result, eval_batch['target'])
                    epoch_eval_loss += loss_value.item() / params['batch_size']
                    pbar.update(1)

        val_loss_history.append(epoch_eval_loss)
        print(f"Val loss: {epoch_eval_loss}")

    create_plot(train_loss_history, val_loss_history, params)

if __name__ == '__main__':
    run()