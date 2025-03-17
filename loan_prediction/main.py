import pandas as pd
import numpy as np
from arch import MyModel
import yaml
import torch

from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.optim import SGD
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from loan_dataset import LoanCollator, LoanDataset, load_loan
from plots import create_loss_plot, create_metric_plot, plot_roc_auc

from matplotlib import pyplot as plt

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
    model = MyModel(hidden_size=params["hidden_size"], drop_p=params['drop_p'])

    loss_bce = BCEWithLogitsLoss()
    optimizer = SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    params['loss_function'] = str(loss_bce)
    # params['optimizer'] = str(optimizer)

    train_loss_history = []
    val_loss_history = []
    rocs_train = []
    rocs_val = []
        
    for i_epoch in range(params['num_epochs']):
        print(f"Epoch {i_epoch}")

        # model.train()
        epoch_train_loss = 0
        roc_epoch = 0
        model.train()
        with tqdm(total=len(train_dataloader)) as pbar:

            for i, train_batch in enumerate(train_dataloader):
                # print(train_batch)
                model_result = model(cat_features=train_batch['cat_features'], numeric_features=train_batch['numeric_features'])
                loss_value = loss_bce(model_result, train_batch['target'])
                loss_value.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_train_loss += loss_value.item()

                fpr, tpr, thresholds = roc_curve(train_batch['target'].detach().numpy(), torch.sigmoid(model_result).detach().numpy())
                roc_epoch += auc(fpr, tpr)

                pbar.update(1)
            
            train_loss_history.append(epoch_train_loss)
            print(f"Train loss: {epoch_train_loss}")
        
        rocs_train.append(roc_epoch/len(train_dataloader))

        epoch_eval_loss = 0
        roc_epoch = 0   
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(test_dataloader)) as pbar:
                for i, eval_batch in enumerate(test_dataloader):
                    model_result = model(cat_features=eval_batch['cat_features'], numeric_features=eval_batch['numeric_features'])
                    loss_value = loss_bce(model_result, eval_batch['target'])
                    epoch_eval_loss += loss_value.item()

                    fpr, tpr, thresholds = roc_curve(eval_batch['target'].detach().numpy(), torch.sigmoid(model_result).detach().cpu())
                    roc_epoch += auc(fpr, tpr)
                    pbar.update(1)

        rocs_val.append(roc_epoch/len(test_dataloader))
        print(f"ROC_AUC val: {roc_epoch/len(test_dataloader)}")

        val_loss_history.append(epoch_eval_loss)
        print(f"Val loss: {epoch_eval_loss}")

    create_loss_plot(train_loss_history, val_loss_history, params)
    create_metric_plot(rocs_val, params)

    model.eval()
    with torch.no_grad():
        val_probs = []
        val_labels = []
        for i, eval_batch in enumerate(test_dataloader):
            model_result = model(cat_features=eval_batch['cat_features'], numeric_features=eval_batch['numeric_features'])
            val_probs.extend(torch.sigmoid(model_result).detach().cpu())
            val_labels.extend(eval_batch['target'])

    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    roc_auc = auc(fpr, tpr) 
    plot_roc_auc(fpr, tpr, roc_auc)

if __name__ == '__main__':
    run()