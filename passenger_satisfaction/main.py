
import numpy as np
from model import MyModel
import yaml
import torch

from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.optim import SGD
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, f1_score

from load_dataset import SatisfactionDataset, MyCollator, load as load_datasets

from matplotlib import pyplot as plt

def create_loss_plot(loss_history, name: str= "Loss"):
    plt.figure(figsize=(10, 6))

    plt.plot(loss_history, label=name, color='blue', linestyle='--', marker='o')


    plt.title('Training and Validation Loss Over Epochs', fontsize=16)
    plt.xlabel(f'Epochs', fontsize=8)
    plt.ylabel('BCE Loss', fontsize=14)

    plt.legend(fontsize=12)

    plt.grid(True)

    plt.savefig(f'plots/loss_history_{name}.png', bbox_inches='tight')


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

    train_dataset, val_dataset, test_dataset = load_datasets()

    collator = MyCollator()

    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], num_workers=1, collate_fn=collator)
    val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], num_workers=1, collate_fn=collator)
    model = MyModel(hidden_size=params["hidden_size"], drop_p=params['drop_p'])

    loss_bce = BCELoss()
    optimizer = SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    params['loss_function'] = str(loss_bce)

    train_loss_history = []
    val_loss_history = []
    metrics_train = []
    metrics_val = []
        
    for i_epoch in range(params['num_epochs']):
        print(f"Epoch {i_epoch}")

        # model.train()
        epoch_train_loss = 0
        metric_epoch = 0
        with tqdm(total=len(train_dataloader)) as pbar:

            for i, train_batch in enumerate(train_dataloader):
                model_result = model(cat_features=train_batch['cat_features'], numeric_features=train_batch['numeric_features'])

                loss_value = loss_bce(model_result, train_batch['target'])
                loss_value.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_train_loss += loss_value.item() / params['batch_size']

                metric_epoch += f1_score(train_batch['target'].detach().numpy(), model_result.round().detach().numpy())

                pbar.update(1)
            
            train_loss_history.append(epoch_train_loss)
            print(f"Train loss: {epoch_train_loss}")
        
        metrics_train.append(metric_epoch/len(train_dataloader))

        epoch_eval_loss = 0
        metric_epoch = 0
        with torch.no_grad():
            with tqdm(total=len(val_dataloader)) as pbar:
                for i, eval_batch in enumerate(val_dataloader):
                    
                    model_result = model(cat_features=eval_batch['cat_features'], numeric_features=eval_batch['numeric_features'])
                    loss_value = loss_bce(model_result, eval_batch['target'])
                    epoch_eval_loss += loss_value.item() / params['batch_size']

                    metric_epoch += f1_score(eval_batch['target'].detach().numpy(), model_result.round().detach().numpy())
                    pbar.update(1)

        metrics_val.append(metric_epoch/len(val_dataloader))

        print(f"F1 val: {metric_epoch/len(val_dataloader)}")

        val_loss_history.append(epoch_eval_loss)
        print(f"Val loss: {epoch_eval_loss}")

    create_loss_plot(train_loss_history, "Loss_train")
    create_loss_plot(val_loss_history, "Loss_val")

    # make predict
    test_dataloader = DataLoader(test_dataset, batch_size=128, num_workers=1, collate_fn=MyCollator(), shuffle=False)
    model.eval()
    with torch.no_grad():
        test_labels = []
        for i, test_batch in enumerate(test_dataloader):
            model_result = model(cat_features=test_batch['cat_features'], numeric_features=test_batch['numeric_features'])
            test_labels.extend([int(x.item()) for x in model_result.view(-1).round()])

    test_dataset.save_prediction(labels=test_labels)

if __name__ == '__main__':
    run()