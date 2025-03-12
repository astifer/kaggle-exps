import numpy as np
from my_model import MyModel
import yaml
import torch
import pandas as pd

from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

from load_dataset import MyCollator, load as load_datasets
from plots import create_loss_plot, create_metric_plot

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



class Trainer():

    def __init__(self, 
            model, 
            task = "regression", 
            train_dataset = None , 
            val_dataset = None, 
            test_dataset = None,
            loss_fn = None,
            optimizer = None,
            seed: int = 1611,
            params: dict = {}
        ):
        torch.random.manual_seed(seed)
        np.random.seed(seed)

        self.model = model
        self.task = task
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.params = params

        self.train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], collate_fn=MyCollator())
        self.val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], collate_fn=MyCollator())
        self.test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], collate_fn=MyCollator())
        

    def train(self):

        print(f"Training...")

        train_loss_history = []
        eval_loss_history = []
        metric_history = []
        collator = MyCollator()

        for i_epoch in range(self.params['num_epochs']):
            print(f"Epoch {i_epoch} / {self.params['num_epochs']}")

            epoch_train_loss = 0
            with tqdm(total=len(self.train_dataloader)) as pbar:

                for i, train_batch in enumerate(self.train_dataloader):
                    target, inputs = train_batch['target'], train_batch['numeric_features']

                    model_result = self.model(numeric_features=inputs)
                    loss_value = self.loss_fn(model_result, target)
                    loss_value.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    epoch_train_loss += loss_value.item() / self.params['batch_size']

                    pbar.update(1)
                
                train_loss_history.append(epoch_train_loss)
                print(f"Train loss: {epoch_train_loss}")
            

            val_metric, val_loss = self.validate()
            eval_loss_history.append(val_loss)
            metric_history.append(val_metric)

            print(f"Val loss: {val_loss}")
            print(f"Val MSE: {val_metric}")

        create_loss_plot(train_loss_history, name='train_loss')
        create_loss_plot(eval_loss_history, name='val_loss')
        create_metric_plot(metric_history, name='MSE')

        self.make_submission()

    def validate(self):

        self.model.eval()
        eval_loss = 0
        metric = 0

        with torch.no_grad():
            with tqdm(total=len(self.val_dataloader)) as pbar:
                for i, eval_batch in enumerate(self.val_dataloader):
                    target, inputs = eval_batch['target'], eval_batch['numeric_features']
                    
                    model_result = self.model(numeric_features=inputs)
                    loss_value = self.loss_fn(model_result, target)

                    eval_loss += loss_value.item() / self.params['batch_size']

                    metric += mean_squared_error(target.detach().numpy(), model_result.detach().numpy())
                    pbar.update(1)

        eval_loss = eval_loss / len(self.val_dataloader)
        eval_metric = metric / len(self.val_dataloader)

        return eval_metric, eval_loss
    
    def make_submission(self):
        print(f"Making submission...")
        self.model.prediction_mode = True
        labels = []
        with torch.no_grad():
            for i, test_batch in enumerate(self.test_dataloader):
                target, inputs = test_batch['target'], test_batch['numeric_features']
                model_result = self.model(numeric_features=inputs)
                labels.extend([float(x) for x in model_result])
        

        self.model.prediction_mode = False
        df = pd.DataFrame(
            { 
                'index': [i for i in range(len(labels))],
                'Life expectancy': labels
            }
        )

        df.to_csv('submission.csv', index=False)
        print(f"Saved!")


def main():
    params = load_config()

    model = MyModel(hidden_size=params['hidden_size'], drop_p=params['drop_p'])
    train_dataset, val_dataset, test_dataset = load_datasets()

    loss_fn = MSELoss()

    optimizer = Adam( 
        model.parameters(), 
        lr=params['lr'], 
        weight_decay=params['weight_decay'], 
        betas=tuple(params['betas'])
        )
    
    trainer  = Trainer(
        model=model,
        task='regression',
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        loss_fn=loss_fn,
        optimizer=optimizer,
        params=params
    )

    trainer.train()

main()