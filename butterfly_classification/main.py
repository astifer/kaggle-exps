import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Trainer():
    def __init__(self,
                 config,
                 transform = None,
                 model = None,
                 loss_function = nn.CrossEntropyLoss(),
                 schedule = True
        ):
        self.config = config
        self.transform = transform

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.config['lr'])

        if schedule:
            self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.loss_function = loss_function

        self.setup_dataloders()

    def setup_dataloders(self):
        pass

    def run(self):
        for epoch in range(self.config['num_epochs']):
            train_loss, train_acc = self.train_step()
            val_loss, val_acc = self.validate()

            print(f"Epoch {epoch + 1} / {self.config['num_epochs']}, train loss {train_loss:.4f}, train accuracy {train_acc:.4f} \
                    val loss {val_loss:.4f}, val accuracy {val_acc:.4f}")

        test_loss, test_acc = self.test()
        print(f"Test loss {test_loss:.4f}, test accuracy {test_acc:.4f}")

    def train_step(self):
        self.model.train()

        total_loss = 0.0
        correct = 0
        total_samples = 0

        for images, labels in tqdm(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.loss_function(outputs, labels)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        self.scheduler.step()

        total_loss = total_loss / len(self.train_loader)
        train_acc = correct / total_samples * 100

        return total_loss, train_acc

    def validate(self):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)

                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        total_loss = total_loss / len(self.val_loader)
        val_acc = correct / total_samples * 100

        return total_loss, val_acc

    def test(self):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)

                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        total_loss = total_loss / len(self.test_loader)
        test_acc = correct / total_samples * 100

        return total_loss, test_acc
    

if __name__ == '__main__':

    trainer = Trainer()

    trainer.run()