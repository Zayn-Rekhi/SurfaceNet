from torchvision.models import resnext50_32x4d
from torch import nn, optim
from torch.nn import functional as F
import torch
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.learning_rate = hyperparams['lr']        
        self.epochs = hyperparams['epochs']
        self.classes = hyperparams['classes']
        self.hyperparams = hyperparams

        self.net = resnext50_32x4d(pretrained=True)
        if hyperparams['transfer_learning']:
            print("starting transfer learning")
            for param in self.net.parameters():
                param.requires_grad = False

        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)        

        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, hyperparams['classes'])

        self._configure_loss()
        self._configure_optimizer()
        
        self.history = {}

    def _configure_loss(self):
        if self.hyperparams['loss_function'] == "MSE": 
            self.loss_function = nn.MSE()
        elif self.hyperparams['loss_function'] == "CE": 
            self.loss_function = nn.CrossEntropyLoss()

    def _configure_optimizer(self):
        if self.hyperparams['optimizer'] == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.hyperparams['momentum'])
        elif self.hyperparams['optimizer'] == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min')

    def forward(self, x):
        return self.net(x)

    def train_batch(self, X, y):
        self.optimizer.zero_grad()
        output = self.forward(X)
        loss = self.loss_function(output, y)
        loss.backward()
        self.optimizer.step()  
        return loss  
    
    def test_batch(self, X, y):
        output = self.forward(X)
        loss = self.loss_function(output, y)
        return loss, output

    def evaluate(self, y, predictions):  
        for metric in self.hyperparams['metrics']:  
            out = metric[1](y, predictions)
            self.history[metric[0]] = out
            print(f"{metric[0]}...........DONE({out})")
         
        return self.history



def save(PATH, model):
    data = {
        "model_state": model.state_dict(),
        "optimizer_state": model.optimizer.state_dict(),
        "loss_state": model.loss_function.state_dict(),
        "history": model.history,
    }
    torch.save(data, PATH)

def delete(PATH):
    os.remove(PATH)
