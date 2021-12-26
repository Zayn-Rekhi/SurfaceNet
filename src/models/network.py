from torchvision.models import resnext50_32x4d
from torch import nn, optim
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.learning_rate = hyperparams['lr']        
        self.epochs = hyperparams['epochs']
        self.classes = hyperparams['classes']
        self.hyperparams = hyperparams

        self.net = resnext50_32x4d(num_classes=15)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)        

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
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.hyperparams['optimizer'] == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        return self.net(x)

    def train_batch(self, X, y):
        self.optimizer.zero_grad()
        output = self.forward(X)
        loss = self.loss_function(output, y)
        loss.backward()
        self.optimizer.step() 
        return loss  

    def evaluate(self, y, predictions): 

        for metric in self.hyperparams['metrics']: 
            print(f"{metric[0]}...........DONE")
            out = metric[1](y, predictions)
            self.history[metric[0]] = out
        
        self.history['loss'] = sum([F.cross_entropy(prediction, label) for prediction, label in zip(predictions, y)])/len(y)
        return self.history




