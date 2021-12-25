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


        #in_features = self.net.fc.in_features
        #self.net.fc = nn.Linear(in_features, len(self.classes))
        self.history = {}       
        
        self._configure_loss()
        self._configure_optimizer()
        


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
        self.history['loss'] = loss 
    
    def val_batch(self, X, y):
        pass

    def evaluate(self, X, y): 
        for metric in self.hyperparams['metrics']:
            output = self.forward(X)
            measure = metric(output, y)
            self.history[metric.str()] = measure





