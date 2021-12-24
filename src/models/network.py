import torch.nn as nn
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # ------------------Model------------------ #
        self.conv1 = nn.Conv2d(1, 8, 5, padding='same')
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding='same')
        self.conv3 = nn.Conv2d(16, 32, 5, padding='same')
        self.conv4 = nn.Conv2d(32, 64, 5, padding='same')
        self.conv5 = nn.Conv2d(64, 128, 5, padding='same')
        self.batch_norm = nn.BatchNorm1d(4608)
        self.fc1 = nn.Linear(4608, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 15)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x))) 
        x = self.pool(F.elu(self.conv2(x)))
        x = self.pool(F.elu(self.conv3(x)))
        x = self.pool(F.elu(self.conv4(x)))
        x = self.pool(F.elu(self.conv5(x)))   
        x = torch.flatten(x, 1)
        x = self.batch_norm(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.softmax(self.fc3(x)) 
        return x
    
    def configure_hyperparams(self, hyperparams):
        self.hyperparams = hyperparams

    def train(self, X, y): 
        imgs = torch.tensor(X).float().cuda()
        labels = torch.tensor(y).float().cuda() 
        
        self.hyperparams['optimizer'].zero_grad()
        outputs = self.forward(imgs)  
        loss = self.hyperparams['loss'](outputs, labels)
        loss.backward()
        self.hyperparams['optimizer'].step()
   
    def validation(self, X, y):
        pass

    def evaluate(self, X, y): 
        test_imgs = torch.tensor(X).float().cuda() 
        outputs = self.forward(test_imgs)
         

    

