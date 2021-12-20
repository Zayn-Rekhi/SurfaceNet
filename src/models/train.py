from network import CNN
import torch.optim as optim
import torch.nn as nn
import torch
import sys 
import numpy as np
from torch.utils import SummaryWriter

sys.path.append("../data")
from load import Dataset


writer = SummaryWriter('../../logs/run')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cnn_network = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_network.parameters(), lr=0.0001, momentum=0.9)

data = Dataset(path="/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/raw/data")
data.to_batches(300)
training_data = data.data["train"]

for epoch in range(2):  # loop over the dataset multiple times

    i = 0 
    for img, label in zip(training_data[0], training_data[1]):

        imgs, labels = np.asarray(img, np.uint8), np.asarray(label, np.uint8)  
        
        labels_ = []
        for label in labels: 
            labels_.append(np.argmax(label))
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn_network(torch.tensor(imgs).float()) 
        loss = criterion(outputs, torch.tensor(labels_))
        loss.backward()
        optimizer.step()
        
        print(loss.item())




