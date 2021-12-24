from network import CNN
import torch.optim as optim
import torch.nn as nn
import torch
import sys 
import numpy as np
from termcolor import colored

sys.path.append("../data")
from load import Dataset
from image_augment import ImageAugmentation, Rescale

from torchmetrics import (Accuracy, AUC, F1, ROC)

# ----------------------------- NETWORK ------------------------------------- #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn_network = CNN()
hyperparams = {
    'optimizer': optim.Adam(cnn_network.parameters(), lr=0.0001),
    'loss': nn.CrossEntropyLoss(),
    'metrics': [Accuracy(), AUC(), F1(), ROC()],
}

cnn_network.configure_hyperparams(hyperparams)
cnn_network.to(device)


# ----------------------------- DATA ------------------------------------- #
generators = {
    'train': ImageAugmentation([Rescale(scale = 1/255)], True),
    'val': ImageAugmentation([Rescale(scale = 1./255)], True),
    'test': ImageAugmentation([Rescale(scale = 1./255)], True),
}
data = Dataset(path="/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/raw/data")
data.to_batches(64, ['train', 'val'])
data_labels = data.labels
training_data, validation_data, testing_data = data.data["train"], data.data["val"], data.data["test"]


for epoch in range(100):  # loop over the dataset multiple times

    i = 0 
    for imgs, labels, names in zip(training_data[0], training_data[1], training_data[2]):
        imgs = imgs / 255 
        cnn_network.train(imgs, labels) 


        if i % 5 == 0 and i > 0:
            correct = 0
              correct += sum([1 for output, test_label in zip(outputs, test_labels)
                                            if torch.argmax(output) == np.argmax(test_label)])
                
                for output, test_label in zip(outputs, test_labels):
                    if torch.argmax(output) == np.argmax(test_label):
                        print("Equal")
                    else:
                        print("NOT EQUAL")
            print(correct)
            print(correct/(len(testing_data[0])*32))

                 
        torch.cuda.empty_cache() 
        i += 1


