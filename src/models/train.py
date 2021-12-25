from network import Model, DEVICE
import torch.optim as optim
import torch.nn as nn
import torch
import sys 
import numpy as np

sys.path.append("../data")
from load import Dataset
from image_augment import ImageAugmentation, Rescale

from torchmetrics import (Accuracy, AUC, F1, ROC)
import wandb

# ----------------------------- DATA LOADING ------------------------------------- #
generators = {
    'train': ImageAugmentation([Rescale(scale = 1/255)], True),
    'val': ImageAugmentation([Rescale(scale = 1./255)], True),
    'test': ImageAugmentation([Rescale(scale = 1./255)], True),
}
data = Dataset(path="/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/raw/data")
data.to_batches(64, ['train', 'val'])
data_labels = data.labels
training_data, validation_data, testing_data = data.data["train"], data.data["val"], data.data["test"]


# ----------------------------- NETWORK ------------------------------------- #
hyperparams = {
    'lr': 0.001,
    'batch_size': 64,
    'epochs': 100,
    'classes': data_labels,
    'optimizer': 'Adam',
    'loss_function': 'CE',
    'metrics': [Accuracy(), AUC(), F1(), ROC()],
}

print("START") 
model = Model(hyperparams)

wandb.init(project="SurfaceNet", entity="zaynr")
wandb.config = hyperparams
wandb.watch(model, log_freq=100)

for epoch in range(hyperparams["epochs"]):
    for imgs, labels in zip(training_data[0], training_data[1]):
        model.history.clear()

        train_batch_imgs = torch.tensor(imgs).float()
        train_batch_labels = torch.tensor(labels).float()
       
        model.train_batch(train_batch_imgs, train_batch_labels)
        model.evaluate(test_imgs, test_labels)

        wandb.log(model.history)
