from network import Model, DEVICE
import torch.optim as optim
import torch.nn as nn
import torch
import sys 
import numpy as np

sys.path.append("../data")
from load import Dataset
from image_augment import ImageAugmentation, Rescale

from sklearn.metrics import (accuracy_score, 
                             recall_score,
                             precision_score, 
                             auc,
                             f1_score, 
                             confusion_matrix) 
import wandb
import gc
import pandas as pd 

# ----------------------------- DATA LOADING ------------------------------------- #
generators = {
    'train': ImageAugmentation([Rescale(scale = 1/255)], True),
    'val': ImageAugmentation([Rescale(scale = 1./255)], True),
    'test': ImageAugmentation([Rescale(scale = 1./255)], True),
}
data = Dataset(path="/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/raw/data")
data.to_batches(8, ['train', 'val'])
data.to_batches(8, ['test'])
data_labels = data.labels
training_data, validation_data, testing_data = data.data["train"], data.data["val"], data.data["test"]


# ----------------------------- NETWORK ------------------------------------- #
hyperparams = {
    'lr': 0.00001,
    'batch_size': 8,
    'epochs': 300,
    'classes': data_labels,
    'optimizer': 'Adam',
    'loss_function': 'CE',
    'metrics': [
        ("accuracy", lambda X, y: accuracy_score(X, y)),
        ("recall_score", lambda X, y: recall_score(X, y, average='macro')), 
        ("precision_score", lambda X, y: precision_score(X, y, average='macro')),
        ("AUC", lambda X, y: auc(X, y, average='macro')),
        ("f1_score", lambda X, y: f1_score(X, y, average='macro')),
        ("confusion_matrix", lambda X, y: confusion_matrix(X, y))
    ],
}

 
model = Model(hyperparams)
model.to(DEVICE)

wandb.init(project="SurfaceNet", entity="zaynr")
wandb.config = hyperparams
wandb.watch(model, log_freq=100)

for epoch in range(hyperparams["epochs"]):
    model.history.clear()
    epoch_loss = 0
    for imgs, labels in zip(training_data[0][:50], training_data[1][:50]):
       
        train_imgs = torch.tensor(imgs).float().to(DEVICE)
        train_labels = torch.tensor(labels).float().to(DEVICE)
        
        loss = model.train_batch(train_imgs, train_labels) 
        epoch_loss+=loss 
        
        gc.collect()
        torch.cuda.empty_cache()

    print(f"{epoch} COMPLETED ----- Loss({epoch_loss})") 
    
    val_predictions, val_labels = [], [] 
    for val_imgs, val_label in zip(validation_data[0], validation_data[1]):      
        val_imgs = torch.tensor(val_imgs).float().to(DEVICE)
        val_label = [np.argmax(label) for label in val_label]
        
        outputs = model.forward(val_imgs).detach().cpu().numpy() 
        prediction = [output.argmax() for output in outputs] 
        
        val_predictions.extend(prediction)
        val_labels.extend(val_label)
    
    evaluation = model.evaluate(val_labels, val_predictions)     
    
    for key, value in evaluation.items():
        print(f"{key} == {value}")
    
    print("\n\n\n")
    wandb.log(evaluation)   
    if evaluation["accuracy"] > 0.05:
        break


test_predictions, test_labels = [], [] 
for test_imgs, test_label in zip(testing_data[0], testing_data[1]):      
    test_imgs = torch.tensor(test_imgs).float().to(DEVICE)
    test_label = [np.argmax(label) for label in test_label]
    
    outputs = model.forward(test_imgs).detach().cpu().numpy() 
    prediction = [output.argmax() for output in outputs] 
    
    test_predictions.extend(prediction)
    test_labels.extend(test_label)

evaluation = model.evaluate(test_labels, test_predictions)     
evaluation = pd.DataFrame(evaluation)
wandb.log({"Testing Results": evaluation})

