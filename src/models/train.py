from network import Model, DEVICE, save, delete
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
                             f1_score, 
                             confusion_matrix) 
import wandb
import gc
import pandas as pd 
import os 
from tqdm import tqdm 

# ----------------------------- DATA LOADING ------------------------------------- #
generators = {
    'train': ImageAugmentation([Rescale(scale = 1/255)], True),
    'val': ImageAugmentation([Rescale(scale = 1./255)], True),
    'test': ImageAugmentation([Rescale(scale = 1./255)], True),
}
data = Dataset(path="/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/raw/data")
data.to_batches(64, ['train'])
data.to_batches(8, ['val', 'test'])
data_labels = data.labels
training_data, validation_data, testing_data = data.data["train"], data.data["val"], data.data["test"]


# ----------------------------- NETWORK ------------------------------------- #
hyperparams = {
    'lr': 0.0001,
    'batch_size': 8,
    'epochs': 300,
    'classes': data_labels,
    'optimizer': 'Adam',
    'loss_function': 'CE',
    'metrics': [
        ("accuracy", lambda X, y: accuracy_score(X, y)),
        ("recall_score", lambda X, y: recall_score(X, y, average='macro')), 
        ("precision_score", lambda X, y: precision_score(X, y, average='macro')), 
        ("f1_score", lambda X, y: f1_score(X, y, average='macro')),
        ("confusion_matrix", lambda X, y: confusion_matrix(X, y)), 
    ],
}




 
model = Model(hyperparams)
model.to(DEVICE)
model_save_path = "../../models"

"""wandb.init(project="SurfaceNet", entity="zaynr")
wandb.config = hyperparams
wandb.watch(model, log_freq=100)"""

for epoch in range(hyperparams["epochs"]):
    
    print(f"-----------------Starting Epoch {epoch}----------------")

    model.history.clear()
    train_loss = 0
    for imgs, labels in tqdm(zip(training_data[0], training_data[1]), desc=f"Progress", ncols=200): 
        imgs = torch.tensor(imgs).float().to(DEVICE)
        labels = torch.tensor(labels).float().to(DEVICE)
         
        train_loss += model.train_batch(imgs, labels) 
        
    gc.collect()
    torch.cuda.empty_cache()

    model.history['train_loss'] = train_loss/len(training_data[0])  
    print(f"Epoch {epoch}.........COMPLETED(Loss = {model.history['train_loss']})") 
    
    test_predictions, test_labels = [], []
    test_loss = 0
    for test_imgs, test_label in zip(testing_data[0][:1], testing_data[1][:1]):      
        test_imgs = torch.tensor(test_imgs).float().to(DEVICE)
        test_label_reg = [np.argmax(label) for label in test_label]
        
        loss, outputs = model.test_batch(test_imgs, torch.tensor(test_label).float().to(DEVICE))
        test_loss += loss
        prediction = [output.argmax() for output in outputs.detach().cpu().numpy()] 
        
        test_predictions.extend(prediction)
        test_labels.extend(test_label_reg)

    evaluation = model.evaluate(test_labels, test_predictions)    
    model.history['test_loss'] = test_loss/len(testing_data[0])      
    save(os.path.join(model_save_path, f"model{epoch}.pt"), model)    
    #wandb.log(evaluaj 


