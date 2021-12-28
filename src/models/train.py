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

# ----------------------------- DATA LOADING ------------------------------------ #
generators = {
    'train': ImageAugmentation([Rescale(scale = 1/255)], True),
    'val': ImageAugmentation([Rescale(scale = 1./255)], True),
    'test': ImageAugmentation([Rescale(scale = 1./255)], True),
}
data = Dataset(path="/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/raw/data")
#data.apply_augmentations(generators)
data.to_batches(8, ['train'])
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
model = nn.DataParallel(model)
model.to(DEVICE)
model_save_path = "../../models"

#wandb.init(project="SurfaceNet", entity="zaynr")
#wandb.config = hyperparams
#wandb.watch(model, log_freq=100)

for epoch in range(hyperparams["epochs"]): 
    print(f"-----------------Starting Epoch {epoch}----------------")

    model.history.clear()
    train_loss = 0
    for imgs, labels in tqdm(zip(training_data[0], training_data[1]), desc=f"Progress", ncols=75): 
        imgs = torch.tensor(imgs/255).float().to(DEVICE)
        labels = torch.tensor(labels).float().to(DEVICE)
         
        train_loss += model.train_batch(imgs, labels)      
        del imgs, labels 

    gc.collect()
    torch.cuda.empty_cache()

    model.history['train_loss'] = train_loss/len(training_data[0])  
    print(f"Epoch {epoch}.........COMPLETED(Loss = {model.history['train_loss']})") 
    with torch.no_grad():    
        val_predictions, val_labels = [], []
        val_loss = 0
        for val_imgs, val_label in zip(validation_data[0], validation_data[1]):      
            val_imgs = torch.tensor(val_imgs/255).float().to(DEVICE)
            val_label_reg = [np.argmax(label) for label in val_label]
            
            loss, outputs = model.val_batch(val_imgs, torch.tensor(val_label).float().to(DEVICE))
            val_loss += loss
            prediction = [output.argmax() for output in outputs.detach().cpu().numpy()] 
            
            val_predictions.extend(prediction)
            val_labels.extend(val_label_reg)

        evaluation = model.evaluate(val_labels, val_predictions)    
        model.history['val_loss'] = val_loss/len(validation_data[0])      
        self.lr_scheduler.step(model.history['val_loss'])

    save(os.path.join(model_save_path, f"model{epoch}.pt"), model)    
    if epoch > 0:
        delete(os.path.join(model_save_path, f"model{epoch-1}.pt"))
    #wandb.log(evaluation) 


