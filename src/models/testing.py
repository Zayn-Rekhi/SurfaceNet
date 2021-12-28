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
import gc
from tqdm import tqdm 



# ----------------------------- DATA LOADING ------------------------------------ #

data = Dataset(path="/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/raw/data")
data.to_batches(16, ['test'])
testing_data = data.data["test"]



hyperparams = {
    'lr': 0.0001,
    'batch_size': 8,
    'epochs': 32,
    'classes': 15,
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

checkpoint = torch.load("/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/models/model4.pt")
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

with torch.no_grad():
    test_predictions, test_labels = [], []
    test_loss = 0
    for test_imgs, test_label in zip(testing_data[0], testing_data[1]):      
        test_imgs = torch.tensor(test_imgs/255).float().to(DEVICE)
        test_label_reg = [np.argmax(label) for label in test_label]
        
        loss, outputs = model.test_batch(test_imgs, torch.tensor(test_label).float().to(DEVICE))
        test_loss += loss
        prediction = [output.argmax() for output in outputs.detach().cpu().numpy()] 
        
        test_predictions.extend(prediction)
        test_labels.extend(test_label_reg)
        del test_imgs, test_label
    evaluation = model.evaluate(test_labels, test_predictions)    
    print(evaluation)
