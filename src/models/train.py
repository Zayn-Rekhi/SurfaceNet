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
    model.history.clear()
    train_loss = 0
    for imgs, labels in zip(training_data[0][:5], training_data[1][:5]): 
        imgs = torch.tensor(imgs).float().to(DEVICE)
        labels = torch.tensor(labels).float().to(DEVICE)
         
        train_loss += model.train_batch(imgs, labels) 
        
        gc.collect()
        torch.cuda.empty_cache()

    model.history['train_loss'] = train_loss  
    print(f"{epoch} COMPLETED ----- Loss({model.history['train_loss']})") 
    
    val_labels, val_predictions = [], []
    val_loss = 0
    for val_imgs, val_label in zip(validation_data[0], validation_data[1]):      
        val_imgs = torch.tensor(val_imgs).float().to(DEVICE)
        val_labels_reg = torch.tensor([np.argmax(label) for label in val_label]).float().to(DEVICE)
        
        outputs = model.forward(val_imgs)
        val_loss += model.loss_function(outputs, val_labels_reg) 
        prediction = [output.argmax() for output in outputs] 

        val_labels.extend(val_labels_reg)     
        val_predictions.extend(prediction)

    print(val_labels, val_predictions) 
    print(np.ptp(val_labels), np.ptp(val_predictions))
    model.history['val_loss'] = val_loss/len(validation_data[0])
    evaluation = model.evaluate(val_labels, val_predictions)     
    save(os.path.join(model_save_path, f"model{epoch}.pt"), model)    


    for key, value in evaluation.items():
        print(f"{key} == {value}")
    
    print("\n\n\n")
    #wandb.log(evaluation)   
    if evaluation["accuracy"] > 0.05:
        break
    delete(os.path.join(model_save_path, f"model{epoch}.pt"))    

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

