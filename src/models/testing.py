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
import json
from tqdm import tqdm
import codecs
# ----------------------------- DATA LOADING ------------------------------------ #

data = Dataset(path="/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/raw/data")
data.to_batches(8, ['train'])
data.to_batches(8, ['val', 'test'])

data_labels = data.labels
training_data, validation_data, testing_data = data.data["train"], data.data["val"], data.data["test"]



hyperparams = {
    'lr': 0.001,
    'batch_size': 8,
    'epochs': 300,
    'classes': 15,
    'optimizer': 'SGD',
    'momentum': 0.9,
    'loss_function': 'CE',
    'metrics': [
        ("accuracy", lambda X, y: accuracy_score(X, y)),
        ("recall_score", lambda X, y: recall_score(X, y, average=None, labels=range(len(data_labels)))), 
        ("precision_score", lambda X, y: precision_score(X, y, average=None, labels=range(len(data_labels)))), 
        ("f1_score", lambda X, y: f1_score(X, y, average=None, labels=range(len(data_labels)))),
        ("confusion_matrix", lambda X, y: confusion_matrix(X, y)), 
    ],
    'transfer_learning': False,
}

def evaluate_model(model, eval_data): 
    with torch.no_grad():
        eval_predictions, eval_labels = [], []
        eval_loss = 0
        for eval_imgs, eval_label in tqdm(zip(eval_data[0], eval_data[1]), total=len(eval_data[0]), desc="Evaluation"):      
            eval_imgs = torch.tensor(eval_imgs/255).float().to(DEVICE)
            eval_label_reg = [np.argmax(label) for label in eval_label]
            
            loss, outputs = model.test_batch(eval_imgs, torch.tensor(eval_label).float().to(DEVICE))
            eval_loss += loss
            prediction = [output.argmax() for output in outputs.detach().cpu().numpy()] 
            
            eval_predictions.extend(prediction)
            eval_labels.extend(eval_label_reg)
            del eval_imgs, eval_label
        
    evaluation = model.evaluate(eval_labels, eval_predictions)    
    evaluation['recall_score'] = evaluation['recall_score'].tolist()
    evaluation['precision_score'] = evaluation['precision_score'].tolist()
    evaluation['confusion_matrix'] = evaluation['confusion_matrix'].tolist()
    evaluation['f1_score'] = evaluation['f1_score'].tolist()
    evaluation['loss'] = (eval_loss / len(eval_data[0])).cpu().tolist()
    
    return evaluation




model = Model(hyperparams)

checkpoint = torch.load("/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/models/model40.pt")
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

model_evaluation = dict()
model_evaluation["train"] = evaluate_model(model, training_data)
model_evaluation["val"] = evaluate_model(model, validation_data)
model_evaluation["test"] = evaluate_model(model, testing_data)

print(model_evaluation)
file_path = "../../results/evaluation.json"
evaluation = json.dump(model_evaluation, 
                        codecs.open(file_path, 'w', encoding='utf-8'), 
                        separators=(',', ':'), 
                        sort_keys=True, 
                        indent=4)

#with open(, "w+") as file:
#    json.dump(evaluation, file)




