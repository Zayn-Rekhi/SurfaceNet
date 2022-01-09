import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

from pathlib import Path
import sys
import os
from sklearn.metrics import (accuracy_score, 
                             recall_score,
                             precision_score,  
                             f1_score, 
                             confusion_matrix) 


sys.path.insert(1, "../data")
from ctx_image import CTX_Image, download_file

sys.path.insert(1, "../models")
from mrf import MRF
from network import Model, DEVICE


torch.backends.cudnn.benchmark = True

CTX_stripe = "D14_032794_1989_XN_18N282W"
path = "../../data/raw/" + CTX_stripe + ".tiff"

cutouts = {
    "D14_032794_1989_XN_18N282W": (1600, 11000, 7000, 14000),  # Jezero
    "F13_040921_1983_XN_18N024W": (3000, 3000, 5800, 7000),  # Oxia Planum
    "G14_023651_2056_XI_25N148W": (1000, 1000, 2000, 2000),  # Lycus Sulci
}

links = {
    "D14_032794_1989_XN_18N282W": "https://image.mars.asu.edu/stream/D14_032794_1989_XN_18N282W.tiff?image=/mars/images/ctx/mrox_1861/prj_full/D14_032794_1989_XN_18N282W.tiff",  # Jezero
    "F13_040921_1983_XN_18N024W": "https://image.mars.asu.edu/stream/F13_040921_1983_XN_18N024W.tiff?image=/mars/images/ctx/mrox_2375/prj_full/F13_040921_1983_XN_18N024W.tiff",  # Oxia Planum
    "G14_023651_2056_XI_25N148W": "https://image.mars.asu.edu/stream/G14_023651_2056_XI_25N148W.tiff?image=/mars/images/ctx/mrox_1385/prj_full/G14_023651_2056_XI_25N148W.tiff",  # Lycus Sulci
}

if not Path(path).exists():
    # Download file
    print("Dowloading...\n")
    download_file(links[CTX_stripe], "../../data/raw/")
    print("...Done")
    

data_transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Grayscale(num_output_channels=1),
    ]
)

ctx_image = CTX_Image(path=path, cutout=cutouts[CTX_stripe], transform=data_transform)
test_loader = DataLoader(
    ctx_image, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
)

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
        ("recall_score", lambda X, y: recall_score(X, y, average='macro')), 
        ("precision_score", lambda X, y: precision_score(X, y, average='macro')), 
        ("f1_score", lambda X, y: f1_score(X, y, average='macro')),
        ("confusion_matrix", lambda X, y: confusion_matrix(X, y)), 
    ],
    'transfer_learning': False,
}

model = Model(hyperparams)


checkpoint = torch.load("/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/models/model40.pt")
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)

predictions = []
scores = []
image_pred = []

# Analysing image
with tqdm(test_loader, desc="Testing", leave=False) as t:
    with torch.no_grad(): 
        for batch in t:
            x, center_pixels = batch
            y_hat = model(x.to(DEVICE))

            pred = torch.argmax(y_hat, dim=1).cpu() 

            image_pred.append(center_pixels.numpy())
            predictions.append(pred.numpy())
            scores.append(F.softmax(y_hat, dim=1).detach().cpu().numpy()) 
            
predictions = np.concatenate(predictions, axis=0)
scores = np.concatenate(scores, axis=0)
scores = np.reshape(
    np.array(scores), ctx_image.out_shape + (int(hyperparams["classes"]),)
)
image_pred = np.reshape(np.concatenate(image_pred, axis=0), ctx_image.out_shape)

# Markov random field smoothing
mrf_probabilities = MRF(scores.astype(np.float64))
mrf_classes = np.argmax(mrf_probabilities, axis=2)

# Create Colormap
n = int(hyperparams["classes"])
from_list = mpl.colors.LinearSegmentedColormap.from_list
cm = from_list(None, plt.cm.tab20(range(0, n)), n)

# Saving Images
plt.imsave(
    "../../results/" + CTX_stripe + "_map.png",
    np.reshape(np.array(predictions), ctx_image.out_shape),
    cmap=cm,
    vmin=0,
    vmax=int(hyperparams["classes"]),
)
plt.imsave(
    "../../results/" + CTX_stripe + "_img.png",
    np.dstack(
        [np.reshape(np.concatenate(image_pred, axis=0), ctx_image.out_shape)] * 3
    ),
)
plt.imsave(
    "../../results/" + CTX_stripe + "_mrf.png",
    mrf_classes,
    cmap=cm,
    vmin=0,
    vmax=int(hyperparams["classes"]),
)
