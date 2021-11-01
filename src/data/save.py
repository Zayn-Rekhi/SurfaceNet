from load import Load
import os
from PIL import Image
import numpy as np

PATH = "/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/processed/data"

def save(dataset: dict, labels: list, path: str) -> bool:
    for subset, data in dataset.items():
        lib_creator = map(lambda label: os.mkdir(os.path.join(path, subset, label)), labels)

        for X, y in zip(data[0], data[1]):
            img = Image.fromarray(X)
            label = labels.index(y.index(1))
            img_path = os.path.join(path, subset, label)

if __name__ == "__main__":
    loader = Load("/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/raw/data", image_aug)
    dataset = loader.dataset
    for key, value in dataset.items():
        path = os.path.join(PATH, key)
        status = save(value, path)
