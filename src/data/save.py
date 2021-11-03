from load import Load
import os
import cv2
import numpy as np

PATH = "/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/processed/data"


def save(dataset: dict, path_to_save: str) -> bool:
    labels = dataset["labels"]
    for subset, data in dataset.items():
        map(lambda label: os.mkdir(os.path.join(path_to_save, subset, label)), labels)

        for i, (X, y) in enumerate(zip(data[0], data[1])):
            label = labels.index(y.index(1))
            path = os.path.join(path_to_save, subset, label, f"{i}.jpg")
            cv2.imwrite(path, X)


if __name__ == "__main__":
    loader = Load("/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/raw/data", False)
    dataset = loader.dataset
    for key, value in dataset.items():
        path = os.path.join(PATH, key)
        status = save(value, path)
