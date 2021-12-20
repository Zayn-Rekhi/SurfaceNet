"""
"src/data/load.py"

Class definition of data loading/processing/augmentation

--path

"""

import numpy as np  # Matrix Operations
import cv2  # Loading images
import os  # Handling files
import matplotlib.pyplot as plt  # Graphing Images
from tqdm import tqdm  # Progress Bar
import argparse  # Terminal
from functools import lru_cache  # Cache Clearing
import sys  # System Utility
from typing import Tuple

try:
    sys.path.insert(1, '../visualization')
    from plotting import plot_imgs 
except ImportError:
    pass


class Dataset:
    def __init__(self, path: str) -> None:
        self.path: str = path
        self.labels: np.array = np.asarray(os.listdir(os.path.join(self.path, "train")), dtype=object) 
        self.data: dict = {"train": None, "val": None, "test": None}

        self._load()

    def _load(self) -> None:
        """
        Loads all the images in the given the path
        :return: None
        """
        for key in self.data.keys():
            path = os.path.join(self.path, key)
            images, labels, names = [], [], []

            for label in tqdm(self.labels, desc=f"{key.upper()}", ncols=75):  
                loaded_imgs, loaded_names = self._load_imgs(os.path.join(path, label))
                loaded_labels = self._encode(label)
                
                images.extend(loaded_imgs)
                labels.extend([loaded_labels for _ in loaded_imgs])
                names.extend(loaded_names) 
            
            self.data[key] = [np.asarray(images, dtype=np.uint8), 
                              np.asarray(labels, dtype=np.uint8), 
                              np.asarray(names, dtype=np.object)]

    @staticmethod 
    def _load_imgs(path) -> np.array:
        """
        Utilizes the path of the classes of images in order to store the images located in that directory.
        :param path: Directory of feature
        :return: Numpy array of images
        """
        imgs, names = [], []
        for image_path in os.listdir(path):
            if image_path.endswith('.jpg'):
                tmp_path = os.path.join(path, image_path)
                image = np.reshape(cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE), (1, 1, 200, 200))                
                names.append([image_path])
                imgs.extend(image) 
        return np.asarray(imgs, dtype=np.uint8), np.asarray(names, dtype=np.object)

    def _encode(self, label) -> np.array:
        """
        This functions use is to encode the label based off the one hot encoding method. It creates an empty array
        (filled with zeros) and sets the label equal to one.
        :param label: Number (0-15) that dictates the number to be set to 1
        :return: One-hot encoded numpy array
        """
        possible_labels = np.zeros((len(self.labels), 1))
        possible_labels[np.array(self.labels == label)] = 1 
        return possible_labels

    @lru_cache(maxsize=None)
    def get_class(self, class_name: str, location: str, amt: int):
        search_label = np.zeros((len(self.labels), 1))
        search_label[np.array(self.labels == class_name)] = 1 
        
        idxs = [idx for idx, label in enumerate(self.data[location][1])  
                                          if np.array_equal(label, search_label)] 
        

        imgs = self.data[location][0][idxs][:amt]      
        labels = np.asarray([search_label] * amt, dtype=np.uint8)
        names = self.data[location][2][idxs][:amt][:, 0]   
        
        assert imgs.shape[0] == labels.shape[0] == names.shape[0], "Not the same shape"

        return imgs, labels, names
    
    def to_batches(self, batch_size=10): 
        num_labels = len(self.labels)
        
        for key, items in self.data.items():
            total_batches = int(len(self.data[key][0]) / batch_size) 
              
            self.data[key][0] = np.array_split(items[0], total_batches) 
            self.data[key][1] = np.array_split(items[1], total_batches)
            self.data[key][2] = np.array_split(items[2], total_batches) 

if __name__ == "__main__":
    from image_augment import *

    os.chdir(os.path.expanduser("~"))
    
    parser = argparse.ArgumentParser(description='State path of dataset')
    parser.add_argument('--path', metavar='path', type=str, help='Path to list')
    args = parser.parse_args()

    generators = {
        "train": ImageAugmentation([ 
                                    Rescale(scale = 1/255)], True),
        "val": ImageAugmentation([Rescale(scale = 1./255)], True),
        "test": ImageAugmentation([Rescale(scale = 1./255)], True),
    }
    loader = Dataset(args.path)
    sample_imgs, sample_labels, sample_names = loader.get_class('rid', 'train', 20)
    
    print(sample_names[0]) 

    plot_imgs(sample_imgs, sample_names, sample_labels, graph_shape=(3, 3))
