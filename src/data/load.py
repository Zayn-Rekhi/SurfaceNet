"""
"src/data/load.py"

Class definition of data loading/processing/augmentation

"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Image Augmentation
import numpy as np  # Matrix Operations
import cv2  # Loading images
import os  # Handling files
import matplotlib.pyplot as plt  # Graphing Images
from tqdm import tqdm  # Progress Bar
import argparse  # Terminal
from functools import lru_cache  # Cache Clearing
import sys  # System Utility

try:
    sys.path.insert(1, '../visualization')
    from plotting import plot_imgs 
except ImportError:
    pass

GENERATORS = {
    "train": ImageDataGenerator(rescale=1 / 255),
    "val": None,
    "test": None,
}


class Dataset:
    def __init__(self, path: str, image_augmentation: bool = False) -> None:
        self.path: str = path
        self.image_augmentation: bool = image_augmentation
        self.labels: np.array = np.asarray(os.listdir(os.path.join(self.path, "train")), dtype=object)
        print(self.labels)
        self.data: dict = {"train": None, "val": None, "test": None}

        self._load()
        if self.image_augmentation:
            self._augment()

    def _augment(self) -> None:
        """
        Utlizes the tensorflow ImageDataGenerator to augment images
        :return: None
        """
        for subset, generator in GENERATORS.items():
            if generator:
                flow = generator.flow(self.dataset[subset][0])
                self.dataset[subset] = np.concatenate([[flow.next()[0], flow.next()[1]]
                                                       for _ in range(flow.__len__())])

    def _load(self) -> None:
        """
        Loads all the images in the given the path
        :return: None
        """
        for key in self.data.keys():
            path = os.path.join(self.path, key)
            images, labels, names = [], [], []

            #for label in tqdm(self.labels, desc=f"{key.upper()}", ncols=75):  
            for label in self.labels:
                loaded_imgs, loaded_names = self._load_imgs(os.path.join(path, label))
                loaded_labels = self._encode(label)
                
                images.extend(loaded_imgs)
                labels.extend([loaded_labels for _ in loaded_imgs])
                names.extend(loaded_names) 
            
            self.data[key] = (np.asarray(images, dtype=np.uint8), np.asarray(labels, dtype=np.uint8), np.asarray(names, dtype=np.object))

    @staticmethod
    @lru_cache(maxsize=None)
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
                image = np.reshape(cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE), (1, 200, 200, 1))                
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


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~"))
    
    parser = argparse.ArgumentParser(description='State path of dataset')
    parser.add_argument('--path', metavar='path', type=str, help='Path to list')
    parser.add_argument('--image_augment', metavar='image_augment', type=int, help='Augmentation')
    args = parser.parse_args()

    path, image_augment = args.path, bool(args.image_augment)
    loader = Dataset(path, image_augment)
        
    examples = loader.data["train"]
    label_names = loader.labels
    
    sample_indexes = np.linspace(start=0, stop=len(examples[0]), endpoint=False, retstep=10, dtype=np.int32) 
    
    sample_imgs = examples[0][sample_indexes[0]]     
    sample_labels = examples[1][sample_indexes[0]]
    sample_names = examples[2][sample_indexes[0]]

    sample_names = sample_names[:, 0]
    sample_labels = label_names[[np.where(sample == 1)[0] for sample in sample_labels]][:, 0] 
    
    plot_imgs(sample_imgs, sample_names, sample_labels, graph_shape=(3, 3))
