from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Image Augmentation
import numpy as np  # Matrix Operations
import cv2  # Loading images
import os  # Handling files
import matplotlib.pyplot as plt  # Graphing Images
from tqdm import tqdm  # Progress Bar


class Load:
    def __init__(self, path: str, image_augmentation: dict = None) -> None:
        self.path: str = path
        self.image_augmentation: dict = image_augmentation
        self.labels: list = os.listdir(os.path.join(self.path, "train"))
        self.dataset: dict = {"train": None, "val": None, "test": None}

        self._load()
        if self.image_augmentation:
            self._augment()

    def _augment(self) -> None:
        """
        Utlizes the tensorflow ImageDataGenerator to augment images
        :return: None
        """
        for subset, generator in self.image_augmentation.items():
            if generator:
                flow = generator.flow(self.dataset[subset][0])
                self.dataset[subset] = np.concatenate([[flow.next()[0], flow.next()[1]]
                                                       for _ in range(generator.__len__())])

    def _load(self) -> None:
        """
        Loads all the images in the given the path
        :return: None
        """
        for key in self.dataset.keys():
            path = os.path.join(self.path, key)
            images, labels = [], []

            for label in tqdm(self.labels, desc=f"{key.upper()}", ncols=75):
                labels.extend(self._encode(label))
                images.extend(self._load_imgs(os.path.join(path, label)))

            self.dataset[key] = (np.asarray(images), np.asarray(labels))

    @staticmethod
    def _load_imgs(path) -> np.array:
        """
        Utilizes the path of the classes of images in order to store the images located in that directory.
        :param path: Directory of feature
        :return: Numpy array of images
        """
        imgs = []
        for image in os.listdir(path):
            if image.endswith(".jpg"):
                tmp_path = os.path.join(path, image)
                image = np.reshape(cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE), (1, 200, 200, 1))
                imgs.extend(image)
        return np.asarray(imgs)

    def _encode(self, label) -> np.array:
        """
        This functions use is to encode the label based off the one hot encoding method. It creates an empty array
        (filled with zeros) and sets the label equal to one.
        :param label: Number (0-15) that dictates the number to be set to 1
        :return: One-hot encoded numpy array
        """
        possible_labels = np.zeros((len(self.labels), 1))
        possible_labels[self.labels.index(label)] = 1
        return possible_labels


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~"))
    image_aug = {
        "train": ImageDataGenerator(rescale=1 / 255),
        "val": None,
        "test": None,
    }

    loader = Load("/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/raw/data", image_aug)
    dataset = loader.dataset
    plt.imshow(dataset["train"][0][0])
    plt.show()
