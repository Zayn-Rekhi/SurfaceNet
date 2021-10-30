from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

TRAIN_DATA_GENERATOR = ImageDataGenerator()
VAL_DATA_GENERATOR = ImageDataGenerator()
TEST_DATA_GENERATOR = ImageDataGenerator()

class Load:
    def __init__(self, path: str, image_augmentation=None) -> None:
        self.path: str = path
        self.labels: list = os.listdir(os.path.join(self.path, "train"))
        self._dataset: dict = {"train": None, "val": None, "test": None}

        self._load()

        if image_augmentation:
            self._augment()

    def _augment(self):
        self._dataset["train"] = TRAIN_DATA_GENERATOR.flow(self._dataset["train"]).as_numpy()
        self._dataset["val"] = VAL_DATA_GENERATOR.flow(self._dataset["val"]).as_numpy()
        self._dataset["test"] = TEST_DATA_GENERATOR.flow(self._dataset["test"]).as_numpy()

    def _load(self) -> None:
        for key in self._dataset.keys():
            path = os.path.join(self.path, key)
            data = []

            for label in self.labels:
                encoded_label = self._encode(label)
                images = self._load_imgs(os.path.join(path, label), encoded_label)
                data.append(images)

            self._dataset[key] = np.asarray(data)

    @staticmethod
    def _load_imgs(path, label) -> np.array:
        imgs = []
        for image in os.listdir(path):
            tmp_path = os.path.join(path, image)
            imgs.append([cv2.imread(tmp_path), label])
        return np.asarray(imgs)

    def _encode(self, label) -> np.array:
        possible_labels = np.zeros(len(self.labels))
        possible_labels[self.labels.index(label)] = 1
        return possible_labels

    def get_dataset(self) -> dict:
        return self._dataset


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~"))
    print(os.getcwd())
    load("/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/raw/data")
