from abc import abstractmethod
from datagen import Dataset
import numpy as np
import random
import scipy
from numba import njit
import matplotlib.pyplot as plt

class Augmentor:
    @abstractmethod
    def augment(self, images):
        raise NotImplementedError()


class Rotation(Augmentor):
    def __init__(self, start=-360, end=360): 
        self.start = start
        self.end = end

    def augment(self, images):
        augmented_images = []
        for img in images:
            rand_range = random.randint(self.start, self.end)
            augmented_images.append(scipy.ndimage.rotate(img, rand_range, reshape=False))
        return np.asarray(augmented_images)

    def __str__(self):
        return f"Rotation({self.start}, {self.end})"



class Flip(Augmentor):
    def __init__(self, types=("Horizontal", "Vertical")):
        self.types = types

    def augment(self, images):
        augmented_images = []
        for type in self.types:
            axis = 0 if type == "Horizontal" else 1
            for img in images:       
                augmented_images.append(np.flip(img, axis=axis))
        return np.asarray(augmented_images)

    def __str__(self):
        return f"Flip({self.types})"


class Rescale(Augmentor):
    def __init__(self, scale: float):
       self.scale = scale

    def augment(self, images: np.array):
        augmented_images = []
        for img in images:
            augmented_images.append(img * self.scale)
        return np.asarray(augmented_images)
   
    def __str__(self):
       return f"Rescale({self.scale})"


class ImageAugmentation:
    def __init__(self, augment_config: set, progress: bool):
        self.augment_config: set = augment_config
        self.progress: bool = progress
        
        if progress:
            print("---------STARTING IMAGE AUG---------")

    def apply(self, images: np.array):
       
        image_shape = images.shape
        for amr in self.augment_config:
            images = amr.augment(images)
            if self.progress:
                print(f"{amr}........DONE({images.shape})")
       
        assert images.shape[1:] == image_shape[1:], f"Augmented images shape don't match original"

        return images
    

if __name__ == "__main__":
    augmentation_config = {
        Rotation(start=-90, end=90),
        Flip(types = ("Horizontal", "Vertical")),  
        Rescale(scale = 1./255)
         
    }

    dataset = Dataset("/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/raw/data")
    train_x = dataset.data["train"][0][:5]
   
    image_augment = ImageAugmentation(augment_config=augmentation_config,
                                      progress=True)
 
    augmented_data = image_augment.apply(train_x)
    plt.imshow(augmented_data[0])
    plt.show()


