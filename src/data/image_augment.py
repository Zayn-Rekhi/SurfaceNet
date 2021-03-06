from abc import abstractmethod
from load import * 
import numpy as np
import random
import scipy.ndimage
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
        for idx, img in enumerate(images): 
            rand_range = random.randint(self.start, self.end)
            image = scipy.ndimage.rotate(img, rand_range, reshape=False)
            augmented_images.append(image)
        return np.asarray(augmented_images)

    def __str__(self):
        return f"Rotation({self.start}, {self.end})"



class Flip(Augmentor):
    def __init__(self, types=("Horizontal", "Vertical")):
        self.types = types

    def augment(self, images):
        augmented_images = []
        for img in images:
            choice = random.choice(self.types)
            axis = 0 if choice == "Horizontal" else 1 
            augmented_images.append(np.flip(img, axis=axis))
        return np.asarray(augmented_images)

    def __str__(self):
        return f"Flip({self.types})"


class Rescale(Augmentor):
    def __init__(self, scale: float):
        self.scale = scale

    def augment(self, images: np.array):
        return images * self.scale 
         
    def __str__(self):
       return f"Rescale({self.scale})"


class ImageAugmentation:
    def __init__(self, augment_config: list, show_progress: bool = False):
        self.augment_config: list = augment_config
        self.show_progress: bool = show_progress
        
  
    def apply(self, images: np.array) -> Tuple[np.array, np.array]: 
        if self.show_progress:
            print("--------STARTING IMAGE AUG---------")

        image_shape = images.shape
        for amr in self.augment_config:  
            images = amr.augment(images)
            print(images) 
            if self.show_progress:
                print(f"{amr}........DONE({images.shape})")
       
        assert images.shape == image_shape, f"Augmented images shape don't match original"
        return images 

if __name__ == "__main__":
    augmentation_config = [
        Rotation(start=-90, end=90),
        Flip(types = ("Horizontal", "Vertical")),  
        Rescale(scale = 1./255)         
    ]

    dataset = Dataset("/home/zayn/Desktop/Programming/PYTHON/ML/MarsNet/data/raw/data", (False, None))
    train_x = dataset.data["train"][0][:5]
   
    image_augment = ImageAugmentation(augment_config=augmentation_config, show_progress=True) 
    augmented_data = image_augment.apply(train_x)
    plt.imshow(augmented_data[0][0])
    plt.show()


