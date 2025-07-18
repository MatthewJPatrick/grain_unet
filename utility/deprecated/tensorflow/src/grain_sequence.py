'''
Grain Sequence

This class reads images from a list of file names, augments them, and returns a
sequence (which keras can train with)

Parameters:
    batch_size      - Number of images per entry in the sequence
    img_size        - The size of the input data
    input_img_paths - List of paths for training images
    label_img_paths - List of paths of label images
                      (in the same order as input_img_paths)
    enable_augment  - If False, doesn't do random data augmentation

Author: Jamie (jamie.k.eckstein@gmail.com)
Credit: fchollet

'''



import random
from tensorflow import keras
from skimage import io, exposure
import numpy as np
import cv2


class GrainSequence(keras.utils.Sequence):
    '''Helper to iterate over the data (as Numpy arrays).'''
    def __init__(self, batch_size, img_size, input_img_paths, label_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.label_img_paths = label_img_paths

    def __len__(self):
        return len(self.label_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_label_img_paths = self.label_img_paths[i : i + self.batch_size]

        input_imgs = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = io.imread(path)
            if len(img.shape) > 2:
                img = img[:,:,0]
            assert img.shape == self.img_size, f"Training images must be downscaled to {self.img_size} manually"

            img = img - img.min()
            img = img.astype('float32') / np.ptp(img)
            #code for contrast stretching
            '''p2, p98 = np.percentile(img, (2, 98))
            img = exposure.rescale_intensity(img, in_range=(p2, p98))'''

            input_imgs[j] = np.expand_dims(img, 2)

        label_imgs = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_label_img_paths):
            img = io.imread(path)
            if len(img.shape) > 2:
                img = img[:,:,0]
            assert img.shape == self.img_size, f"Training images must be downscaled to {self.img_size} manually"

            img = img / 255
            label_imgs[j] = np.expand_dims(img, 2)
        
        for j in range(self.batch_size):
            input_imgs[j], label_imgs[j] = self.augment(input_imgs[j], label_imgs[j])

        return input_imgs, label_imgs

    def augment(self, image, label, extra_augmentations=True):
        '''Augments the input images'''
        if random.choice([True, False]):
            image = np.flipud(image)
            label = np.flipud(label)
        if random.choice([True, False]):
            image = np.fliplr(image)
            label = np.fliplr(label)
        if random.choice([True, False]):
            image = np.rot90(image)
            label = np.rot90(label)
        if extra_augmentations:
            #adjust contrast
            if random.choice([True, False]):

                image = change_contrast(image)
            #adjust brightness
            if random.choice([True, False]):
                image = change_brightness(image)

            #add some blurring
            if random.choice([True,False]):
                #image = add_random_blurring(image)
                pass
        return image, label

def change_contrast(img, max_contrast_factor=0.5):

    original_contrast = img.max() - img.min()
    normalized_contrast = original_contrast/255

    normalized_img = img/original_contrast
    
    old_average = np.average(img)

    #determine actual contrast factor and whether we multiply (decrease contrast) or divide by the number (increase contrast)

    increase_decrease = random.choice([1,-1])*random.random()
    contrast_factor = ((1-max_contrast_factor))**increase_decrease
    new_contrast = contrast_factor*normalized_contrast

    new_img = (normalized_img*new_contrast*255)

    I = np.ones(shape=np.shape(img))
    new_average = np.average(new_img)
    new_img = np.add(new_img, I*(old_average-new_average))
    new_img[new_img > 255] = 255
    new_img[new_img < 1] = 0

    return new_img


def change_brightness(img:np.array, intensity=50):

    I = np.ones(shape=np.shape(img))*random.random()

    new_img = np.add(img,intensity*random.choice([-1])*I)

    new_img[new_img > 255] = 255
    new_img[new_img < 1] = 0

    return new_img

def add_random_blurring(img, sig_intensity=1.5):

    z = int(sig_intensity*(40+10)*random.random())
    if z%2 ==0:
        z +=1
    new_img = cv2.GaussianBlur(img,(z,z),0)

    return new_img

