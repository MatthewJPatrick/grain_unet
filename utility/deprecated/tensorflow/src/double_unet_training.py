from utils import compile_imgs
from ImgProcessFunct import train_unet, train_loss_plt
from skimage.io import imread
import numpy as np
list_of_images = []
images = []
for image in list_of_images:
    images.append(np.array(imread(image)))
for ii in range(len(list_of_images)-1):
    if max(images[ii] - images[ii+1]) == 0:
        a = 0
        

train_unet("Data/training_validation_data/", "Data/model_weights/added_augmentation_very_epoch=500_batch=5.hdf5",epochs=500, batch_size = 5)
train_loss_plt()
#generate original unet predictions
