'''
Data handling helper functions
'''
from skimage import io, transform
import numpy as np
import cv2 as cv

def image_generator(test_paths, target_size=(256,256)):
    '''Generator which yields normalized images from list of paths
    At some point this functionality COULD (maybe should) be built into GrainSequence
    '''
    for ind, path in enumerate(test_paths):
        img = io.imread(path)
        if len(img.shape) > 2:
            img = img[:,:,0]
        if img.shape != target_size:
            img = transform.resize(img, target_size)

        img -= img.min()
        img = img.astype('float32') / np.ptp(img)
        img = np.expand_dims(img, 0)

        yield img
