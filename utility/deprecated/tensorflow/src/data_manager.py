'''
Data handling helper functions
'''
from skimage import io, transform
import numpy as np
import cv2

def image_generator(test_paths, target_size=(256,256), n_stack = 1):
    '''Generator which yields normalized images from list of paths
    At some point this functionality COULD (maybe should) be built into GrainSequence
    '''
    for ind, path in enumerate(test_paths):
        if ind > len(test_paths) - n_stack:
            n_stack = 1 #for the last images, run them individually

        for ii in range(n_stack):
            img = np.zeros(target_size)
            img_lst = []
            if "._" not in path.stem and "._" not in test_paths[ind+ii].stem:
                img_tmp = io.imread(test_paths[ind+ii])
                img_lst.append(img_tmp)
                res = img_tmp.shape
            else:
                img_tmp = img
            if len(img_lst)>1:
                img = np.dstack(img_lst)
                img = np.add.reduce(img, axis=-1)
                print('\n')
                print(img)
                print('\n')
                img = ((img /img.max()))*255 #normalize and make maximum 255
        else:
            img = img_tmp
        
        if len(img.shape) > 2:
            img = img[:,:,0]
        if img.shape != target_size:
            img = transform.resize(img, target_size)
       
        img -= img.min()
        img = img.astype('float32') / np.ptp(img)
        img = np.expand_dims(img, 0)

        yield img

def path_generator(test_paths, n_stack):
    for ind, path in enumerate(test_paths):
        path_lst = []
        if ind > len(test_paths) - n_stack:
            n_stack = 1 #for the last images, run them individually
        for ii in range(n_stack):
            if "._" not in path.stem:
                path_lst.append(test_paths[ind+ii])
    return path_lst

def image_generator_stack(test_paths, target_size=(256,256), n_stack = 1):
    return