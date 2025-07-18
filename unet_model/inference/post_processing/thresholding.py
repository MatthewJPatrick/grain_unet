'''Utils

This module contains some utilities for image post processing
'''
import warnings
from skimage import measure, io
import numpy as np


def double_thresh(img, conservative_thresh=160, liberal_thresh=200, \
                    invert_double_thresh=True, **kwargs):
    '''
    This wraps confidence_overlap, it finds the "double threshold"
    '''
    if invert_double_thresh:
        lib_img = img < liberal_thresh
        cons_img = img < conservative_thresh
    else:
        lib_img = img > liberal_thresh
        cons_img = img > conservative_thresh

    return confidence_overlap(cons_img, lib_img)

def confidence_overlap(conservative, liberal):
    '''
    This code takes two images of a grain boundary, one which only includes high
    certainty grain boundaries (conservative) and another which includes more
    boundaries with lower certainty (liberal). It combines them by adding to the
    conservative image boundaries in the liberal image which overlap with the
    conservative image

    conservative - (N, M) 2D image
    liberal - (N, M) 2D image
    '''
    output = np.copy(conservative)

    lib_lab = measure.label(liberal)
    overlap = np.unique(lib_lab[conservative])

    for label in overlap:
        if label == 0:
            continue
        output[lib_lab == label] = 1

    return output

