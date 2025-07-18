'''
kwargs:
    'compilation': (default 'min') defines the image compilation technique
    'liberal_thresh': (default 200) liberal threshold for double threshold
    'conservative_thresh': (default 160) conservative threshold for double threshold
    'invert_double_thresh': (default True) Changes < to > in double threshold
    'n_dilations': (default 3) Number of dilations to apply in closing
    'min_grain_area': (default 100) Max size of a hole to close
    'prune_size': (default 30) Size to prune with plantcv
    'out_dict': (default False) return a dict with all the intermediate steps
'''

#Standard Library Imports
import sys, os
import numpy as np
from pathlib import Path
from skimage import morphology, io
from plantcv import plantcv as pcv
from tqdm import tqdm

#Local Imports
import utility.file_manager as fm
from unet_model.inference.post_processing.thresholding import double_thresh
from unet_model.inference.post_processing import Overlays
from utility.settings import *


# Test inline comment
def post_process(imgs, n_dilations=3, min_grain_area=100, prune_size=0, debug=False,
        out_dict=False, convert_to_trans = True, **kwargs):
    '''This tries to make clean skeletons with N Unet output image(s) from an FOV
    '''
    if len(imgs.shape) > 2:
        img_compiled = Overlays.compile_imgs(imgs, **kwargs)
    else:
        img_compiled = imgs

    img_double_thresh = double_thresh(img_compiled, **kwargs)

    img_dilated = np.copy(img_double_thresh)
    for _ in range(n_dilations):
        img_dilated = morphology.binary_dilation(img_dilated)
    img_closed = morphology.remove_small_holes(img_dilated, area_threshold=min_grain_area)

    skeleton = morphology.skeletonize(img_closed)
    pruned_skeleton, _, _ = pcv.morphology.prune(skeleton.astype('uint8'), prune_size)

    if out_dict:
        return {'compiled': img_compiled,
                'double_thresh': img_double_thresh,
                'dilated': img_dilated,
                'closed': img_closed,
                'skeleton': skeleton,
                'pruned_skeleton': pruned_skeleton
                }
    # if convert_to_trans:
    #     pruned_skeleton = convert_black_to_transparent(pruned_skeleton)
    return pruned_skeleton

def bulk_compile_and_pp(pattern=f'fov*/predict_{PREFIX}_{TARGET_RESOLUTION}/', folder=PREDICT_DATA_DIR, post_process_option=True):
    """
    Compiles predictions from multiple images and optionally applies post-processing.

    Parameters:
    pattern (str): The pattern to match prediction directories.
    folder (str): The top-level directory containing the FoVs you wish to post-process.
    post_process_option (bool): Whether to apply post-processing to the compiled images.

    Returns:
    None
    """
    compiled_fovs = Overlays.overlay_fov_generator(folder, pattern)
   
    for compiled_fov in compiled_fovs:
        fov_predictions_folder, compiled_img, fname = compiled_fov['fov_folder'], compiled_fov['img'], compiled_fov['fname']

        os.makedirs(os.path.join(fov_predictions_folder, 'post_process'), exist_ok=True)
        save_path_comp = os.path.join(fov_predictions_folder,f'post_process/compiled_{Path(fname).stem}.png')
        save_path_post = os.path.join(fov_predictions_folder,f'post_process/postprocess_{Path(fname).stem}.png')
        
        fm.save_output(compiled_fov['img'], save_path_comp)
        if post_process_option:
            post_processed = post_process(compiled_img, **args_pp)
            fm.save_output(post_processed, save_path_post)
            
            if FINAL_OUTPUT_DIRECTORY:
                final_save_dir = os.path.join(FINAL_OUTPUT_DIRECTORY, f'predict_{PREFIX}_{TARGET_RESOLUTION}')
                if not os.path.isdir(final_save_dir):
                    os.makedirs(final_save_dir, exist_ok=True)
                save_path_final_output = os.path.join(final_save_dir,f'postprocess_{Path(fname).stem}.png')
                fm.save_output(post_processed, save_path_final_output)

def in_situ_post_process(in_folder, out_folder, integration = 3):
 
    images = fm.get_file_names(in_folder, pattern = '[!.]*.png')
    images = [str(image) for image in images]
    images.sort()

    if len(images) == 0:
        raise ValueError('No images found in the specified folder')

    for ii in tqdm(range(0, len(images), integration), desc='Post-processing', total=len(images)//integration):
        image = images[ii]

        if ii + integration > len(images):
            break
        img = []
        for jj in range(integration):
            img.append(images[ii + jj])
               
        img_compiled = Overlays.compile_imgs(img, **args_pp)
        save_path_comp = os.path.join(out_folder,f'compiled_{Path(image).stem}.png')
        save_path_post = os.path.join(out_folder,f'postprocess_{Path(image).stem}.png')
        fm.save_output(img_compiled, save_path_comp)
        post_processed = post_process(img_compiled, **args_pp)
        fm.save_output(post_processed, save_path_post) 

if __name__ == '__main__':
    from skimage import io

    fnames = ['../data/test_all/10HR/2400/predict/10hr2400_1.png',
              '../data/test_all/10HR/2400/predict/10hr2401_2.png',
              '../data/test_all/10HR/2400/predict/10hr2402_3.png']

    for ind, fname in enumerate(fnames):
        if ind == 0:
            predictions = io.imread(fname)
        else:
            img = io.imread(fname)
            predictions = np.dstack((predictions, img))

    args = {
            'compilation': 'min',
            'n_dilations': 3,
            'liberal_thresh': 200,
            'conservative_thresh': 160,
            'invert_double_thresh': True,
    }

    post_process(predictions, debug=True, **args)
