import cv2
from skimage import io
from pathlib import Path
import numpy as np
import sys
import os, cv2
sys.path.append('.')
from src.post_process import post_process    
from Unet_Pytorch.alpha_converter import convert_black_to_transparent
from src.printProgressBar import print_progress_bar
from src.ImgProcessFunct import n_test
# this function returns a list of lists which maps a set of 
# conical BF video frames to pools of size N

def generate_pools(folder='', n=3):
    
    folder_path = Path(folder)
    list_of_images = sorted(list(folder_path.glob("[!._]*.png")))
    for item in list_of_images:
        print(f"{item}\n")
    n_images = len(list_of_images)
    n_pools = int(n_images / n)

    pools = []
    for ii in range(n_pools):
        pool = []
        for jj in range(n):
            pool.append(list_of_images[ii*n+jj])
        pools.append(pool)
    return pools

def post_process_pools(n_pools, prediction_folder, post_process_dictionary, out_dir = "post_processed"):
    #generate pools
    pools = generate_pools(folder = prediction_folder, n = n_pools)
    n = 1
    kwargs = post_process_dictionary
    for pool in pools:

        n+=1
        imgs = list([])
        for fname in pool:
            if not fname.is_file():
                continue
            else:
                img = io.imread(fname)
                if len(img.shape) > 2:
                    img = img[:, :, 0]

                if len(imgs) == 0:
                    imgs = img
                else:
                    imgs = np.dstack((imgs, img))

        pp = post_process(imgs, **kwargs)
        save_dir = f"{prediction_folder}/{out_dir}"
        if not Path(save_dir).is_dir():
            os.mkdir(save_dir)

        out_name = f"{save_dir}/{n_pools}_{n}_post.png"

        pp = convert_black_to_transparent(pp)

        io.imsave(out_name, pp*255)
        print_progress_bar(n, len(pools))

def process_in_pools(folder, n_pool, res):

    pattern_raw = pattern_fov + pattern_raw
    n_test(res, res, test_image_dir, pattern_raw, network_file)

if __name__ == "__main__":
    folder = "/Volumes/HPP600/Matthew/2024-01-29-133420-Other/2024-01-29-133420-Other/predict_1024"
    dictionary = {

    'lib_thresh': [250],
    'con_thresh' : [245],
    'prune_sizes' : [0],
    'dils' : [0],
    'min_areas':[0],
    'compilation':'min'

}
    for n_pool in [3,4,5]:
        post_process_pools(n_pools=n_pool, prediction_folder=folder, post_process_dictionary=dictionary, out_dir="post_processed")










