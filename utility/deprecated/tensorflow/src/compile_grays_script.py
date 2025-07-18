from utils.post_processing_utils import compile_imgs
from pathlib import Path
from skimage.io import imread, imsave, imshow, show
from skimage import io
import numpy as np
from grain_sequence import change_brightness, change_contrast, add_random_blurring

def get_imgs(path:Path or str = "", pattern:str = "*.png"):
    
    imgs = np.array([])
    path = Path(path)
    for fname in path.glob(pattern):
        if not fname.is_file(): 
            continue

        img = io.imread(fname)
        if len(img.shape) > 2:
            img = img[:, :, 0]

        if len(imgs) == 0:
            imgs = img
        else:
            imgs = np.dstack((imgs, img))
    return imgs


# for ii in range(12):
#     number = ii +1
#     folder = f"../Data/Al-324-asdep/fov{number}/predict_1024"
#     print(folder)
#     imgs = np.stack(get_imgs(folder))
#     compiled = compile_imgs(imgs)
#     imsave(f"../tests/compiled_{number}.png", compiled)
import cv2

img = cv2.imread("Data/training_validation_data/training/2315_train/aligned/2hr2315_1.png")

for ii in range(10):
    print(ii)
    contrast = change_contrast(img)
    brightness = change_brightness(contrast)
    blurred = add_random_blurring(brightness)
    # cv2.imwrite(f"./tests/change_CB_{ii}.png", contrast.astype("uint8"))
    # cv2.imwrite(f"./tests/change_brightness_{ii}.png", brightness.astype("uint8"))
    cv2.imwrite(f"./tests/change_all_{ii}.png", blurred.astype("uint8"))




