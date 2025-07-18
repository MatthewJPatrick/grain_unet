import sys
sys.path.append("/Users/matthew/Documents/research/code/grain_boundary_detection/grain_unet_working/")
from unet_model.unet import UNet
from utility.settings import *
from utility.settings import init, init_training
from utility.file_manager import load_image_tensor
import torch
import skimage.io as io
from skimage import transform
import re
from unet_model.inference.post_processing.Overlays import compile_imgs
from torchvision.transforms import functional as F
import numpy as np
import skimage

init()
init_training()
from pathlib import Path

def get_image_list(folder:str|Path):
    return [str(item) for item in list(folder.glob('*.png'))]

def load_img(label_path:str|Path, invert:bool = False, rescale:tuple = (256, 256), gt= True)->torch.Tensor:
    
    img = io.imread(label_path, as_gray=True)
    # if gt: 
    #     img = 255-img
    #     img =skimage.morphology.dilation(img)
    #     img = 255-img
    if invert:
        img = 255 - img   
    if img.shape != rescale:
        img = transform.resize(img, rescale)

    threshold = skimage.filters.threshold_otsu(img)
    img[img < threshold] = 0
    img[img >= threshold] = 1
    torch_tensor = load_image_tensor(img)

    return torch_tensor

def compute_loss(loss_fn = None, 
                 label:torch.Tensor = None, 
                 prediction:torch.Tensor = None):
    
    loss_fn = torch.nn.BCELoss()
    loss = loss_fn(prediction, label)
    return loss.item()

def fov_loss_with_tilts(gt_path:str|Path, image_folder:str|Path, addons:list[str]=None)->list:
    gt = load_img(gt_path, invert=False)
    inferences_family = []
    inferences = []
    losses = []
    for addon in addons:
        path = Path(str(image_folder)+addon)
        inferences_family.append(sorted(get_image_list(path), key=extract_last_number))
    epochs = []

    for ii in range(len(inferences_family[0])):
        img_paths = [inference[ii] for inference in inferences_family]
        epochs.append(extract_last_number(img_paths[0]))
        imgs = []
        for img in img_paths:
            imgs.append(io.imread(img)/255)
        imgs = np.dstack(imgs)
        compiled_img = compile_imgs(imgs, compilation="min")
        
        inferences.append(compiled_img)
    
    for ii, inference in enumerate(inferences):
        epoch = epochs[ii]
        pred = F.to_tensor(inference).to(torch.float32)
        loss = compute_loss(label=gt, prediction=pred)
        print(f'Image: {inference}, Loss: {loss}')
        losses.append([epoch, loss])

    return losses

def fov_loss(gt_path:str|Path, image_folder:str|Path)->list:
    
    inferences = get_image_list(image_folder)
    gt = load_img(gt_path, invert=False)
    
    losses = []
    for inference in inferences:
        epoch = extract_last_number(inference)
        pred = load_img(inference)
        loss = compute_loss(label=gt, prediction=pred)
        print(f'Image: {inference}, Loss: {loss}')
        losses.append([epoch, loss])
    return losses

def save_loss(losses:list, output_path:str|Path):
    with open(output_path, 'w') as f:
        f.write('Image, Loss\n')
        for loss in losses:
            f.write(f'{loss[0]}, {loss[1]}\n')

def extract_last_number(file_path: str) -> int:
    """
    Extract the last number before '.png' in a given file path string.

    Args:
        file_path (str): The file path string.

    Returns:
        int: The extracted number.

    Raises:
        ValueError: If no number is found before '.png'.
    """
    file_path = str(file_path)
 
    match = re.search(r'(\d+)\.png$', file_path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f'No number found before .png in {file_path}.')


if __name__ == "__main__":
    gt_path = Path('Data/image_data/2025-01-28_test-data-over-epochs/al-324-200C_15min_aligned_fov1_0/predict_al-324-200C_15min_aligned_fov1_0_testing_saving_300.png')
    image_folder = Path('Data/image_data/2025-01-28_test-data-over-epochs/al-324-200C_15min_aligned_fov1_0')

    #image_folder = Path("unet_model/losses/299")
    #gt_path      = Path("unet_model/losses/299/testing_saving.pth.png")


    output_path = Path('./15_1_segmentation-comparison.csv')
    losses = fov_loss(gt_path, image_folder)#, addons = ['_0', '_+1', '_-1'])
    save_loss(losses, output_path)

if __name__ == "__main__m":
    import PIL
    gt_path = Path('Data/image_data/2025-01-28_test-data-over-epochs/al-324-200C_15min_aligned_fov1_0/predict_al-324-200C_15min_aligned_fov1_0_testing_saving_300.png')

    image_1 = PIL.Image.open('Data/image_data/2025-01-28_test-data-over-epochs/al-324-200C_15min_aligned_fov1_0/predict_al-324-200C_15min_aligned_fov1_0_testing_saving_300.png')
    image_2 = PIL.Image.open('Data/image_data/2025-01-28_test-data-over-epochs/al-324-200C_15min_aligned_fov1_0/predict_al-324-200C_15min_aligned_fov1_0_testing_saving_300.png')

    image_1 = F.to_tensor(image_1).to(torch.float32)
    image_2 = F.to_tensor(image_2).to(torch.float32)

    loss = compute_loss(label=image_1, prediction=image_1)
    print(image_1)
    print(image_2)
    print(torch.equal(image_1, image_2))
    print(loss)

