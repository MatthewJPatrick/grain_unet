
import os
from MultiInferenceRunner import run_inference, load_model, save_output
from pathlib import Path
from typing import Generator
from tqdm import tqdm
import torch
from torch import nn
import cv2
import csv

def multiModel_MultiImage(images : list | Generator = [], model_paths : list |Generator = [], device : str = 'mps', save_directory_parent:str|Path = 'Data/data_for_JMR', model_filter = '.pth') -> None:
    
    chkdir(save_directory_parent)
    model_paths = list(model_paths)
    images = list(images)

    for model_path in tqdm(model_paths, total = len(model_paths), desc = 'Models'):
        for model_number in model_filter:
            if model_number in str(model_path):
                print(f'Loading model: {model_path.stem}')
                model = load_model(model_path, device=device)
                for image in tqdm(images, total=len(images), desc = 'Images'):
                    save_folder = os.path.join(save_directory_parent, image.stem)
                    chkdir(save_folder)
                    save_path = os.path.join(save_folder, f'predict_{image.stem}_{model_path.stem}.png')
                    inference = run_inference(model, image, device)
                    save_output(inference, save_path)
                
        

def chkdir(directory:str | Path = 'Data/data_for_JMR') -> None:
    '''
    This function will check if a directory exists and if not create it
    '''
    if not os.path.isdir(directory):
        os.mkdir(directory)

def get_loss_folder(folder:str| Path, loss_function = nn.BCELoss(), gt_path:str|Path = None, device:str = 'mps') -> None:
    '''
    This function will take in a folder of images and compare them to a ground truth image to get the loss of the images
    '''
    gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_image = cv2.resize(gt_image, (512, 512))

    gt_image = 1-gt_image/255
    gt_data = torch.tensor(gt_image)
    
    images_for_comparison = sorted(list(Path(folder).glob('*.png')))
    losses = []
    for image_path in tqdm(images_for_comparison, total = len(images_for_comparison), desc = 'Images'):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        image_data = torch.tensor(image/255)
        loss = loss_function(image_data, gt_data)
        losses.append([image_path.stem, loss.item()])
    return losses

if __name__ == "__main__":
    model_paths = sorted(list(Path('/Users/matthew/Documents/research/code/grain_boundary_detection/grain_unet_working/Data/model_weights/lauren_model_weights').glob('*.pth')))
    print(model_paths)
    images = list(Path('/Users/matthew/Documents/research/code/grain_boundary_detection/data/Summer2024_LHG-Tracing/2025-01-14_test-data-over-epochs/original_images').glob('*.png'))
    print(images)
    multiModel_MultiImage(images, model_paths, device='mps', save_directory_parent='/Users/matthew/Documents/research/code/grain_boundary_detection/data/Summer2024_LHG-Tracing/2025-01-14_test-data-over-epochs/data_for_JMR_3')
    
    # base_path = "/Users/matthew/Documents/research/code/grain_boundary_detection/data/Summer2024_LHG-Tracing/2025-01-28_test-data-over-epochs/"
    # FOV = "al-324-200C_15min_aligned_fov3"
    # tilt = "-1"
    # losses = get_loss_folder(folder = f'{base_path}/data_for_JMR/{FOV}_{tilt}', 
    #                          gt_path = f'{base_path}/tracings/{FOV}.png')
    # loss_table = []
    # ii= 10
    # for loss in losses:
    #     ii+=5
    #     loss_table.append([ii, loss[1]])

    # csv.writer(open(f'{base_path}/data_for_JMR/{FOV}_{tilt}/{FOV}_{tilt}.csv', 'w+')).writerows(loss_table)
    