#Standard Library Imports
import torch
import numpy as np
import os
from pathlib import Path
#Local Imports
from utility.settings import *
import utility.file_manager as fm
import unet_model.unet as unet
from tqdm import tqdm
import utility.transformation as t

# Function to run inference
def run_inference(model:unet.UNet, image:str|Path|np.ndarray|torch.Tensor, device = DEVICE_COMPUTE_PLATFORM, 
                  image_transform = t.inference_transforms(TARGET_RESOLUTION))->np.ndarray:
    # Load and transform image
    if isinstance(image, torch.Tensor):
        input_tensor = image
    else:
        input_tensor = fm.load_image_tensor(image)
        input_tensor = image_transform(input_tensor).unsqueeze(0).to(device)  # Add batch dimension and move to deviceto(device) 
    
    # Perform inference and apply sigmoid to get probabilities
    with torch.no_grad():
    
        output = model(input_tensor)
        output = torch.sigmoid(output)
        output_np = output.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU as a numpy array

    return output_np

# Main function to run single inference process
def single_inference(image_path:str|Path, output_path:str|Path, model:unet.UNet|str|Path): 

    device = torch.device(DEVICE_COMPUTE_PLATFORM)

    if isinstance(model, unet.UNet):
        model = model
    elif isinstance(model, str|Path):
        print("Running a single inference. loading model: ", model)
        model = unet.load_model_weights(model, device) 
    else:
        raise ValueError('The model parameter must be an instance of the UNet class or a path to a model weights file.')

    output_np = run_inference(model, image_path, device)
    fm.save_output(output_np, output_path)
    #[REF] save only once, there's a bug here

def multi_inference(image_paths, output_paths, model:unet.UNet|str|Path): #add model_pth for bulk export
 
    if isinstance(model, unet.UNet):
        model = model
    else:
        model = unet.load_model_weights(model, device = DEVICE_COMPUTE_PLATFORM)

    for image_path, output_path in tqdm(zip(image_paths, output_paths), total = len(image_paths), desc= 'Inference Progress'):
        if Path(image_path).name[0] == '.':
            continue

        single_inference(image_path, output_path, model)
        output_np = run_inference(model, image_path, DEVICE_COMPUTE_PLATFORM)
        fm.save_output(output_np, output_path)

def multi_folder_inference(model_path=MODEL_PARAMS, folder='', pattern:str="fov*/*.tif", resolution = TARGET_RESOLUTION, prefix = PREFIX):

    ''' This function will take in the test data directory and create inferences for the different fovs of those images'''
    
    folder = Path(folder)
    
    model = unet.load_model_weights(model_path=model_path, device = DEVICE_COMPUTE_PLATFORM )
    print(f"Loaded model from {model_path}")
    
    print("Compute platform is: ", DEVICE_COMPUTE_PLATFORM)

    print(f"Looking for images in: {folder}/{pattern}")
    image_paths = fm.get_file_names(folder, pattern = pattern)
    if len(list(image_paths)) == 0:
        raise ValueError('\n\nNo images found in the specified folder')
    else:
        print(f"Found {len(list(image_paths))} images")
    
    save_paths = [os.path.join(path.parent.parent, f'predict_{prefix}_{resolution}', f'predict_{path.name}') for path in image_paths]
    multi_inference(image_paths, save_paths, model)

def in_situ_inference(folder_name, pattern):

    ''' This function will take in the test data directory and create inferences for the different fovs of those images'''
    folder = Path(folder_name)
    model = unet.load_model_weights(model_path=MODEL_PARAMS, device = DEVICE_COMPUTE_PLATFORM )
    print(f"Loaded model from {MODEL_PARAMS}")
    print("Compute platform is: ", DEVICE_COMPUTE_PLATFORM)
    print(f"Looking for images in: {folder}/{pattern}")
    image_paths = fm.get_file_names(folder, pattern = pattern)
    if len(list(image_paths)) == 0:
        raise ValueError('\n\nNo images found in the specified folder')
    else:
        print(f"Found {len(list(image_paths))} images")
    save_paths = [os.path.join(path.parent, f'predict_{MODEL_NAME}_{TARGET_RESOLUTION}', f'predict_{path.name}') for path in image_paths]
    multi_inference(image_paths, save_paths, model)

    return os.path.join(folder, f'predict_{MODEL_NAME}_{TARGET_RESOLUTION}') #return the folder where the predictions are saved
