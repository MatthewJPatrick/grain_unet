'''
Inference Runner
Reads config for paths containing images and outputs segmentations without post-processing
'''
__author__ = "Matthew Patrick, Lauren Grae, Rosnel Leyva-Cortes" 

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from skimage import io
import numpy as np
import postprocessing.post_process
import unetModel.unet as unet  # Ensure this imports your UNet model class
import configparser
from pathlib import Path
import os
import sys
import postprocessing.Overlays
# Load model parameters
config = configparser.ConfigParser()
config.read('MyUnetConfig.ini')

INFERENCE = config['INFERENCE_PARAMS']['INFERENCE']
MODEL_PARAMS = config['PATH']['MODEL_PARAMS']
MODEL_NAME = MODEL_PARAMS.split('/')[-1].split('.')[0]
TEST_DATA_DIR = config['INFERENCE_PARAMS']['N_TEST_DATA_DIR']
PREDICT_DATA_DIR = config['INFERENCE_PARAMS']['OVERLAY_DATA_DIR']
TARGET_RESOLUTION = int(config['INFERENCE_PARAMS']['OUTPUT_RESOLUTION'])
FINAL_OUTPUT_DIRECTORY = config['INFERENCE_PARAMS']['FINAL_OUTPUT_DIR']

PREFIX = MODEL_PARAMS.split('/')[-1].split('.')[0]

args_pp = { # for post processing 
            'compilation': config['POST_PROCESS']['COMPILATION'],
            'n_dilations': int(config['POST_PROCESS']['N_DILATIONS']),
            'liberal_thresh': int(config['POST_PROCESS']['LIBERAL_THRESHOLD']),
            'conservative_thresh': int(config['POST_PROCESS']['CONSERVATIVE_THRESHOLD']),
            'invert_double_thresh': bool(config['POST_PROCESS']['INVERT_DOUBLE_THRESHOLD']),
            'min_grain_area': int(config['POST_PROCESS']['MIN_GRAIN_AREA']),
            'prune_size': int(config['POST_PROCESS']['PRUNE_SIZE'])
    }

# Define transformations
image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),# Convert image to grayscale
    transforms.Resize((TARGET_RESOLUTION, TARGET_RESOLUTION)),#Resize the image to 256x256 pixels
    transforms.ToTensor(), 
])

def bulk_compile_and_pp(pattern=f'fov*/predict_{PREFIX}_{TARGET_RESOLUTION}/',folder=PREDICT_DATA_DIR, post_process = True):
    
    #folder = PREDICT_DATA_DIR
    predictions = []
    image_formats = '*.png'  # Image formats to look for
    print(f'{folder}/{pattern}')
    for FOV_predictions in Path(folder).glob(pattern):
        #print(str(FOV_predictions))
        fnames = []
        if FOV_predictions.is_dir():
            #print('is dir')
            for image in FOV_predictions.glob(image_formats):
                if 'trace' not in str(image.name):
                    if 'post_process' not in str(image.name):
                        fnames.append(str(Path(image)))

        ii=0
        for fname in (fnames):
            print(fname)
            if ii == 0: 
                predictions = io.imread(fname)
                ii+=1
                #print(fname)
            else:
                print(str(fname))
                img = io.imread(fname)
                predictions = np.dstack((predictions,img))

        print([predictions.shape])
        compiled = postprocessing.Overlays.compile_imgs(predictions, compilation='max')
        if not os.path.isdir(os.path.join(FOV_predictions, 'post_process')):
            os.mkdir(os.path.join(FOV_predictions, 'post_process'))
        save_path_comp = os.path.join(FOV_predictions,f'post_process/compiled_{Path(fname).stem}.png')
        save_path_post = os.path.join(FOV_predictions,f'post_process/postprocess_{Path(fname).stem}.png')
        save_output(compiled*255,save_path_comp)
        if post_process:
            post_processed = postprocessing.post_process.post_process(predictions, **args_pp)
            save_output(post_processed, save_path_post)
            
            if FINAL_OUTPUT_DIRECTORY:
                final_save_dir = os.path.join(FINAL_OUTPUT_DIRECTORY, f'predict_{PREFIX}_{TARGET_RESOLUTION}')
                if not os.path.isdir(final_save_dir):
                    os.mkdir(final_save_dir)
                save_path_final_output = os.path.join(final_save_dir,f'postprocess_{Path(fname).stem}.png')
                save_output(post_processed, save_path_final_output)

def n_test(model_path=MODEL_PARAMS, folder='', pattern:str="fov*/*.tif", resolution = str(TARGET_RESOLUTION)):
    ''' This function will take in the test data directory and create inferences for the different fovs of those images'''
    
    ##compute platform check 
    if torch.backends.mps.is_available():
        device_compute_platform = "mps"
    elif torch.cuda.is_available(): 
        device_compute_platform = "cuda"
    else:
        device_compute_platform = "cpu"
    ##end compute platform check 

    # Function will run inferences within a rigid file structure and create a prediction folder with inferences
    model = load_model(model_path=model_path, device = device_compute_platform )
    folder = Path(folder)

    paths = folder.glob(pattern)

    for path in paths:
        print(path)
        #generate a savepath ased on the path
        save_path = os.path.join(path.parent.parent, f'predict_{PREFIX}_{resolution}', f'predict_{path.name}')
        inference = run_inference(model, path, device=device_compute_platform)
        print('Inferring: ', path)
        save_output(inference, save_path)

# Function to load model
def load_model(model_path, device):
    num_class = 1  # Assuming binary segmentation
    model = unet.UNet(num_class)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to run inference
def run_inference(model, image_path, device):
    # Load and transform image
    image = Image.open(image_path)
    if image.mode == 'I;16':
        img_8bit = image.point(lambda x: x/255)
        image = img_8bit
        
    image.convert('L')  
    input_tensor = image_transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
  # Apply sigmoid to get probabilities
        output = torch.sigmoid(output)
        #output = (output > 0.5).float()  # Thresholding to get binary output
        output_np = output.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU

    return output_np

# Function to save the output mask as an image
def save_output(output_np, output_path):
    if not Path(output_path).parent.is_dir():
        os.mkdir(Path(output_path).parent)
    output_image = Image.fromarray((output_np * 255).astype(np.uint8))  # Convert to 8-bit grayscale image
    output_image.save(output_path)

# Main function to run single inference process
def singleInference(image_path, output_path, model_path = MODEL_PARAMS): 
    ##compute platform check 
    if torch.backends.mps.is_available():
        device_compute_platform = "mps"
    elif torch.cuda.is_available(): 
        device_compute_platform = "cuda"
    else:
        device_compute_platform = "cpu"
    ##end compute platform check 

    print("Compute platform is: ", device_compute_platform) #for debugging my lil GPU issues - Rosnel 
    device = torch.device(device_compute_platform)
    model = load_model(model_path, device) 
    output_np = run_inference(model, image_path, device)
    save_output(output_np, output_path)

# Main function to run benchmarking inference process
def bulkInference(image_path, output_path, model_pth): #add model_pth for bulk export
    ##compute platform check 
    if torch.backends.mps.is_available():
        device_compute_platform = "mps"
    elif torch.cuda.is_available(): 
        device_compute_platform = "cuda"
    else:
        device_compute_platform = "cpu"
    ##end compute platform check 

    #print("Compute platform is: ", device_compute_platform) #for debugging my lil GPU issues - Rosnel 
    device = torch.device(device_compute_platform)
    model = load_model(model_pth, device) #for batch export, change MODEL_PARAMS to model_pth
    output_np = run_inference(model, image_path, device)
    save_output(output_np, output_path)
    #print('Success!', )

if __name__ == "__main__":
    import argparse
    import unetModel.userInterface as userInterface
    #pattern = "fov*/raw/*.tif" #Original for new "test data", do not lose
    pattern = "fov*/raw/*.png"
    pattern_post = f"fov*/predict_{MODEL_NAME}_{TARGET_RESOLUTION}"
    userInterface.logoPrint()


    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="U-Net Inference Runner")
        parser.add_argument("folder_of_images", type=str, help="Path to the input image")
        parser.add_argument("output_path", type=str, help="Path to save the output image")
        args = parser.parse_args()

        singleInference(args.image_path, args.output_path)
    else: #no args are passed, assuming n_test functionality use your own pattern
        #sif(config['POST_PROCESS']['PP_ACTIVATE']=='False'):
        if INFERENCE == "True":
            n_test(folder = TEST_DATA_DIR, pattern = pattern) 
        
        if (config['POST_PROCESS']['PP_ACTIVATE']=='True'):
            bulk_compile_and_pp(folder=PREDICT_DATA_DIR, pattern = pattern_post)
        else:
            pass


    
    