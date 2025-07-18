

import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

def get_file_names(folder:str|Path, pattern:str, exclude:list[str]=['/.'], include:list[str]=[''])->list[Path] | list[str]:
    """
    Get a list of file names from a folder that match a pattern and optionally exclude/include certain files.
    
    Parameters:
    folder (str): The folder to search for files.
    pattern (str): The pattern to match file names.
    exclude (list): A list of strings to exclude from the list of file names (the file name, not full path).
    include (list): A list of strings to include in the list of file names (the file name, not full path).
    Usage Note: The variables include and exclude are redundant with the pattern, but can be useful for better 
          readability in contexts of more complex filtering. Use pattern for filtering the paths, use include 
          and exclude for filtering the file names.
    
    Returns:
    list: A list of file names that match the pattern and do not contain any strings in the exclude list and contain all strings in the include list.
    """

    file_names = []
    for file in Path(folder).glob(pattern):
        if all([excl not in str(file.name) for excl in exclude]) and all([incl in str(file.name) for incl in include]):
            file_names.append(Path(file))
    return file_names

def list_fovs(folder:str|Path, pattern:str='fov*/predict/')->list[Path]:
    """
    List the field of views (FOVs) in a folder that match a pattern.
    
    Parameters:
    folder (str): The folder to search for FOVs.
    pattern (str): The pattern to match FOV names.
    
    Returns:
    list: A list of FOVs that match the pattern.
    """
    fovs = []
    for fov in Path(folder).glob(pattern):
        if fov.is_dir():
            fovs.append(fov)
    return fovs

def load_image_tensor(image:str|Path|np.ndarray|Image.Image)->torch.Tensor:
    """
    Load an image and returns a PyTorch tensor.

    """
    loaded_image = load_image(image)
    return F.to_tensor(loaded_image).to(torch.float32)

def load_image(image:str|Path|np.ndarray)->np.ndarray:
    """
    Load an image and return a numpy array.
    """
    if isinstance(image, str|Path):
        loaded_image = Image.open(image)

    elif isinstance(image, np.ndarray):
        loaded_image = Image.fromarray(image)

    elif isinstance(image, Image):
        loaded_image = image
    else:
        raise ValueError('Image must be a path to an image or numpy array')
    
    if loaded_image.mode == 'I;16':

        img_8bit = loaded_image.point(lambda x: x/255)
        loaded_image = img_8bit

    loaded_image.convert('L')

    loaded_image = np.array(loaded_image)

    #this section of code removes the alpha channel, if there is one:
    if loaded_image.shape[-1] == 4:
        loaded_image = loaded_image[..., :3]

    return np.array(loaded_image)

def save_output(output_np, output_path, convert_png = True):
    ''''
    This function saves a numpy array as an image to a specified path.
    output_np (np.ndarray): numpy array of the image
    output_path (str|pathlib.Path): path to save the image
    '''
    os.makedirs(Path(output_path).parent, exist_ok=True)
    if convert_png:
        output_path = Path(output_path).with_suffix(".png")


    if output_np.max() <= 1.01:
        output_np = output_np * 255
    output_image = Image.fromarray(output_np.astype(np.uint8))  # Convert to 8-bit grayscale image
    output_image.save(output_path)