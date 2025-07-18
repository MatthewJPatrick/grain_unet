from PIL import Image
from pathlib import Path
import numpy as np
import os


def get_file_names(folder:str|Path, pattern:str, exclude:list[str]=[], include:list[str]=[''])->list[Path] | list[str]:
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
            file_names.append(str(file))
    return file_names

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

        img_8bit = loaded_image.point(lambda x: x/256)
        loaded_image = img_8bit

    loaded_image.convert('L')

    return np.array(loaded_image)

def save_output(output_np, output_path):
    ''''
    This function saves a numpy array as an image to a specified path.
    output_np (np.ndarray): numpy array of the image
    output_path (str|pathlib.Path): path to save the image
    '''
    os.makedirs(Path(output_path).parent, exist_ok=True)
    if output_np.max() <= 1.01:
        output_np = output_np * 255
    output_image = Image.fromarray(output_np.astype(np.uint8))  # Convert to 8-bit grayscale image
    output_image.save(output_path)

def compile_imgs(imgs, compilation='min', **kwargs):
    '''Compiles grayscale images

    imgs - a numpy depth-wise stack of images (i.e. shape (512,512,n))
    compilation - compilation technique. 'min' or 'max' supported
    '''
    if compilation == 'min':
        img_compiled = np.amin(imgs, axis=-1)
    elif compilation == 'max':
        img_compiled = np.amax(imgs, axis=0)
    elif compilation == 'sum':

        img_compiled = np.add.reduce(imgs, axis=-1)
        img_compiled = ((img_compiled / img_compiled.max()))*255

    else:
        Warning(f'Compilation technique \'{compilation}\' not recognized', stacklevel=2)
        img_compiled = imgs[:,:,0]

    return img_compiled

def overlay_any_image(image_folder:str, output_path:str='overlayed_image.png'):
    image_paths = get_file_names(image_folder, pattern = "*.tif", include=["SAD", "Ru"], exclude = ["Pt", "Al", "Gamma"])
    images = []
    for image_path in image_paths:
        image = load_image(image_path)        
        images.append(image)
    images = np.array(images)
        
    print(images.shape)

    overlayed_image = compile_imgs(images, compilation='max')
    save_output(overlayed_image, output_path)

if __name__ == "__main__":
    overlay_any_image('/Users/matthew/Downloads/2025-01-21_Ru191001_02-thinner')
