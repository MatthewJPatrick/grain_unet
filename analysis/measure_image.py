__author__ = "Matthew Patrick, Rosnel Leyva-Cortes, Lauren Grae"

import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from scipy.ndimage import label, center_of_mass
import cv2

def measure_image(image_path,csv_save_path:str='',centroid_save_path:str='', save_csv:bool=True, save_image:bool=True, scale_factor_nm_per_pixel:float=1.0):
    ''''
    This function is equivalent to the Analyze Particles function in ImageJ
    It takes a single grayscale image, binarizes it, labels the objects, and calculates their areas and centroids.
    It then opetionally saves the results to a CSV file and draws the centroids on the image.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    csv_save_path : str
        Path to save the CSV file.
    centroid_save_path : str
        Path to save the image with centroids.
    save_csv : bool
        Whether to save the CSV file.
    save_image : bool
        Whether to save the image with centroids.
    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the areas and centroids of the objects in the image.
    '''

    # Load image
    if Path(image_path).stem[0] == '.':
        return None
    if "centroid" in str(image_path):
        return None
    image_name = Path(image_path).name
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale

    # Convert image to numpy array and binarize
    image_array = np.array(image)
    binary_array = (image_array == 255).astype(int)  # Objects are 0, boundaries are 1
    
    # Label objects in the array
    labeled_array, num_features = label(binary_array == 0)  # Label the objects (0s in the binary array)

    # Calculate area and centroid of each object
    areas = []
    centroids = []
    for i in range(1, num_features + 1):
        area = np.sum(labeled_array == i)  # Count pixels per label
        centroid = center_of_mass(labeled_array == i)
        centroid = (centroid[0]*scale_factor_nm_per_pixel, centroid[1]*scale_factor_nm_per_pixel)  # Calculate centroid
        scaled_area = area * (scale_factor_nm_per_pixel**2)  # Convert area to nm^2
        areas.append({'Object Label': i, 'Area': scaled_area})  # Convert to nm^2
        centroids.append((centroid, i))  # Store centroid with its label

    # Create DataFrame
    df = pd.DataFrame(areas)
    df['Centroid Y'], df['Centroid X'] = zip(*[centroid[0] for centroid in centroids])  # Split centroids into their components

    # Save to CSV
    if save_csv:
        df.to_csv(f'{Path(csv_save_path).absolute()}/{image_name}_grain_areas_and_centroids.csv', index=False)
        print(f"Areas and centroids of objects saved to /{csv_save_path}/{image_name}_grain_areas_and_centroids.csv")

    # Draw centroids and labels on the image
    if save_image:
        image_color = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White color for text
        thickness = 1
        for centroid, grain_label in centroids:
            # Draw a red dot at each centroid
            y, x = int(centroid[0]), int(centroid[1])
            cv2.circle(image_color, (x, y), 5, (255, 0, 0), -1)  # Red color
            # Put label number near each centroid
            cv2.putText(image_color, str(grain_label), (x + 6, y), font, font_scale, font_color, thickness)

        # Save the modified image
        print(f'/{Path(centroid_save_path).absolute()}/{image_name}_centroids.png')
        Image.fromarray(image_color).save(f'{Path(centroid_save_path).absolute()}/{image_name}_centroids.png')
        print(centroid_save_path)
        print(f"Modified image saved as '/{centroid_save_path}/{image_name}_centroids.png")
    return df

def measure_folder(folder:str|Path, pattern = "*.png", scale_factor_nm_per_pixel:float=1.0):
    '''
    This function measures all images in a folder. It assumes that the images are in the format 'fov*/raw/*.tif'.
    It saves the results to a CSV file and draws the centroids on the images.
    '''
    folder = Path(folder)
    for image_path in folder.glob(pattern):
        print(f"Processing {image_path}")
        data = measure_image(image_path, csv_save_path=folder, centroid_save_path=folder, scale_factor_nm_per_pixel=scale_factor_nm_per_pixel)

if __name__ == "__main__":
    folder = Path('/Volumes/Samsung_T5/Matthew/TEM/Al-324/asdep_predict_grae_retraced_train_100_512')
    pattern = "*.png"
    measure_folder(folder, pattern, scale_factor_nm_per_pixel=1668/512)





