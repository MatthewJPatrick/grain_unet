'''
notes from LG:

this is going to produce an individaul table for EVERY image in your input folder and then a summary folder
I haven't yet written a parameter to toggle that off and on; I will, but I figured more info better

to run put path to input folder and path to an output folder in terminal

'''
import os
import sys
sys.path.append(".")
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from scipy.ndimage import label, center_of_mass
import cv2
import math
from analysis.analyze_distributions.analyze_distributions import GrainDataset, compare_distributions, plot_distros
from objective_functions.validation_functions import get_objectives
from skimage import morphology
from skimage.segmentation import clear_border
import skimage.io as io

# Global table to store summary data
summary_table = []

def scale_centroids(centroids, scale_factor):
    """
    Scale the centroids by a given scale factor.
    Args:
    - centroids (list of tuples): The centroids to scale.
    - scale_factor (float): The factor by which to scale the centroids.
    
    Returns:
    - scaled_centroids (list of tuples): The scaled centroids.
    """
    scaled_centroids = [(x * scale_factor, y * scale_factor) for x, y in centroids]
    return scaled_centroids


def calculate_scale_factor(unet_diameters, hand_traced_diameters, fov_size=None):
    """
    Calculate the scale factor based on the mean diameters from two datasets.
    Args:
    - unet_diameters (list of floats): List of diameters from the U-Net dataset.
    - hand_traced_diameters (list of floats): List of diameters from the hand-traced dataset.
    
    Returns:
    - scale_factor (float): The calculated scale factor.
    """
    mean_unet_diameter = np.mean(unet_diameters)
    mean_hand_traced_diameter = np.mean(hand_traced_diameters)
    
    scale_factor = mean_unet_diameter / mean_hand_traced_diameter
    return scale_factor


def calculate_areas_and_centroids(image_path, output_dir, background = 'black', save_image = False, fov_size = None):
    """
    Calculate the areas and centroids of grains in the image and save them to CSV.
    Also draw the centroids on the image and save the image, if save_image is active.
    """
    # Load image
    image_name = Path(image_path).stem
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale
    # Convert image to numpy array and binarize
    image_array = np.array(image)


    if fov_size is not None:
        sf_nmpx = fov_size/image_array.shape[0]
    else:
        sf_nmpx = 1
    # Label objects in the array
    if background == 'white':
        binary_array = (image_array == 0).astype(int)  # Objects are 0, boundaries are 1
    elif background == 'black':
        binary_array = (image_array == 255).astype(int)  # Objects are 0, boundaries are 1
    
    skeleton = morphology.skeletonize(binary_array)
    dilated_array = morphology.dilation(skeleton)

    labeled_array, num_features = label(dilated_array == 0) #grains are black, and boundaries are white
    print(f'Found {(num_features)} grains in {image_name}')
    
    # Calculate area and centroid of each object
    areas = []
    centroids = []
    areas_list = []
    for ii in range(1, num_features + 1):
        mask = (labeled_array == ii)

        raw_area = np.sum(labeled_array == ii)*(sf_nmpx**2)  # Count pixels per label
        perimeter = np.logical_xor(morphology.binary_dilation(mask), morphology.binary_erosion(mask))
        area = raw_area + 1.5*np.sum(perimeter)*(sf_nmpx)
        centroid = center_of_mass(labeled_array == ii)
        centroid = (centroid[0]*sf_nmpx, centroid[1]*sf_nmpx)

        #determine if the grain intersects the edge of the image
        if not (np.any(mask[0,:]) or np.any(mask[-1,:]) or np.any(mask[:,0]) or np.any(mask[:,-1])):
            areas_list.append(area)
            areas.append({'Object Label': ii, 'Area (pixels)': area})
            centroids.append((centroid, ii))  # Store centroid with its label
    print("Added ", len(areas), " grains to the list")
    # Create DataFrame and save CSV for the image
    df = pd.DataFrame(areas)
    
    df['Centroid Y'], df['Centroid X'] = zip(*[centroid[0] for centroid in centroids])  # Split centroids into their components
    image_csv_path = f'{Path(output_dir).absolute()}/{image_name}_grain_areas_and_centroids.csv'
    df.to_csv(image_csv_path, index=False)
    
    if save_image:
        # Draw centroids and labels on the image
        image_color = cv2.cvtColor(dilated_array.astype("uint8")*255, cv2.COLOR_GRAY2RGB)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White color for text
        thickness = 1
        for centroid, grain_label in centroids:
            y, x = int(centroid[0]/sf_nmpx), int(centroid[1]/sf_nmpx)
            cv2.circle(image_color, (x, y), 5, (255, 0, 0), -1)  # Red color
            cv2.putText(image_color, str(grain_label), (x + 6, y), font, font_scale, font_color, thickness) 
        # Save the modified image
        modified_image_path = f'{Path(output_dir).absolute()}/{image_name}_centroids.png'
        Image.fromarray(image_color).save(modified_image_path)

    # Return centroids in a format compatible with get_objectives
    ref_centroids = [(x, y) for y, x in zip(df['Centroid Y'], df['Centroid X'])]
    ref_areas = []

    for area in areas_list:
        if area > 0.05**2 * np.mean(areas_list):
            ref_areas.append(area)

    #print(abs(len(ref_areas)-len(df['Area (pixels)'].tolist())))
          
    return df, ref_centroids, ref_areas

def read_diameters(file_name:str, delimiter:str = ','):
    '''
    Reads a text file with diameters and returns lists of floats
    '''
    with open(file_name, 'r') as file:
        data = file.readlines()
    lines = [list(line.split(delimiter)) for line in data]
    areas = [float(line[1]) for line in lines[1:-1]]
    centroids = [(float(line[2]), float(line[3])) for line in lines[1:-1]]
    return areas, centroids

def compute_grain_stats_and_save_summary(output_dir, summary_csv_path, saving = False):
    """
    Compute the total number of grains and the average grain area for each image,
    and save a summary CSV.
    """
    summary_data = []

    # Loop through each CSV in the output directory
    for file in os.listdir(output_dir):
        if file.endswith('_grain_areas_and_centroids.csv'):
            file_path = os.path.join(output_dir, file)
            image_name = Path(file).stem.replace('_grain_areas_and_centroids', '')
            
            # Read the grain area CSV
            df = pd.read_csv(file_path)
            
            # Compute total number of grains and average area
            total_grains = df.shape[0]
            average_area = df['Area (pixels)'].mean()
            
            # Append the result to the summary data
            summary_data.append({
                'Image Name': image_name,
                'Total Grains': total_grains,
                'Average Area (pixels)': average_area
            })

    # Save the summary CSV
    summary_df = pd.DataFrame(summary_data)
    if saving:
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Summary CSV saved to {summary_csv_path}")
    return summary_df

def generate_datasets(input_dir, output_dir, saving = False, measure = True, fov_size = 1):
    os.makedirs(output_dir, exist_ok=True)

    # Collect datasets from processed images
    datasets = []
    dataset_diameters = []
    datasets = []


    for file in sorted(os.listdir(input_dir)):
        if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')):
            image_path = os.path.join(input_dir, file)
            print(f'Processing {file}...')

            
            if measure:
                # Calculate grain areas and centroids for each image
                if "centroids" not in str(file):
                    df, ref_centroids, ref_areas = calculate_areas_and_centroids(image_path, output_dir, fov_size=fov_size)
            elif f'{Path(file).stem}_grain_areas_and_centroids.csv' in os.listdir(input_dir):  
                print('Found diameters in text file')
                ref_areas, ref_centroids = read_diameters(os.path.join(input_dir, f'{Path(file).stem}_grain_areas_and_centroids.csv'))
            else:
                try:
                    if "centroids" not in str(file):
                        df, ref_centroids, ref_areas = calculate_areas_and_centroids(image_path, output_dir, fov_size=fov_size)

                except:
                    Exception('bad folder, no diameters in files or images. try changing to measurement mode.')
                
                
            # Convert areas to diameters for GrainDataset (assuming grains are circular)
            diameters = [np.sqrt(area / 3.14159) * 2 for area in ref_areas]

            datasets.append({'file':file, 'centroids':ref_centroids, 'areas':ref_areas, 'GrainDataset':GrainDataset(diameters=diameters, name=file, centroids = ref_centroids)})
            #add diameters to a list of all diameters across many fields of view
            mean = np.mean(diameters)
            for diameter in diameters:
                dataset_diameters.append(diameter)
    # Once all images are processed, compute summary statistics
    summary_csv_path = os.path.join(output_dir, 'grain_summary.csv')
    compute_grain_stats_and_save_summary(output_dir, summary_csv_path, saving=saving)
    
    return datasets, dataset_diameters #return a dictionary with all of the data from that folder

def compare_folders(gt_folder, test_folder, objectives = False, measure = True, fov_size = 1661):
    '''
    Takes a folder of tracings and a folder of U-Net (or other binary) inferences and calculates statistics comparing the results
    
    Inputs: 
        -gt_folder   : str containing the path to the folder with the hand tracing (ground truth)
        -test_folder : str containing the path to the folder with the inferences or other comparison
    Note:
        It is assumed that there is a 1:1 correspondence between the FoVs present in the gt and test folders, 
        and that the 1:1 correspondence can be reconstructed by sorting the names of the files using list.sort()

    Returns tuple of (distribution_statistics, objectives)
        -distribution_statistics: a dictionary of the t-, KS, and CVM test results for the two grain size distributions
        -objectives             : a list of dictionaries each containing the objecrtive functions 'orphan_fraction' and 'disregistry'

    '''
    
    # Example: Compare the first two datasets as a test
    dataset_dicts = [None, None]
    diameters = [None, None]

    print("Now I am doing the gt_folder")
    dataset_dicts[0], diameters[0] = generate_datasets(gt_folder, gt_folder, measure = measure, fov_size = fov_size)

    print("Now I am doing the test_folder")
    dataset_dicts[1], diameters[1] = generate_datasets(test_folder, test_folder, measure=measure, fov_size = fov_size)
    
    grain_dataset_0 = GrainDataset(diameters = diameters[0], name = 'reference')
    grain_dataset_1 = GrainDataset(diameters = diameters[1], name = 'comparison')

    distribution_statistics = compare_distributions(grain_dataset_0, grain_dataset_1, lognormal=True, area=False)
    plot_distros([{'data':grain_dataset_0, 'name':grain_dataset_0.name},{'data':grain_dataset_1, 'name':grain_dataset_1.name} ], display = True, fit = True, reduced=True, hist = True)# save_path = os.path.join(test_folder, 'distribution_comparison.png'))
    plot_distros([{'data':grain_dataset_0, 'name':grain_dataset_0.name},{'data':grain_dataset_1, 'name':grain_dataset_1.name} ], display = True, fit = True, reduced=False, hist = True, xlim = (0,400), ylim = (0,0.014))# save_path = os.path.join(test_folder, 'distribution_comparison.png'))

    if objectives:
        if len(dataset_dicts[0]) == len(dataset_dicts[1]): #verify same number of FoVs
            objectives = []
            for FoV_ref, FoV_comp in zip(dataset_dicts[0], dataset_dicts[1]):
                objectives.append(get_objectives(FoV_ref['centroids'], FoV_comp['centroids']))

            # print(f'Average disregistry: {np.mean([FoV['disregistry'] for FoV in objectives])}'
            #       f'Average orphan frac: {np.mean([FoV['orphan_frac'] for FoV in objectives])}')
            return distribution_statistics,  objectives
    else:
        return distribution_statistics, None


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python model_evaluation.py <mode> <input_folder> <output_folder>")
        sys.exit(1)
    mode = sys.argv[1]
    input_folder = sys.argv[2]
    output_folder = sys.argv[3]
    fov_size = 1661 #nm/fov
    if mode == 'ri':
        compare_folders(input_folder, output_folder, measure = True, objectives = True, fov_size = fov_size)
    elif mode == 'ri':
        print(' mode RI')
        compare_folders(input_folder, output_folder,measure = True, fov_size=fov_size, objectives = True)




