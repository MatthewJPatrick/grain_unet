'''
notes from LG:

this is going to produce an individaul table for EVERY image in your input folder and then a summary folder
I haven't yet written a parameter to toggle that off and on; I will, but I figured more info better

to run put path to input folder and path to an output folder in terminal

'''
import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from scipy.ndimage import label, center_of_mass
import cv2
import math

# Global table to store summary data
summary_table = []

def calculate_areas_and_centroids(image_path, output_dir):
    """
    Calculate the areas and centroids of grains in the image and save them to CSV.
    Also draw the centroids on the image and save the image.
    """
    # Load image
    image_name = Path(image_path).stem
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale

    # Convert image to numpy array and binarize
    image_array = np.array(image)
    binary_array = (image_array == 255).astype(int)  # Objects are 0, boundaries are 1

    # Label objects in the array
    labeled_array, num_features = label(binary_array == 1)  # Label the objects (0s in the binary array)
    #make binary-array == 1 if hand tracings w/ white backgrounds, 0 if post processed images with black backgrounds

    # Calculate area and centroid of each object
    areas = []
    centroids = []
    for i in range(1, num_features + 1):
        area = np.sum(labeled_array == i)  # Count pixels per label
        centroid = center_of_mass(labeled_array == i)
        areas.append({'Object Label': i, 'Area (pixels)': area})
        centroids.append((centroid, i))  # Store centroid with its label

    # Create DataFrame and save CSV for the image
    df = pd.DataFrame(areas)
    df['Centroid Y'], df['Centroid X'] = zip(*[centroid[0] for centroid in centroids])  # Split centroids into their components
    image_csv_path = f'{Path(output_dir).absolute()}/{image_name}_grain_areas_and_centroids.csv'
    df.to_csv(image_csv_path, index=False)

    # Draw centroids and labels on the image
    image_color = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # White color for text
    thickness = 1
    for centroid, grain_label in centroids:
        y, x = int(centroid[0]), int(centroid[1])
        cv2.circle(image_color, (x, y), 5, (255, 0, 0), -1)  # Red color
        cv2.putText(image_color, str(grain_label), (x + 6, y), font, font_scale, font_color, thickness)

    # Save the modified image
    modified_image_path = f'{Path(output_dir).absolute()}/{image_name}_centroids.png'
    Image.fromarray(image_color).save(modified_image_path)

    return df

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
    return summary_df

def main(input_dir, output_dir):
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the input directory
    for file in os.listdir(input_dir):
        if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            image_path = os.path.join(input_dir, file)
            print(f'Processing {file}...')
            
            # Calculate grain areas and centroids for each image
            calculate_areas_and_centroids(image_path, output_dir)

    # Once all images are processed, compute summary statistics
    summary_csv_path = os.path.join(output_dir, 'grain_summary.csv')
    compute_grain_stats_and_save_summary(output_dir, summary_csv_path)

    print(f"Summary CSV saved to {summary_csv_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python grain_finder.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    main(input_folder, output_folder)
