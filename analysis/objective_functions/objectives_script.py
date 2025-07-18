from validation_functions import get_objectives
from pathlib import Path
from centroids import import_centroids
from big_data import elapsed_time
import csv
from skimage import io

# Author, Date: Sylvia Whang (siw2111@barnard.edu), November 2023
'''This program uses python centroid analysis to calcualate objective functions and write them to a csv file -- connecting the pipeline'''

def calculate_objs(path_m, path_n):
# takes python centroid data from two images and returns objective functions between them 

    img_n = io.imread(path_n)
    n_centroids = import_centroids(img_n, path = path_n)
    comp_points = list()
    for centroid in n_centroids: 
        comp_points.append(centroid.coords)

    img_m = io.imread(path_m)
    m_centroids = import_centroids(img_m, path = path_m)
    ref_points = list()
    for centroid in m_centroids: 
        ref_points.append(centroid.coords)
    
    time = n_centroids[0].time
    objectives = get_objectives(ref_points, comp_points, cutoff_fraction=0.3)
    
    return objectives, time

def objs_to_csv(img_dir, csv_fname):
# This function looks at a folder of images and records the total number of triple junctions and total disconinutuies as a function of time. Data is saved to a csv file. 
    
    img_dir = Path(img_dir)
    img_paths = list(img_dir.glob('frames_*/post_processed/*new.png'))
    img_paths = sorted(img_paths, key = lambda x: x.parts[-1])

    with open(csv_fname, 'w', newline = '') as file:
        print(f'writing to... {file}')
        writer = csv.writer(file)
        writer.writerow(['n', 'time (HHMMSS.MS)', 'time elapsed (s)', 'centroids', 'orphan', 'disregistry'])
        writer.writerow([0, 0, 0, 0,0])
        n = 1
        
        while n < len(img_paths):
            print('______________________________________________')
            print(f'image {n}: {img_paths[n]}...')
            objectives, time = calculate_objs(img_paths[n-1], img_paths[n])
            time_elapsed = elapsed_time(time)
            writer.writerow([n, time, time_elapsed, objectives['n_pairs'], objectives['orphans'], objectives['disregistry']])
            print(f'time: {time_elapsed}, orphan fraction: {objectives["orphans"]}, disregistry: {objectives["disregistry"]}')
            n+=1

if __name__ == '__main__':
    img_dir = '/media/shared_data/platinum/PtAxonData_denoised-2021-12-07/sorted'
    csv_fname = '/media/shared_data/platinum/PtAxonData_denoised-2021-12-07/sorted/objs_global.csv'
    objs_to_csv(img_dir, csv_fname)