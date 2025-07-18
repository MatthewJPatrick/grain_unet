#first, make save dirs

import os
import numpy as np
from pathlib import Path, PurePath
from skimage import io
import subprocess
from multiprocessing import Pool
from functools import partial
from time import sleep
from tqdm import tqdm
#--------------------------------------------------------------------------------------------------------------#

#change the four directory paths below to whatever suits your folder formatting
main_dir = Path('/Volumes/MyPassport/Matthew/AXON_05-04/Sessions/2025-05-04-131139-Fusion/2025-05-04-131139-Fusion/DriftCorrected')

FOV_pattern = False
raw_pattern = '*.png'
#set the level of denoising below
noise=30

#--------------------------------------------------------------------------------------------------------------#
def __main__():
    #can comment out either of these functions if not being used for now
    if FOV_pattern:
        create_subfolder_in_each_folder(main_dir,'cropped-noisy')
        create_subfolder_in_each_folder(main_dir,'denoised')
        for frame in main_dir.iterdir():
            print(frame, type(frame))
            test_dir=Path(os.path.join(frame,raw_pattern))
            out_dir=Path(os.path.join(frame,'cropped-noisy'))
            noisy_dir=out_dir
            clean_dir=Path(os.path.join(frame,'denoised'))
            convert_files_multicore(test_dir, out_dir)
            denoise_multicore(noisy_dir, clean_dir, noise)
            
    else:
        test_dir=Path(main_dir)
        out_dir=Path(os.path.join(main_dir,'cropped-noisy'))
        if not out_dir.is_dir():
            os.makedirs(out_dir)
        noisy_dir=out_dir
        clean_dir=Path(os.path.join(main_dir,'denoised'))
        if not out_dir.is_dir():
            os.makedirs(out_dir)
        #convert_files_multicore(test_dir, out_dir)
        denoise_multicore(test_dir, clean_dir, noise)


#reading in images and converting to png + cropping
def create_subfolder_in_each_folder(root_dir, new_subfolder_name):
    #root_dir = Path(root_dir)
    for folder in root_dir.iterdir():
        subfolder_path=Path(os.path.join(root_dir,folder,new_subfolder_name))
        if not subfolder_path.is_dir():
            os.makedirs(subfolder_path)
            
def convert_files_single_core(test_dir, out_dir, pattern = "*.png"):

    n_files = len(list(test_dir.glob(pattern)))

    ii = 0
    for image in test_dir.glob(pattern):
        print(image)
        convert_file(image, out_dir)
        ii+=1
    
#using subprocess to run the c++ code through command line. If a faster/more elegant/more accessible solution is desired, recommend a boost python implementation
def denoise_folder_single_core(noisy_dir, clean_dir, noise):

    if not Path(clean_dir).is_dir():
        os.mkdir(clean_dir)
    noisy_dir = list(noisy_dir.glob('*.png'))
    n_files = len(noisy_dir)
 
    ii = 0
    for image in tqdm(noisy_dir, desc='Denoising', total=n_files):
        denoise(image, clean_dir, noise)
        ii+=1
        

def convert_files_multicore(test_dir,out_dir,pattern="*.png"):

    images = []
    for image in test_dir.glob(pattern):
        images.append(f'{image}')
    with Pool() as pool:
        funct = partial(convert_file, out_dir=out_dir)
        rs = pool.map_async(funct, images)
        total = rs._number_left
        print(f'{total} pools generated')
        pool.close()

        while (not rs.ready()):
            remaining = rs._number_left
            done = total - remaining
            #print_progress_bar(done, total)
            #sleep(0.25)

def denoise_multicore(noisy_dir, clean_dir, noise):

    if not Path(clean_dir).is_dir():
        os.mkdir(clean_dir)
    #get list of images in the folder
    images = []
    for image in noisy_dir.glob('*.png'):
        if (clean_dir / f'raw-denoised_{PurePath(image).name}').is_file():
            print(f'File {image} already exists, skipping')
            continue
        else:
            images.append(image)

   

    #allocate resources to run this on multiple cores
    with Pool() as pool:
        funct = partial(denoise, clean_dir=clean_dir, noise=noise)

        #count the number of closed proecessed for progress reporting
        rs = pool.map_async(funct, images)
        total = rs._number_left
        print(f'{total} pools generated')
        pool.close()

        while (not rs.ready()):
            remaining = rs._number_left
            done = total - remaining
            print_progress_bar(done, total)
            sleep(0.25)
        results = rs.get()

def denoise(image, clean_dir, noise):
        subprocess.run(['sh', '-c', './nl-bayes_20130617/NL_Bayes'], capture_output=False)
        save_dir =clean_dir / f'raw-denoised_{PurePath(image).name}'
        program = './nl-bayes_20130617/NL_Bayes'
        arg = f' {image} {noise} 0 noisy.png {save_dir} basic.png difference.png bias.png biasbasic.png diffbias.png 1 0 0'
        subprocess.run(['sh', '-c', program + arg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('Finished image' + str(PurePath(image).name))

def convert_file(image, out_dir, base_type = '.tiff', new_type = '.png'):
        img = io.imread(image)

        #cropping function (can comment out below if not useful)
        img = img[124:1924, 124:1924]

        image_name = f'{PurePath(image).name}'.split(".")[0]
        original_type = f'{PurePath(image).name}'.split(".")[1]
        save_path = out_dir/(f'{image_name}-{original_type}-{new_type}')
        io.imsave(save_path, (img))
def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
if __name__ == "__main__":
    __main__()