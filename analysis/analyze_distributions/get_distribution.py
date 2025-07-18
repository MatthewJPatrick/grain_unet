''' sandbox for rosnel'''
import json
import os
import configparser
import pandas as pd 
from pathlib import Path
import math 
import matplotlib.pyplot as plt 
# from pprint import pprint

config = configparser.ConfigParser()
config.read('MyUnetConfig.ini')

PREDICT_DATA_DIR = config['INFERENCE_PARAMS']['OVERLAY_DATA_DIR']
TARGET_RESOLUTION = int(config['INFERENCE_PARAMS']['OUTPUT_RESOLUTION'])


json_file = 'scale_factors_mod.json'
scale_factors_dict = any

with open(json_file) as json_data:
    scale_factors_dict = json.load(json_data)
    #print(data['1hr2751_1']['sf'])


#global diameter metrics, to be saved in git directory 
diameter_table_global = []


def main(model_path = '', sf= ''):

    for filename in os.listdir(model_path): #filename are model pth filenames

        filename_str = str(filename)

        if not filename_str.endswith('.pth'):
             continue #skip this file 

        file_path = os.path.join(model_path, filename)

        if os.path.isfile(file_path): #make sure we have a file
            #now that we have a pth file, look for its csv file
            multipleModelGrainSizeCompute(modelFileName=filename_str,pattern=f'*_validation/{filename_str}_predict_{TARGET_RESOLUTION}/*.csv')

    
    #plt.hist(diameter_table_global, bins=50, alpha=0.7, color='purple', edgecolor='black', density=True)
    #plt.title(f'Grain Size Distribution for validation set w/ 290 epochs') 
    #plt.xlabel('log(Diameter, nm)')
    #plt.ylabel('Probability Density')
    #plt.savefig(f'validation_sizeDistribution.png',dpi=300,bbox_inches='tight')

    




def multipleModelGrainSizeCompute(pattern:str='',modelFileName:str='',folder=PREDICT_DATA_DIR):
    model_path_name = Path(modelFileName).name
    folder = Path(folder)
    paths = folder.glob(pattern)

    for path in paths: #path is a .csv file!
        #find dict entry from this data's sample (sample name in dict needs to match sample name in csv)
        if(path.name.__contains__('testing_saving_65')):
            df_tmp = pd.read_csv(path.absolute())

            print(f'Computing stats for: {path.name}')
            for key, val in scale_factors_dict.items():
                if path.name.__contains__(key):
                #print(key)

                #calculate resolution scaling factor
                    res_sf_new = float(TARGET_RESOLUTION / float(scale_factors_dict[key]["original_res"]))
                    new_tot_sf = float(scale_factors_dict[key]['sf']) * res_sf_new

                    #print(new_tot_sf)
                    #need to pull csv
                    

                    pixel_area_column = df_tmp['Area (pixels)']
                    grain_area_column = pixel_area_column * new_tot_sf

                    df_tmp['Area(nm2)'] = grain_area_column

                    df_tmp.to_csv(path.absolute(),index=False)

                    #now calculate diameters to make histogram 
                    diameters_tmp = []
                    for area in grain_area_column:
                        diameter_entry = 2*(math.sqrt((area)/(math.pi))) # using area of a circle as an approximation
                        if(diameter_entry > diame): #adjust for mean size of grains
                            diameters_tmp.append(diameter_entry)
                            diameter_table_global.append(math.log(diameter_entry)) #log the log of diameters
                    plt.hist(diameters_tmp, bins=100, alpha=0.7, color='purple',edgecolor='black')
                    plt.title(f'Grain Size Distribution for {modelFileName}')
                    plt.xlabel('Diameter')
                    plt.ylabel('Frequency')
                    plt.savefig(f'{path.absolute()}_sizeDistribution.png',dpi=300,bbox_inches='tight')
                
               


            



if __name__ == "__main__":
    ModelPTH_directory = 'D:/Grain_Boundary_data_2023/epochint-checkpoint-models/epochint-checkpoint-models'
    main(model_path=ModelPTH_directory)