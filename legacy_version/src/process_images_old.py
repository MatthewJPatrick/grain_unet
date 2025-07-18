#---------------------------------------------------------------------------------------------------------------#
#
# This program executes the full image processing workflow from the manuscript
# M. Patrick, J. K. Eckstein, J. R. Lopez, S. Toderas, S. Levine, J. M. Rickman, K. Barmak. Automated
# "Grain Boundary Detection for Bright-Field Transmission Electron Microscopy Images via U-Net",
# Submitted to Microscopy and Microanalysis (2023).
#
# The parameters required to run are:
#
#   data_dir             : the parent directory containing the data of interest
#
#   pattern_raw          : the UNIX style pattern for the subdirectories containing the raw *.tif images
#
#   pattern_predict      : the UNIX style pattern for the expected location of the grayscale predictions
#                          this will usually be the same directory as pattern_raw / predict_{resolution}
#
#   pattern_f            : the expected format of the name of the grayscale prediction files
#
#   res                  : this parameter is a list of resoltuions which you would like to perform the U-Net
#                          predictions. if left as an empty list, the program will skip to post-processing
#
#   variation_parameters : this is a dictionary of lists. the program will perform the post processing on the
#                          grayscale images which are found in the folders with the pattern pattern_predict
#                          and output them into a directory called {pattern_predict} / post_processed. It will
#                          perform the post processing on all combinations of the parameters given in the lists
#
#---------------------------------------------------------------------------------------------------------------#

# Modify the parameters below

data_dir = 'Data/test/fovs15/'
pattern_raw = f'fov1/raw/*.tif'
pattern_predict = f'fov1/predict_1012'
pattern_f = '*[0-9].png'
res =[]

variation_parameters = {

    'lib_thresh': [212],
    'con_thresh_diff' : [50],
    'prune_sizes' : [30],
    'dils' : [2],
    'min_areas':[60],

}

#---------------------------------------------------------------------------------------------------------------#

from src.high_throughput_multicore import high_throughput_multicore
high_throughput_multicore(data_dir, pattern_raw, pattern_predict, pattern_f, variation_parameters, ntest_res = res)

