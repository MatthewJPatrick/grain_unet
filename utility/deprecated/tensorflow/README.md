# grain_unet

This program will process BF TEM images of Al, and post-process them with your desired parameters.
This program runs in an Anaconda Environment, and has only been tested on a Linux machine, and comees with no warranty.
Setup for the working environment can be performed by using the following command in the working directory:

  conda env create -f environment.yml python=3.10.9

The training sequences are not packaged, but are included in the source code.

config.ini contains the input parameters for the process_images.py program. 
Explanations for all input files are included in the comments of that file.

To run the program, activate the conda environment with the appropriate packages installed.
Modify config.ini to your desired parameters, and save it
then run the command

  python process_images.py <configuration_file>
  
where <configuration_file> is the name of your desired file.
If <configuration_file> is left blank, it will default to config.ini

The approach in this repository is paired with data at
  Nano Initiative Electron Microscopy Lab. (2023). Automated Grain Boundary Detection via U-Net: Images (Version 1.0) 
  [Data set]. Redivis. https://doi.org/10.57783/w6n3-5b02

This is part of the work published in Microscopy and Microanalysis:
  [1] Matthew J Patrick, James K Eckstein, Javier R Lopez, Silvia Toderas, Sarah A Asher, Sylvia I Whang, Stacey Levine, Jeffrey M Rickman, Katayun Barmak, Automated Grain Boundary Detection for Bright-Field Transmission Electron Microscopy Images via U-Net, Microscopy and Microanalysis, Volume 29, Issue 6, December 2023, Pages 1968–1979, https://doi.org/10.1093/micmic/ozad115
