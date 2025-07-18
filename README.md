# Pytorch Implementation of U-Net for Automated Grain Boundary Detection 

```
   ____                             _       ____                       
  | __ )  __ _ _ __ _ __ ___   __ _| | __  / ___|_ __ ___  _   _ _ __  
  |  _ \ / _` | '__| '_ ` _ \ / _` | |/ / | |  _| '__/ _ \| | | | '_ \ 
  | |_) | (_| | |  | | | | | | (_| |   <  | |_| | | | (_) | |_| | |_) |
  |____/ \__,_|_|  |_| |_| |_|\__,_|_|\_\  \____|_|  \___/ \__,_| .__/ 
                                                               |_| 
   U-Net Model for Automated Grain Boundary Detection Software Suite   
```

This folder is a migration of the codebase used in the [original paper](https://academic.oup.com/mam/article/29/6/1968/7422794?login=false) to construct a neural network capable of segmenting BF TEM images of polycrystalline materials from Tensorflow to Pytorch. There are some QOL features sprinkled in for the sake automating analysis of performance of said models.  

The intent of this makeover is to make the workflow for running  experiments on variegated data or model parameters a simpler process. A typical workflow for working with this model will have you execute directory-level scripts on your terminal of choice, modifying parameters in a configuration file, and analyzing the model predictions/metrics. 

All software is intended to work on all platforms (Windows/Mac/Linux) with minimal hardware set-up if using discrete NVIDIA GPUs and/or integrated M-series GPUs. 

![alt text](exampleWorkflowFigure.png)

># Quickstart Guide (TL;DR): 
>A more detailed breakdown of the installation and execution process can be found in the latter sections of this document. If you already have the background on this project, or just want to make sure the code is working on its own, you can use the following steps to check all is working as it should.  
>This latest release that includes all the example files can be downloaded [here](https://github.com/MatthewJPatrick/grain_unet_working/releases)
>The scope of this project in particular is held within the `Unet_Pytorch` folder. 
>Example configuration files, model weights, dataset, and singular test images are provided. 
>## Installation
>Run the following command to download all dependencies
>``` pip install -r requirements.txt```
>## Single Inference 
>To verify that all components are functioning run the following command in terminal within this directory
>```bash
>python run_inference.py testimage_299.png testpred_299.png
>```
>This will pass the `testimage_299.png` file into the pre-trained model and output its prediction in `testpred_299.png`. 
>## Automated Inferencing
> The `InferenceRunner.py` script as the option to iterate through a dataset and run inference on a large batch of samples. To activate this functionality, ensure the the `PP_ACTIVE` parameters in the configuraiton file is set to `False`, then run the following command in your project directory:
>```
>python run_inference.py
>```
>Within the provided test_data files, the predictions will have their own folders within the folder containing the raw inputs. 
>## Post-Processing
>To run post-processing on the example files run the following command in your terminal after changing the `PP_ACTIVATE` parameter in the configuration file to `True`. This will circumvent automated inferencing and only run post-processing function on images in the test data directory. 
>```bash
>python run_inference.py
>```
>This will store post-processed binarized segmentations in the same directory as the raw input images within the example test data directory.
>## Training
> If you haven't checked out the configuration file before starting training, be sure to change the `MODEL_PARAMS` file to a different filename to avoid overwriting the example pre-trained model! 
> Once that's all set, to test out training, using the provided data folder and simply run the following command 
>```bash
>python run_training.py 
>```
>Training should commence with a graphical interface indicating the duration of the training and validation steps, along with a log of the loss function for the model over training steps.

## Installation 
Run the following command in your terminal environment whilst in the Unet_Pytorch folder

```bash
pip install -r requirements.txt
```

If you run into package/run-time errors try updating your PIP version using the following commands (for Mac omit the .exe extension): 
```bash 
python.exe -m pip install --upgrade pip
```

This will download all requirements for getting the code in the pytorch directory to run on your machine. 

> ***NOTE ON FILE STRUCTURES***: Try to adhere to the file structures presented in train_nouveaux_256. This will allow the code presented to work <i>mostly</i> seamlessly. If you use your own data, warnings like this will be made on necessary functional modifications.
### Configuration File 

Your configuration file is the collection of parameters and file path unique to your machine. The file will allow you change where your images and labels are pulled from, turn/off augmentations, modify model hyper-parameters, etc. 

This project uses a simple .ini format for its configuration file, allowing for a human-readable and machine-accessible key-pair dictionary formatting. You can create your own, or download the sample configuration file from this project's latest release. 

An example configuration file is shown here

```ini
#Your configuration file
#Make sure to name your file 'MyUnetConfig.ini'
;You can also do comments like this 
[PATH]
IMAGE_PATH = C:/Your/Image/Path/With/Out/Spaces/Here
LABEL_PATH = D:/path
MODEL_PARAMS = 

#These parameters are ~1:1 from the original paper
[TRAINING_PARAMS]
BATCH_SIZE = 5 
EPOCHS = 80
NUM_WORKERS = 0 #determines the amount of cores assigned to training
LEARNING_RATE = 0.0001
SAVING_RATE = 50 #stores checkpoints of your model 

#These are recommended parameters
[DATA_PARAMS]
AUGMENT_ACTIVE = True 
TRAINING_SPLIT = 0.9

[INFERENCE_PARAMS]
INFERENCE = True
N_TEST_DATA_DIR = A:/path/for/automated/inferencing/input/images
N_TEST_PATTERN = *.png
OUTPUT_RESOLUTION = 512 #images should be square 
OVERLAY_DATA_DIR =  D:/put/images/that/already/have/been/inferenced/for/post_processing

#These are parameters for adjusting the output segmentation post-processing
[POST_PROCESS]
POST_PROCESS = False
COMPILATION = min
N_DILATOINS = 3
LIBERAL_THRESHOLD = 161
CONSERVATIVE_THRESHOLD = 212 
INVERT_DOUBLE_THRESHOLD = True
MIN_GRAIN_AREA = 100
PRUNE_SIZE = 30
```

Modifying these parameters is as simple as changing the values associated with a particular value on the right hand side of the assignment. Within the code, if you wish to modify or pull values as needed you note the structure of parameters within the file. 

The hierarchy works like a folder where the values in [brackets] are the category of parameters, and the value underneath the bracketed value is a specific data point you might be looking for. So, if you want to for example, print out whether or not you are currently allowing augmentation to take place, you could add this line in the code:
```python
config = configparser.ConfigParser('MyUnetConfig.ini')

test_augment_check = bool(config['DATA_PARAMS']['AUGMENT_ACTIVE'])

print(test_augment)
```

> ***NOTE ON CONFIGURATION FILE***: Key names and the output from the configparser object are all raw strings, so be sure to cast as your needed data type when using the configuration file as shown in the example! 

## Training U-Net 
The model from the original paper was trained on this [dataset](https://redivis.com/datasets/ezwc-6yhc9b71p?v=1.0) of Aluminum thin films BF TEM images. It is possible to train the model to perform segmentations on other data as long as the image and label file paths are properly provided in the configuration file. 


### Set-Up

In your configuration file ensure the following parameters are set: 
- IMAGE_PATH
- LABEL_PATH
- MODEL_PARAMS
- AUGMENT_ACTIVE
- TRAINING SPLIT
- EPOCHS 
- BATCH_SIZE
- LEARNING_RATE
- NUM_WORKERS

IMAGE_PATH should be the input images that model will be trained on. 

LABEL_PATH should be the labels that are the expected output the model should create when presented with the paired training image

MODEL_PARAMS is the filename that will be used to store the models during training. This is a heavy binary file, and depending on your set number of checkpoints will take a considerable amount. 
> ***NOTE ON MODEL PARAMS*** : [PyTorch recommends](https://pytorch.org/tutorials/beginner/saving_loading_models.html) .pth or .pt as the file extension for these files, however, there isn't a significant difference in using others like .hdf5, so use whatever works for your use case. 

AUGMENT_ACTIVE is a boolean (true/false) on whether or not the data set should have augmentations applied randomly. 

>***NOTE ON AUGMENTATIONS***: For our group's images, only flips and rotations are applied in attempt to maintain fidelity of the physical processes being studied. You might find that you need other image transformation functions, all of which can be added through the Pytorch image [transform compositions](https://pytorch.org/vision/stable/transforms.html) within the `train.py` and `dataloader.py` function.

TRAINING_SPLIT is a float value (default 0.9) which will determine how much of the total dataset to use for training x 100. So, with the default value 90% of the dataset will be used for training, and the resulting 10% will be used for validating performance of the model. 

EPOCHS is how many training and validation steps will be executed. 

NUM_WORKERS is how many cores of your processor you'd want to dedicate towards training the model. 

BATCH_SIZE is how many 'compartments' each training step will use at each epoch. This is useful for information dense training data where loading thousands of images for each training step isn't practical from a computational perspective. 

Once all these parameters are set, the script is ready to run. 

# Execute Training
Assuming all parameters have been set, running the following script will initiate the training sequence

```bash
python /unetModel/train.py 
```

Your model weights and biases will be saved in the folder you've indicated for `MODEL_PARAMS` from your configuration file along with each training step depending on the resolution desired. The hyper-parameters within the configuration file are also what affect training performance overall in this script. 

If all goes well, you should see the following dialog in your terminal:
![alt text](exampleTrainingUI.png)

The script will verify the computational unit used for training. 

>***NOTE ON COMPUTE PLATFORMS:*** Typically in ML workflows, a processor suited for parallelization is prefered. Regardless of the platform, the script will choose the most compatible and high performance processors. These are the supported platforms: 
> - Discrete NVIDIA GPU with CUDA cores
> - Apple Silicon M-series processors 
> - Multi-core generic CPU 
> 
>Unfortunately, there is no official documented support for AMD graphical processors for this project. Pytorch does offer support for [ROCm on AMD processors](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html), however this is untested in our project.

The script will also print out the amount of images found in your dataset as a quick way to check if the data processing step is functional. After all these checks, a dialog will note the start of training and load a progress bar for each epoch's training and validation step. Once exhausted, and 'End Training' dialog will be prompted. 

## Training Metrics 
As an added feature, the training script will log the loss of each training and validation step over all epochs. This data is put into two csv files and a preliminary plot is also generated in the relative path of your project directory. This gives a cursory view of model performance, while also allowing you to export the data into a software of your choice to analyze as you wish. 

This is an example plot generated by the script after training on a dataset for 300 epochs:
![alt text](lossValExample.png) ![alt text](lossTrainExample.png)

## How to Run a Single Inference

Assuming you have a trained model (stored .pth file with model weights and biases), you can execute a single inference!

The single inference function is encapsulated within `InferenceRunner.py` and is a good way to make sure that your model when given an input generates a desired output for automation onto a large dataset (which is discussed next). 

To exectue a single inference, run the following command in a terminal within your project directory:

```bash
python InferenceRunner.py inputImageName.png outputImageName.png
```
The first argument for the script will read the path to your input image. The image will then be passed through the model to evaluate and construct its predicted output image based on your second argument's path. 

The simplest way to test functionality is with an image that the model is already trained on to make sure that training was able to capture and learn of it at least. Below is an example image from the training set:

![alt text](testimage_299.png)

After running it through a model trained from the original paper's dataset, the expected output should looks close to this:

![alt text](expectedOutput_299.png)


## Automated Data-specific Inferencing 
The `InferenceRunner.py` script has the functionality to run through an entire dataset of raw images (.tif and .png supported) and generate predictions in a separate folder within the sample's directory. Below is a visualization of how the `n_test()` function within the code traverses our own dataset to create predicted segmentations. 

![alt text](exampleDataStructure.png)

## Set up n_test()

There are some preliminary measures that need to be taken to ensure the automated mass inference will work smoothly. 
Firstly, make sure that the `N_TEST_DATA_DIR` is the proper path where your raw images are within your configuration file. This can be the top most folder that encompasses sub-folders containing the raw images. 

The second thing to do is to go to line 171 in `InferenceRunner.py` where the `n_test()` function is invoked. It should be within the main loop and look like this:
```python
n_test(folder = TEST_DATA_DIR, pattern = 'folder1/folder2/*.png')
```
For your specific data-organization scheme, you might have your data broken up into folders. So long as this data is within a predictable pattern of folder structure you can give the pattern of folder paths into the pattern parameter. This will give the script a pattern to retrace for every directory it falls into to compute its predictions and move onto the next folder with the next data it needs. 

The folder parameter should be untouched as it is pulled from the configuration file, the pattern argument however should be hard-coded as a parameter. The reason this is done is because this is usually a parameter so infrequently changed that it need only be configured once depending on the use-case. However, the configuration file can always be modified to include this parameters on user-specific basis. 

> ***NOTE ON PATTERN***: The final portion of the pattern should be * and the subsequent file extension that your input data requires. In our case, the data might be all in .png formats so we would want the final file tag to *.png to find all png files within the directory pattern provided. 

The resulting predictions will be stored in a folder within the directory where the original input images are stored with a folder name of `predict_resolution` where "resolution" is based off the `OUTPUT_RESOLUTION` parameter specified in the configuration file. 

# Post-Processing

The goal of post-processing is to clean up the model predictions to create a binarized image that is clearly identifiable for data analysis workflows that need to clearly discriminate between grains and boundaries. 

The post-processing functions can have their parameters modified within the configuration file depending on the aggressiveness desired for thresholding and other filters. 

To execute this function, modify the `PP_ACTIVATE` parameter in your configuration file to `True`. This will circumvent the automated inferencing and instead apply post-processing to all images within your specific data directory.

To execute, run the following command in your project directory after modifying the `PP_ACTIVATE` parameter:
```
python InferenceRunner.py
```


Below is an example of the type of segmentation one should expect after executing the post-processing function.
![alt text](unet_skel_al-324-200C_15min_aligned_fov1_+1_dils_003_cons_183_lib_213_prune_070_minarea_060_new.png) 

# Contributors

Matthew J. Patrick 
Rosnel Leyva-Cortes
Lauren Grae

# Misc.
On the team we liked to have auditory indicators that are model was done training. The sound related parameters in the configuration give you the option to play .mp3 files as the model starts training, and also when it ends. Just a fun little easter egg for those interested :D

```
                        .----------------.
                       /__________________\ 
                       || |:Matthew:||o(_)|
                       || |;Rosnel;| |o(_)|
                       || |_Lauren_| | __ |
                       ||/__________\|[__]| Cook n' look
                       "------------------"
```
