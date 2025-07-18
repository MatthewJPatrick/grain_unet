
import configparser
import torch
import torch.nn as nn
global INFERENCE, IMAGE_PATH, OUTPUT_PATH, DEVICE_COMPUTE_PLATFORM, LABEL_PATH, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS
global MODEL_PARAMS, MODEL_NAME, TEST_DATA_DIR, PREFIX, PP_ACTIVATE
global PREDICT_DATA_DIR, TARGET_RESOLUTION, FINAL_OUTPUT_DIRECTORY, args_pp
global TRAINING_SPLIT, AUGMENT_ACTIVE, LEARNING_RATE
global CUSTOM_LOSS_FUNCTION, UNET_MODEL, LOSS_FN, OPTIMIZER, SAVING_RATE

def init(config_file_name:str = "MyUnetConfig.ini")->None :

    config = configparser.ConfigParser()
    config.read(config_file_name)
    
    #Configuration file variables    
    global INFERENCE, IMAGE_PATH, OUTPUT_PATH, DEVICE_COMPUTE_PLATFORM, LABEL_PATH, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS
    global MODEL_PARAMS, MODEL_NAME, TEST_DATA_DIR, PREFIX, PP_ACTIVATE
    global PREDICT_DATA_DIR, TARGET_RESOLUTION, FINAL_OUTPUT_DIRECTORY, args_pp
    global TRAINING_SPLIT, AUGMENT_ACTIVE, LEARNING_RATE
    global N_TEST_PATTERN
    global MODEL_PARAMS
    #determine compute platform
    global DEVICE_COMPUTE_PLATFORM

    DEVICE_COMPUTE_PLATFORM = get_compute_platform()
    AUGMENT_ACTIVE = int(config['DATA_PARAMS']['AUGMENT_ACTIVE'])



    TRAINING_SPLIT = config['DATA_PARAMS']['TRAINING_SPLIT']
    IMAGE_PATH = config['PATH']['IMAGE_PATH']
    LABEL_PATH = config['PATH']['LABEL_PATH']
    BATCH_SIZE=int(config['TRAINING_PARAMS']['BATCH_SIZE'])
    NUM_EPOCHS=int(config['TRAINING_PARAMS']['EPOCHS'])
    NUM_WORKERS=int(config['TRAINING_PARAMS']['NUM_WORKERS'])  
    LEARNING_RATE = float(config['TRAINING_PARAMS']['LEARNING_RATE'])

    MODEL_PARAMS = config['PATH']['MODEL_PARAMS']
    TARGET_RESOLUTION = int(config['INFERENCE_PARAMS']['OUTPUT_RESOLUTION'])
    INFERENCE = config['INFERENCE_PARAMS']['INFERENCE']
    MODEL_PARAMS = config['PATH']['MODEL_PARAMS']
    MODEL_NAME = MODEL_PARAMS.split('/')[-1].split('.')[0]
    TEST_DATA_DIR = config['INFERENCE_PARAMS']['N_TEST_DATA_DIR']
    N_TEST_PATTERN = config['INFERENCE_PARAMS']['N_TEST_PATTERN']
    PREDICT_DATA_DIR = config['INFERENCE_PARAMS']['OVERLAY_DATA_DIR']
    TARGET_RESOLUTION = int(config['INFERENCE_PARAMS']['OUTPUT_RESOLUTION'])
    FINAL_OUTPUT_DIRECTORY = config['INFERENCE_PARAMS']['FINAL_OUTPUT_DIR']
    PP_ACTIVATE = int(config['POST_PROCESS']['POST_PROCESS'])

    PREFIX = MODEL_PARAMS.split('/')[-1].split('.')[0]

    args_pp = { # for post processing 
                'compilation': config['POST_PROCESS']['COMPILATION'],
                'n_dilations': int(config['POST_PROCESS']['N_DILATIONS']),
                'liberal_thresh': int(config['POST_PROCESS']['LIBERAL_THRESHOLD']),
                'conservative_thresh': int(config['POST_PROCESS']['CONSERVATIVE_THRESHOLD']),
                'invert_double_thresh': bool(config['POST_PROCESS']['INVERT_DOUBLE_THRESHOLD']),
                'min_grain_area': int(config['POST_PROCESS']['MIN_GRAIN_AREA']),
                'prune_size': int(config['POST_PROCESS']['PRUNE_SIZE'])
        }


def init_training(config_file_name:str = "MyUnetConfig.ini")->None :
    
    global CUSTOM_LOSS_FUNCTION, UNET_MODEL, LOSS_FN, DEVICE_COMPUTE_PLATFORM, OPTIMIZER, SAVING_RATE
    init()

    config = configparser.ConfigParser()
    config.read(config_file_name)
    pretrained_weights = config['TRAINING_PARAMS']['PRETRAINED_WEIGHTS']
    CUSTOM_LOSS_FUNCTION = config['TRAINING_PARAMS']['CUSTOM_LOSS_FUNCTION']
    SAVING_RATE = int(config['TRAINING_PARAMS']['SAVING_RATE'])

    try:
        num_class = int(config['TRAINING_PARAMS']['NUM_CLASSES'])
    except:
        num_class = 1

    if CUSTOM_LOSS_FUNCTION:
        LOSS_FN = CUSTOM_LOSS_FUNCTION
    else:
        LOSS_FN = nn.BCEWithLogitsLoss()

    if pretrained_weights:
        UNET_MODEL = pretrained_weights
    else:
        UNET_MODEL = None

    OPTIMIZER = torch.optim.Adam
    if CUSTOM_LOSS_FUNCTION == "BCEWithLogitsLoss":
        LOSS_FN = nn.BCEWithLogitsLoss()
    else:
        Warning("No loss function specified. Using default BCEWithLogitsLoss")
        LOSS_FN = nn.BCEWithLogitsLoss()

def get_compute_platform():

    if torch.cuda.is_available():
        compute_platform = 'cuda'
    elif torch.backends.mps.is_available():
        compute_platform = 'mps'
    else:
        compute_platform = 'cpu'
    return compute_platform

init()
init_training() 
