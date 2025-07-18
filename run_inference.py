'''
Inference Runner
Reads config for paths containing images and outputs segmentations without post-processing
'''
__author__ = "Matthew Patrick, Lauren Grae, Rosnel Leyva-Cortes" 

from utility.settings import *


import sys
import utility.user_interface as user_interface
from   unet_model.inference.post_processing.post_process import bulk_compile_and_pp, in_situ_post_process
from   unet_model.inference.infer  import single_inference, multi_folder_inference
from unet_model.inference.infer import in_situ_inference
from utility.settings import MODEL_NAME, TARGET_RESOLUTION, INFERENCE, PP_ACTIVATE, TEST_DATA_DIR, PREDICT_DATA_DIR, MODEL_PARAMS


if __name__ == "__main__":
    import argparse
    import utility.user_interface as user_interface
    #pattern = "fov*/raw/*.tif" #Original for new "test data", do not lose
    user_interface.logoPrint()

    if len(sys.argv) == 2:
        parser = argparse.ArgumentParser(description="U-Net Inference Runner")
        parser.add_argument("mode", type=str, help="mode")
        args = parser.parse_args()
        if args.mode == "in_situ":
        
            if PP_ACTIVATE:
                in_situ_post_process(in_folder = TEST_DATA_DIR, out_folder = TEST_DATA_DIR + "/post_process", integration = 3)
           
        elif args.mode == "post_process":
            print(f"fov*/predict_{MODEL_NAME}_{TARGET_RESOLUTION}/")
            bulk_compile_and_pp(folder=PREDICT_DATA_DIR, pattern = f'fov*/predict_{MODEL_NAME}_{TARGET_RESOLUTION}/', post_process_option=True)
        else:
            print("Invalid mode. Use 'in situ' or 'post process'.")
            sys.exit(1)

    if len(sys.argv) == 3:
        parser = argparse.ArgumentParser(description="U-Net Inference Runner")
        parser.add_argument("image_path", type=str, help="Path to the input image")
        parser.add_argument("output_path", type=str, help="Path to save the output image")
        args = parser.parse_args()
        single_inference(args.image_path, args.output_path, model = MODEL_PARAMS)

    elif len(sys.argv) == 1: #no args are passed, assuming n_test functionality use your own pattern
      
        if (INFERENCE):
            multi_folder_inference(folder = TEST_DATA_DIR, pattern = N_TEST_PATTERN) 

        if (PP_ACTIVATE):
            bulk_compile_and_pp(folder=PREDICT_DATA_DIR, pattern = f'fov*/predict_{MODEL_NAME}_{TARGET_RESOLUTION}/', post_process_option=True)
        else:
            pass