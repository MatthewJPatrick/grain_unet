
from src.high_throughput_multicore import high_throughput_multicore
import sys
from src.configuration_helper import read_cfg_img_proc


def __main__():
    print("\n#-------------------------------Begin-------------------------------#\n")
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print("Filename:", filename)
    else:
        filename = 'config.ini'
        print("Used Default: config.ini")

    #read the configuration file for all the requisite inputs
    data_dir, pattern_fov, pattern_raw, res, unet_predict_tf, post_process_tf, variation_parameters, pattern_predict, network_file = read_cfg_img_proc(filename)

    print(f" Reading files in {data_dir}/{pattern_fov}/{pattern_raw}")

    #This runs U-Net and/or post-processing with the configuration in the config.ini (or other specified name)
    high_throughput_multicore(test_image_dir = data_dir, pattern_fov = pattern_fov, pattern_raw=pattern_raw,
                              ntest_res = res, unet_predict_tf = unet_predict_tf, post_process_tf = post_process_tf, 
                              variations = variation_parameters, pattern_predict = pattern_predict, network_file =  network_file)
    
    print("\n#--------------------------------End--------------------------------#\n")

if __name__ == "__main__":
    __main__()