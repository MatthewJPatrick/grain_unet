# This section reads the configuration file specified as a command line argument, or the default 
# config.ini file in the same folder as the script.

from pathlib import Path
import configparser
from ast import literal_eval

def read_cfg_img_proc(configuration_file = 'config.ini'):
    # Load the configuration file
    if Path(configuration_file).is_file():
        config = configparser.ConfigParser()
        config.read(configuration_file)
    else:
        print(f'No file named {configuration_file} \nAborting program')
        return

    # Read the values from the configuration file
    data_dir = config['DEFAULT']['data_dir']
    pattern_fov = config['DEFAULT']['pattern_fov']
    pattern_raw = config['DEFAULT']['pattern_raw']
    res = eval(config['DEFAULT']['res'])
    unet_predict_tf = bool(config['DEFAULT']['UNet_predict'])
    post_process_tf = bool(config['DEFAULT']['Post_Process'])
    variation_parameters = dict(config['variation_parameters'])
    pattern_predict = config['DEFAULT']['pattern_predict']
    network_file = Path(config['DEFAULT']['network_file'])

    # Convert variation parameter values to appropriate types i.e. integers
    for key, value in variation_parameters.items():
        try:
            variation_parameters[key] = literal_eval(value)
            if not isinstance(variation_parameters[key], list):
                variation_parameters[key] = [variation_parameters[key]]
            variation_parameters[key] = [int(item) for item in variation_parameters[key]]
        except (ValueError, SyntaxError):
            # Unable to convert, leave the value as is
            pass
    return(data_dir, pattern_fov, pattern_raw, res, unet_predict_tf, post_process_tf, variation_parameters, pattern_predict, network_file)
    # Call the function with the loaded values

def read_cfg_train(configuration_file = 'config.ini'):
    print('This function is under construction!')
