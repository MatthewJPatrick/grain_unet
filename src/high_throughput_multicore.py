from src.ImgProcessFunct import post_process_folder, gen_parameter_list, n_test
from src.printProgressBar import print_progress_bar
from multiprocessing import Pool
from pathlib import Path
from functools import partial
from time import sleep, time
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(4)


def high_throughput_multicore(test_image_dir:str or Path = './Data',pattern_fov:str = '/fov*', ntest_res:list = [], unet_predict_tf:bool = False, 
                              post_process_tf:bool = True, variations:dict = None, pattern_predict = '*.png', 
                              network_file:Path = Path('./unet_grain_nouveaux_pretune.hdf5') ):

    test_image_dir = Path(test_image_dir)
    
    #input validation
    print(f'U-Net Prediction: {unet_predict_tf}')
    print(f'Post-Processing : {post_process_tf}')
    #Run U-Net predictions for given raw images if there is a specified resolution
    if unet_predict_tf:
        pattern_raw = pattern_fov + '/raw/*.tif'
        for res in ntest_res:
            n_test(res, res, test_image_dir, pattern_raw, network_file)
    
    if post_process_tf:
        
        #define post processing parameter to permute
        liberal_thresholds = variations['lib_thresh'] #[180,185,190,195,200,205,210,215,220,225,230]
        con_thresh_diffs = variations['con_thresh_diff'] #[10,30,40,50,60,70]
        prune_sizes = variations['prune_sizes'] #[50,100,150]#[50,75,100,125,150]
        dilations = variations['dils'] #[5,4,3,2,1]
        min_areas = variations['min_areas'] #[0,50,100,150,200,250,300]

        #generate parameter list
        params = gen_parameter_list(liberal_thresholds, con_thresh_diffs, prune_sizes, dilations, min_areas)

        #Display how many total sets of parameters there are
        total_attempts = len(params)
        print(f'I will generate {total_attempts} post-processed images for each requested FOV, but those with no contrast will not be saved.')

        # setup and run program with max available cores on the dormnet server. 
        # To lower the number of cores, use an argument for Pool(n), 
        # where n is your desired number of cores

        # this only speeds up processing if you are using many different combinations of 
        # post-processing parameters. Otherwise, it probably makes it a lot slower.
        # The section can be replaced by just calling post_process_folder. In the future
        # it will be wise to implement multicore post-processing for each individual image
        # in a given FoV, but for now it images in series and different post-process parameter
        # combinations in parallel. This is useful when checking many PPPs when optimizing them.

        for res in ntest_res:
            time_0 = time()
            pattern_predict_res = f'{pattern_fov}/predict_{res}'
            with Pool() as pool:
                
                funct = partial(post_process_folder,test_directory = test_image_dir, pattern_prediction_folder = pattern_predict_res, 
                                pattern_file = pattern_predict)
                rs = pool.map_async(funct, params)
                total = rs._number_left
                print(f'{total} pools generated')
                pool.close()
                while (not rs.ready()):
                    remaining = rs._number_left
                    done = total - remaining
                    print_progress_bar(done, total)
                    sleep(0.25)
                results = rs.get()

            time_1 = time()
            print(len(results))
            print_progress_bar(total, total)

            if time_1 - time_0 < 60.0:
                print(f'Done in {time_1 - time_0} s \n')
            if time_1 - time_0 > 60.0:
                print(f'Done in {(time_1 - time_0)/60} min')
            if time_1 - time_0 > 3600.0:
                print(f'Done in {(time_1 - time_0)/3600} hours')

