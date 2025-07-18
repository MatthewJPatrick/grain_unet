def train_unet(directory, save_name, epochs = 30, batch_size = 5):
    
    print(directory)
    img_data, label_data = import_data(directory)
    train_seq(img_data, label_data, save_name, epochs=epochs, batch_size= batch_size)


def import_data(directory):
    #directory is the location of the images you want to import. Enter in a format like "Data/train_Pt_16'

    from pathlib import Path

    global input_names, label_names

    train_dir = Path(directory)

    input_names = list(train_dir.glob('*/image/*.png'))
    label_names = list(train_dir.glob('*/label/*.png'))
    print(f"Found {len(input_names)} samples and {len(label_names)} tracings")

    return (input_names, label_names)

def train_seq(input_names, label_names, save_name, epochs = 30, batch_size=5):

    #directory is the path where your images are located. Enter in a format like 'Data/train_nouveaux_256'
    #batch_size has a default value of 5 if not specified
    #save_name is the name of your trained network hdf5 file that you will use. A format like 'unet_grain_nouveaux_pretune.hdf5' is recommended

    from grain_sequence import GrainSequence
    import random
    from unet import get_unet
    from tensorflow import keras

    global history, train_gen, valid_gen

    validation_samples = len(input_names) // 10

    random.Random(1337).shuffle(input_names)
    random.Random(1337).shuffle(label_names)

    train_input = input_names[:-validation_samples]
    train_label = label_names[:-validation_samples]
    train_gen = GrainSequence(batch_size, (256, 256), train_input, train_label)

    valid_input = input_names[-validation_samples:]
    valid_label = label_names[-validation_samples:]
    valid_gen = GrainSequence(batch_size, (256, 256), valid_input, valid_label)

    print(f"Training set size: {len(train_input)}, {len(train_gen)} batches")
    print(f"Validation set size: {len(valid_input)}, {len(valid_gen)} batches")

    if False:
        _ = [print(f'{ind}:\n{i}\n{l}\n\n') for ind, (i, l) in enumerate(zip(train_input, train_label))]

    model = get_unet(input_size=(256, 256, 1))
    model_checkpoint = keras.callbacks.ModelCheckpoint(save_name, monitor='loss', verbose=1, save_best_only=False)
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=epochs,
        callbacks=[model_checkpoint],
        validation_data=valid_gen
    )

def train_loss_plt():

    import matplotlib.pyplot as plt

    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']

    plt.figure(figsize=(10, 5), facecolor='White')

    plt.title("Loss With Pre Augmentation and Full Dataset (b5, s59)")
    plt.xlabel("Epochs")
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.legend()
    plt.show()

def n_test(image_size_x, image_size_y, directory, path_pattern, network):

    #image_size_x and image_size_y are the sizes of your images in pixels along the x and y axes, respectively
        #use image_size_x = 1216 and image_size_y = 1216 for Pt images
    #directory is the name of the directory you want to test. Enter in a format like 'Data/test_Pt'
    #network is the name of the hdf5 network file previously created that you would like to use. Enter in a format like "unet_grain_nouveaux_pretune.hdf5"

    from pathlib import Path
    from skimage import io
    from src import get_unet, image_generator
    import numpy as np

    global Model

    test_dir = Path(directory)
    paths = list(test_dir.glob(path_pattern))
    target_size = (image_size_x, image_size_y)
    
    paths = list(test_dir.glob(path_pattern))

    print(f'{len(paths)} images found')
 
    img_gen = image_generator(paths, target_size=target_size)

    model = get_unet(input_size=(target_size + (1,)))
    model.load_weights(network)
    results = 255 * model.predict(img_gen, steps=len(paths), verbose=1)

    assert len(paths) == len(results), 'Not all the files ran'

    for ind, path in enumerate(paths):
        save_dir = path.parents[1] / f'predict_{image_size_x}'
        if not save_dir.is_dir():
            print('made dir')
            save_dir.mkdir()

        save_path = save_dir / path.with_suffix('.png').name
        result = results[ind, :, :, 0]

        print(f"\nSaving to {save_path}")
        print(f"Min: {np.min(result)}, Max: {np.max(result)}, Shape: {result.shape}")
        io.imsave(save_path, result.astype('uint8'))
        if ind == 0:
            io.imshow(result)
            io.show()

def load_trained(model, name):

    #model is the name of the hdf5 netowrk you would like to use. Enter in a format like 'unet_grain_nouveaux_pretune.hdf5'
    #name is the new name for the transfer network. A format like 'unet_grain_nouveaux_transfer.hdf5' is recommended

    from src import get_unet
    from tensorflow import keras
    import matplotlib.pyplot as plt

    model1 = get_unet(input_size=(256, 256, 1))

    load_model = model1
    load_model.load_weights(model)

    out = load_model.layers[-1].output

    model = Model(inputs=model1.input, outputs=out)
    opt = keras.optimizers.legacy.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # freezing the layers of the network we don't want to train

    for layer in model.layers[:16]:  ###:16
        layer.trainable = False
    model.summary()

    save_name = name

    model_checkpoint = keras.callbacks.ModelCheckpoint(save_name, monitor='loss', verbose=1, save_best_only=False)
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=20,
        callbacks=[model_checkpoint],
        validation_data=valid_gen
    )

    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']

    plt.figure(figsize=(10, 5), facecolor='White')

    plt.title("Loss With Pre Augmentation and Full Dataset (b5, s59)")
    plt.xlabel("Epochs")
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.legend()
    plt.show()

    # unfreezing network

    for layer in model.layers[:16]:  # 16:
        layer.trainable = True
    model.summary()

    opt = keras.optimizers.Adam(learning_rate=5e-6)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    model_checkpoint = keras.callbacks.ModelCheckpoint(save_name, monitor='loss', verbose=1, save_best_only=False)
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=10,
        callbacks=[model_checkpoint],
        validation_data=valid_gen
    )

    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']

    plt.figure(figsize=(10, 5), facecolor='White')

    plt.title("Loss With Pre Augmentation and Full Dataset (b5, s59)")
    plt.xlabel("Epochs")
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.legend()
    plt.show()

def post_process_folder(process_args = {
        'compilation': 'min',
        'liberal_thresh': 200,  ##220-180 for best (was 240) 2_test is 200
        'conservative_thresh': 160,  ##was 200
        'invert_double_thresh': True,
        'n_dilations': 3,
        'min_grain_area': 100,  # usually 300, cut to 100 for 2019 data
        'prune_size': 30,
        'out_dict': True
    }, test_directory = '', pattern_prediction_folder = '*', pattern_file = '*.png'):

    #function finds all files in subfolders of "directory". File paths are of the form
    # directory / * / pattern. The default is directory/*/predict/*.png
    

    # kwargs:
    #    'compilation': (default 'min') defines the image compilation technique
    #    'liberal_thresh': (default 200) liberal threshold for double threshold
    #    'conservative_thresh': (default 160) conservative threshold for double threshold
    #    'invert_double_thresh': (default True) Changes < to > in double threshold
    #    'n_dilations': (default 3) Number of dilations to apply in closing
    #    'min_grain_area': (default 100) Max size of a hole to close
    #    'prune_size': (default 30) Size to prune with plantcv
    #    'out_dict': (default False) return a dict with all the intermediate steps
    import os
    import numpy as np
    from pathlib import Path
    from skimage import io, exposure
    from src import post_process, printProgressBar
   
    test_directory = Path(test_directory) 

    for FOV in test_directory.glob(pattern_prediction_folder): 
        imgs = np.array([])
        for fname in FOV.glob(pattern_file):
            if not fname.is_file(): 
                continue

            img = io.imread(fname)
            if len(img.shape) > 2:
                img = img[:, :, 0]

            if len(imgs) == 0:
                imgs = img
            else:
                imgs = np.dstack((imgs, img))

        #generate post-processed image
        try:
            data = post_process(imgs, **process_args)
        except:
            print('Could not post-process. Check that the files exist!')
            print(f'FoV      : {FOV}')
            print(f'Filename : {fname}')

        #make a folder in case there is not a post-processed folder
        if not (FOV / 'post_processed').is_dir():
            os.mkdir(FOV / 'post_processed')
        
        # Name file with its post processing parameters in the file's name
        output_name = f"{os.path.split(fname)[1].split('.')[0]}_dils_{str(process_args['n_dilations']).zfill(3)}_cons_{str(process_args['conservative_thresh']).zfill(3)}_lib_{str(process_args['liberal_thresh']).zfill(3)}_prune_{str(process_args['prune_size']).zfill(3)}_minarea_{str(process_args['min_grain_area']).zfill(3)}"

        # Only save the file if it contains an image, i.e. it it not low-contrast
        if not exposure.is_low_contrast(255*data['pruned_skeleton'].astype('uint8')):
            print(f'saving to {output_name}')
            io.imsave( FOV / 'post_processed' / f'unet_skel_{output_name}_new.png', 255 * data['pruned_skeleton'].astype('uint8'))

        else:
            print('Low contrast!!!')


def overlay(directory):
    # directory is the name of the test directory you would like to use. Enter in a format like 'Data/test_Pt'

    from skimage import io, transform
    from pathlib import Path
    import numpy as np
    import os

    test_dir = Path(directory)

    for FOV in test_dir.glob('*'):
        if not (FOV / 'raw').is_dir():
            continue
        for fname in FOV.glob('raw/*.png'):
            raw_img = io.imread(fname)

            if (np.max(raw_img) > 255):  # if not 8 bit depth, scale
                raw_img = raw_img - raw_img.min()
                raw_img = 255 * (raw_img / np.ptp(raw_img))

            sk_img = io.imread(FOV / 'unet_skel.png')  # sk_img is unet
            sk_img = 255 * transform.resize(sk_img, raw_img.shape,
                                            anti_aliasing=False)  # resizing network output to raw
            fusion_img = np.zeros((raw_img.shape[0], raw_img.shape[1], 3), dtype=int)
            fusion_img[:, :, 0] = raw_img
            fusion_img[:, :, 1] = raw_img
            fusion_img[:, :, 2] = raw_img
            fusion_img[sk_img > 0, 0] = 255

            '''ht_img = io.imread(FOV / 'key' / 'trace.png')
            ht_img = 255 - (255*transform.resize(ht_img, raw_img.shape, anti_aliasing=False))
            fusion_img[ht_img > 0, 1] = 255'''
            io.imshow(fusion_img)
            io.show()
            io.imsave(FOV / 'unet_overlay.png', fusion_img.astype('uint8'))
            print('saving to ' + str(FOV) + '/unet_overlay.png')
            break

def gen_parameter_list(liberal_thresholds, con_thresh_diffs, prune_sizes, dilations, min_areas):
    #define a list of post-processing parameter dictionaries based on the settings above

    total_attempts = len(liberal_thresholds)*len(con_thresh_diffs)*len(prune_sizes)*len(dilations)*len(min_areas)
    params = [dict()]*total_attempts

    ii = 0
    for lib_thresh in liberal_thresholds:
        for con_thresh_diff in con_thresh_diffs:
            for prune_size in prune_sizes:
                for min_area in min_areas:
                    for dils in dilations:
                        params[ii] = {
                            'compilation': 'min',
                            'liberal_thresh': lib_thresh,     ##220-180 for best axon
                            'conservative_thresh': lib_thresh - con_thresh_diff, 
                            'invert_double_thresh': True,
                            'n_dilations': dils,
                            'min_grain_area': min_area, #usually 300, cut to 100 for 2019 data
                            'prune_size': prune_size, #usually 100
                            'out_dict': True
                            }
                        ii += 1
    return params
