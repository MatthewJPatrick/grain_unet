from utility.settings import *
from utility.transformation import training_transforms
from unet_model.training.ImageDataset import ImageDataset
from torch.utils.data import random_split
import torch

def get_train_data_loaders(image_path:str = IMAGE_PATH, label_path:str = LABEL_PATH, training_split:float = TRAINING_SPLIT, batch_size:int = BATCH_SIZE, num_workers=NUM_WORKERS, transform_generator=training_transforms()):
    

    transforms = transform_generator()
    imageset = ImageDataset(
        image_dir=image_path,
        label_dir=label_path,
        transform=transforms
    )

    print('images found = ',imageset.__len__())

    # Define the sizes of your training and validation sets
    train_size = int(float(training_split) * len(imageset))
    val_size = len(imageset) - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(imageset, [train_size, val_size])


    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return train_dataloader, val_dataloader