
from torchvision import transforms
import utility.settings as globals
# Define transformations

def inference_transforms(target_resolution:int=globals.TARGET_RESOLUTION)->transforms.Compose:
    image_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),# Convert image to grayscale
        transforms.Resize((target_resolution,target_resolution)),#Resize the image to 256x256 pixels
    ])
    return image_transform

def training_transforms(TARGET_RESOLUTION:int=256, prob_flip = 1.0)->transforms.Compose: #Note, prob is 1 so that we can apply to label and image at same time in the dataloader

    horizontal_flip = transforms.Compose([
 # Converts to Tensor and scales to [0, 1]
    transforms.Resize((TARGET_RESOLUTION, TARGET_RESOLUTION)),
    transforms.RandomHorizontalFlip(prob_flip),
    ])

    vertical_flip = transforms.Compose([
        #transforms.RandomRotation(degrees=90),  # Randomly rotate the image
        #transforms.Lambda(lambda img: img.rotate(random.choice([90,180,270]))),
        transforms.Resize((TARGET_RESOLUTION, TARGET_RESOLUTION)),#Resize the image to 256x256 pixels
        transforms.RandomVerticalFlip(prob_flip)  # Randomly flip the image vertically
        # If labels are categorical indices, consider using a custom transform to convert them to long tensor instead
    ])

    no_transform = transforms.Compose([
        # Converts to Tensor, assume labels can also be scaled to [0, 1]
        transforms.Resize((TARGET_RESOLUTION, TARGET_RESOLUTION)),#Resize the image to 256x256 pixels
        # If labels are categorical indices, consider using a custom transform to convert them to long tensor instead
    ])

    other_transforms = None


    return {"horizontal_flip":horizontal_flip, "vertical_flip":vertical_flip, "no_transform":no_transform, "other_transforms":other_transforms}