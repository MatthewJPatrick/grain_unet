

import torch.nn as nn
import torch
import cv2
from pathlib import Path
import csv
import re 
from matplotlib import pyplot as plt


def extract_integer(filename):
    """
    Extracts the integer from a filename of the format something_something_<digits>.pth.png

    Args:
        filename (str): The filename to process.

    Returns:
        int: The extracted integer, or None if no match is found.
    """
    match = re.search(r'_(\d+)\.pth\.png$', filename)
    if match:
        return int(match.group(1))
    return 300

gt_path = 'losses/expectedOutput_299.png'
image_folder = 'losses/299'


loss_fn = nn.BCELoss()


gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
gt_data = torch.tensor(gt_image/255)
print(gt_data)
images_for_comparison = sorted(list(Path(image_folder).glob('*.png')))

losses = []

for image_path in images_for_comparison:

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image_data = torch.tensor(image/255)
    epoch = extract_integer(image_path.name)
    
    loss = loss_fn(image_data, gt_data)
    print(loss.item())
    losses.append([epoch, loss.item()])

losses = sorted(losses, key=lambda x: x[0])
#save a csv file with the losses
csv.writer(open('299_losses.csv', 'w+')).writerows(losses)


