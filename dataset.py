import torch
import os
import pandas as pd
from PIL import Image

"""
Defines a Steering Angle Dataset

__getitem__ returns:
    image (PIL image object): image from front-facing car camera
    steering_angle (float): angle at which steering wheel is turned
        in associated photo (degrees)
"""

class SteeringAngleDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, img_dir, transform=None):
        self.datapoints = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, index):
        example = self.datapoints.iloc[index]

        img_path = os.path.join(self.img_dir, example[0])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        steering_angle = example[1]

        return image, steering_angle
