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

class SteeringAngleDiffDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, img_dir, transform=None, lookback=1):
        self.datapoints = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.lookback = lookback

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, index):
        example = self.datapoints.iloc[index]

        img1_name = example[0]
        img1_number = int(img1_name.split('.')[0])
        img2_number = img1_number - self.lookback
        if img2_number < 0:
            img2_number = 0

        img2_name = str(img2_number) + '.jpg'

        img1_path = os.path.join(self.img_dir, img1_name)
        img2_path = os.path.join(self.img_dir, img2_name)

        image1 = Image.open(img1_path)
        image2 = Image.open(img2_path)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        image_diff = image1-image2
        
        steering_angle = example[1]

        return image_diff, steering_angle
