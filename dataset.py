import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CocoDetection
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, transform=None, train=True, filter="sobel"):
        coco_root = '/data/sarthak/coco'
        self.filter = filter

        if train:
            # Load the COCO train dataset
            self.dataset = CocoDetection(root=f'{coco_root}/train2017', 
                                        annFile=f'{coco_root}/annotations/instances_train2017.json', 
                                        transform=transform)
        else:
            # Load the COCO train dataset
            self.dataset = CocoDetection(root=f'{coco_root}/val2017', 
                                        annFile=f'{coco_root}/annotations/instances_val2017.json', 
                                        transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image_gray = transforms.Grayscale()(image)
        target = get_gt_values(image_gray, self.filter)

        return (image_gray, target)


# Function to generate Sobel-filtered images
def get_gt_values(image, filter):
    if filter == 'sobel':
        filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        out_x = np.expand_dims(filter_x, axis=(0, 1))
        out_y = np.expand_dims(filter_y, axis=(0, 1))

        out_x = torch.from_numpy(out_x).float()
        out_y = torch.from_numpy(out_y).float()

        image = image.unsqueeze(0)

        fil_out_x = nn.functional.conv2d(image, out_x, padding=1)
        fil_out_y = nn.functional.conv2d(image, out_y, padding=1)

        fil_magnitude = torch.sqrt(fil_out_x**2 + fil_out_y**2)

        return fil_magnitude.squeeze(0)
    
    elif filter == 'avg':
        # Averaging filter
        filter_avg = np.ones((3, 3)) / 9.0  

        out_avg = np.expand_dims(filter_avg, axis=(0, 1))
        out_avg = torch.from_numpy(out_avg).float()

        image = image.unsqueeze(0)

        fil_out_avg = nn.functional.conv2d(image, out_avg, padding=1)

        return fil_out_avg.squeeze(0)
    
    elif filter == 'blur':
        # Blurring filter -- similar to avg filter
        filter_custom = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

        out_custom = np.expand_dims(filter_custom, axis=(0, 1))
        out_custom = torch.from_numpy(out_custom).float()

        image = image.unsqueeze(0)

        fil_out_custom = nn.functional.conv2d(image, out_custom, padding=1)

        return fil_out_custom.squeeze(0)
    
    else:
        # define custom filter here
        filter_x = np.array([])
        filter_y = np.array([])

    
