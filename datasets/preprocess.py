import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

#__imagenet_stats = {'mean': [0.5, 0.5, 0.5],
#                   'std': [0.5, 0.5, 0.5]}

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}



def totensor_normalize():

    return A.Compose([
        A.Normalize(
            mean=__imagenet_stats['mean'],
            std=__imagenet_stats['std'],
            max_pixel_value=255.0,
        ),
        ToTensorV2(always_apply=True)
    ], p=1)



def augmentv1():
    photometric  = [
        A.Blur(p=0.5),
        A.HueSaturationValue(20,30,20,p=0.5),
        A.RandomBrightnessContrast(0.2,p=0.5),
        A.RandomGamma(p=0.5),
        #A.ISONoise(p=1),
        A.GaussNoise(p=0.5),
        A.Normalize(
            mean=__imagenet_stats['mean'],
            std=__imagenet_stats['std'],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ]

    geometric = [
        # A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3,p=1)
        A.ShiftScaleRotate(shift_limit=0.01,scale_limit=0.01,rotate_limit=5,p=0.5)
        #A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=30, p=0.5)
    ]

    return A.Compose(photometric)

def get_transform(augment=True):


    if augment:
            return augmentv1()
    else:
            return totensor_normalize()







class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size=3):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


