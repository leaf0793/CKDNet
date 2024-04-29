import re
import numpy as np
import torchvision.transforms as transforms
# from datasets.preprocess import GaussianBlur
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

# def get_transform2(size, s=1):
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
#     data_transforms = transforms.Compose([
#         # transforms.RandomResizedCrop(size=size),
#         # transforms.RandomHorizontalFlip(),
#         transforms.RandomApply(
#             [color_jitter], p=0.2),
#         transforms.RandomGrayscale(p=0.2),
#         GaussianBlur(
#             kernel_size=int(0.1 * size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std)]
#         )

#     return data_transforms



def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  
        endian = '<'
        scale = -scale
    else:
        endian = '>'  

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def totensor_normalize():
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(always_apply=True)
    ], p=1)