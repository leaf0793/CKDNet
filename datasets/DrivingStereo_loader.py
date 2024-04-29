import torch
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import random
import cv2
# from . import preprocess
# from .preprocess import get_transform
from .data_io import totensor_normalize as get_transform
from .data_io import read_all_lines
# import preprocess

from albumentations.pytorch import ToTensorV2


# from . import preprocess
# IMG_EXTENSIONS = [
#     '.jpg', '.JPG', '.jpeg', '.JPEG',
#     '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
# ]


class Drivingstereo(data.Dataset):
    def __init__(self, want_size=(256, 256), training=True, mode='val', list_filename='filenames/driving_train.txt', dir_path="/data2/DrivingStereo", server_name='18.16'):
        self.dict = {
            '18.16': '/data2/DrivingStereo',
            '17.17': '/data/cv/baiyu.pan',
            'guiyang': '/CV_team_data_01/pby_data/Dataset/DrivingStereo',
            'local':'/data2/dataset/DrivingStereo',
            'LARGE':'/data/datasets/DrivingStereo',
        }
        # self.dir_path = self.dict[server_name]
        self.datapath = self.dict[server_name]
        self.training = training
        # self.left_path, self.right_path, self.disparity_path, self.depth_path = find_all_file(
        #     self.dir_path) if self.training else find_all_testfile(self.dir_path)
        self.left_path, self.right_path, self.disparity_path = self.load_path(list_filename)
        self.augment_more = False
        self.want_size = want_size  # (H,W)
        # print('Dataset initialized')
        # print('len', len(self.left_path))

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [self.datapath+'/'+x[0] for x in splits]
        right_images = [self.datapath+'/'+x[1] for x in splits]
        disp_images = [self.datapath+'/'+x[2] for x in splits]
        return left_images, right_images, disp_images


    def loader(self, path, type=1):
        if os.path.isfile(path):
            image = cv2.imread(path, type)
            if type ==1:
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            return image
        else:
            return None

    def __getitem__(self, index):
        left = self.left_path[index]
        right = self.right_path[index]
        disp_L = self.disparity_path[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.loader(disp_L, -1)
        assert left_img is not None and right_img is not None and dataL is not None
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)/255
        # print("dataL:{}".format(dataL.shape))
        th, tw = self.want_size[0], self.want_size[1]
        if self.training:
            h, w = dataL.shape
            assert th <= h and tw <= w
            x1 = random.randint(0, w - tw)
            y1 =\
                random.randint(0, h - th)

            left_img = left_img[y1:y1+th, x1:x1+tw, :]
            right_img = right_img[y1:y1+th, x1:x1+tw, :]
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = get_transform()
            # To -> Tensor(C,H,W)
            left_img = processed(image=left_img)['image']
            right_img = processed(image=right_img)['image']

            # if self.augment_more:
            #     delta_h = np.floor(np.random.uniform(50, 180))
            #     delta_w = np.floor(np.random.uniform(50, 250))
            #     x1_aug = random.randint(0, th - delta_h)
            #     y1_aug = random.randint(0, tw - delta_w)
            #     x2_aug = random.randint(0, th - delta_h)
            #     y2_aug = random.randint(0, tw - delta_w)
            #     right_img[:, int(x1_aug):int(x1_aug+delta_h), int(y1_aug):int(y1_aug+delta_w)] =\
            #         right_img[:, int(x2_aug):int(x2_aug+delta_h),
            #                   int(y2_aug):int(y2_aug+delta_w)]
            return [left_img, right_img, ToTensorV2()(image=dataL)['image']]
        else:
            # h = left_img.shape[0]
            # w = left_img.shape[1]
            # left_img = left_img[y1:y1 + th, x1:x1 + tw, :]
            # right_img = right_img[y1:y1 + th, x1:x1 + tw, :]

            # dataL = dataL[y1:y1 + th, x1:x1 + tw]
            # if left_img.shape[1:] != self.want_size or right_img.shape[1:] != self.want_size:
            # print(left_img.shape)
            # top_pad = self.want_size[0] - left_img.shape[0]
            # right_pad = self.want_size[1] - left_img.shape[1]
            # # print(top_pad,right_pad)
            # left_img = np.pad(
            #     left_img, ((top_pad, 0), (0, right_pad), (0, 0)),
            #     'constant', constant_values=(0, 0)
            # )
            # right_img = np.pad(
            #     right_img, ((top_pad, 0), (0, right_pad), (0, 0)),
            #     'constant', constant_values=(0, 0)
            # )
            # top_pad = dataL.shape[0]-800
            # right_pad = 1762 - dataL.shape[1]
            # dataL = np.pad(
            #     dataL, ((dataL.shape[0]-800, 0),
            #             (0, 1762 - dataL.shape[1])),
            #     'constant', constant_values=(0, 0)
            # )
            # print(left_img.shape)
            processed = get_transform()
            left_img = processed(image=left_img)['image']
            right_img = processed(image=right_img)['image']

            return [left_img, right_img, ToTensorV2()(image=dataL)['image']]

    def __len__(self):
        return len(self.left_path)


def find_all_file(file_dir):
    file_dir += '/train-left-image'
    files = os.listdir(file_dir)
    lefts, rights, disparitys, depths = [], [], [], []
    for file in files:  # 遍历文件夹
        if os.path.isdir(file_dir+'/'+file):  # 判断是否是文件夹，不是文件夹才打开
            for filename in os.listdir(file_dir+'/'+file):
                if(filename.endswith('.jpg')):
                    lefts.append(file_dir+'/'+file+'/'+filename)
    lefts = sorted(lefts)
    for i in lefts:
        right = i.replace('-left-', '-right-')
        disparity = i.replace(
            '-left-image', '-disparity-map').replace('.jpg', '.png')
        depth = i.replace('-left-image', '-depth-map').replace('.jpg', '.png')
        if os.path.exists(right) and os.path.exists(disparity) and os.path.exists(depth):
            rights.append(right)
            disparitys.append(disparity)
            depths.append(depth)
    return lefts, rights, disparitys, depths


def find_all_testfile(file_dir):
    file_dir += '/test-left-image/left-image-half-size'
    files = os.listdir(file_dir)
    left, right, disparity, depth = [], [], [], []
    for roots, dirs, files in os.walk(file_dir):
        for file in files:
            if(file.endswith('.jpg')):
                filename = roots + '/'+file
                left.append(filename)
    # for file in files:  # 遍历文件夹
    #     filename = file_dir+'/'+file
    #     if(filename.endswith('.jpg')):
    #         left.append(filename)
    for i in left:
        right.append(i.replace('left-', 'right-'))
        disparity.append(
            i.replace('left-image', 'disparity-map').replace('.jpg', '.png'))
        depth.append(i.replace('left-image', 'depth-map'))
    return left, right, disparity, depth


if __name__ == '__main__':
    import os
    dataset = Drivingstereo(dir_path="/data2/DrivingStereo", training=False)
    print(len(dataset))
    # did = dataset.__getitem__(0)
    # a = did

    # assert os.path.isfile('/media/pan/53E384F363C56C79/Work/Project/dis2dep/disparity/dataloader/log.txt')
    # with open('/media/pan/53E384F363C56C79/Work/Project/dis2dep/disparity/dataloader/log.txt', 'w') as f:
    #     s = find_all_file(
    #         '/media/pan/53E384F363C56C79/Work/Dataset/DrivingStereo')
    #     disparity = s[2]
    #     max_value = 0
    #     for image in disparity:
    #         a = cv2.imread(image, -1)
    #         max_c = np.amax(a)
    #         # print(max_c)
    #         f.write('{:d}\n'.format(max_c))
    #         max_value = max_c if max_c > max_value else max_value
    #     f.write('{:d}\n'.format(max_value))

    # Drivingstereo

    # print([len(_) for _ in s])
    # a = cv2.imread(s[0][0])
    # set = Drivingstereo(
    #     '/media/pan/53E384F363C56C79/Work/Dataset/DrivingStereo')

    # print(len(set[0]))
    # cv2.imshow()
    # print(a.shape)
