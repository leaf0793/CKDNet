from ast import Pass
from email.policy import default
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from datasets.data_io import get_transform, read_all_lines, pfm_imread
from albumentations.pytorch import ToTensorV2


def zero_num(btrain, all):
    if btrain==0:
        return 0
    res = all % btrain
    if res > 0:
        return btrain - (all % btrain)
    else:
        return 0



class SceneFlowDataset(Dataset):
    def __init__(self,  want_size, training=True, server_name='18.16', list_filename=None, cleanpass=False, btrain = 0, mode='val'):
        self.dict = {
            '18.16': '/data2',
            '17.17': '/data/cv/visual_team1/boyu',
            'guiyang': '/CV_team_data_01/pby_data/Dataset',
            '18.15': '/data/pby/dataset',
            '18.18': '/data3/pby/dataset',
            'local': '/data2/dataset',
            'LARGE':'/data/datasets',
        }
        self.btrain = btrain
        self.cleanpass = cleanpass
        self.server_name = server_name
        self.datapath = self.dict[server_name]
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.want_size = want_size
        # self.left_filenames, self.right_filenames, self.disp_filenames = self.load_all_path(self.datapath)

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [self.datapath+'/Sceneflow/'+x[0] for x in splits]
        right_images = [self.datapath+'/Sceneflow/'+x[1] for x in splits]
        disp_images = [self.datapath+'/Sceneflow/'+x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_all_path(self, path):
        if self.cleanpass:
            dir_path = '/Sceneflow/frames_cleanpass'
        else: 
            dir_path = '/Sceneflow/frames_finalpass'
        if self.training is True:
            left_path = find_all_file(path +
                                      dir_path + '/TRAIN', [])
        else:
            left_path = find_all_file(path +
                                      dir_path + '/TEST', [])
        right_path, disp_path = [], []
        left_path = sorted(left_path)
        for pt in left_path:
            right_path.append(
                pt.replace('left', 'right')
            )
            pt = pt.replace('.png', '.pfm').replace(dir_path,
                                               '/Sceneflow/flyingthings3d__disparity/disparity')
            if self.server_name == '18.16':
                pt = pt.replace(
                    '/data2', '/data1')
            disp_path.append(pt.replace('.png', '.pfm').replace(dir_path,
                                               '/Sceneflow/flyingthings3d__disparity/disparity'))
        
        # if self.zero_num > 0:
        for i in range(zero_num(self.btrain, len(left_path))):
            left_path.append("NONE")
            right_path.append("NONE")
            disp_path.append("NONE")
    
        return left_path, right_path, disp_path
        
    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        # if (self.left_filenames[index] == "NONE"):
        #     if self.training:
        #         size = (self.want_size[0], self.want_size[1])
        #     else:
        #         size = (540, 960)
        #     left_img = Image.new(mode="RGB", size=(
        #         size[1], size[0]), color=(0, 0, 0))
        #     right_img = Image.new(mode="RGB", size=(
        #         size[1], size[0]), color=(0, 0, 0))
        #     processed = get_transform()
        #     left_img = processed(left_img)
        #     right_img = processed(right_img)
        #     disparity = -np.ones(size, dtype=np.float32)
        #     if self.training:
        #         return [left_img.unsqueeze(0), right_img.unsqueeze(0), ToTensorV2()(image=disparity)['image'].unsqueeze(0)]
        #     else:
        #         return [left_img.unsqueeze(0), right_img.unsqueeze(0), ToTensorV2()(image=disparity)['image']]

        left_img = self.load_image(self.left_filenames[index])
        right_img = self.load_image(self.right_filenames[index])
        disparity = self.load_disp(self.disp_filenames[index])

        if self.training:
            w, h = left_img.size
            crop_h, crop_w = self.want_size

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            return [left_img, right_img, ToTensorV2()(image=disparity)['image']]

        else:
            w, h = left_img.size

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return [left_img, right_img, ToTensorV2()(image=disparity)['image']]



class KITTIDataset(Dataset):
    def __init__(self,  want_size, training, datapath=None, server_name='18.16', mode='val'):
        if training is True:
            # list_filename = './filenames/kitti15_train.txt'
            list_filename = './filenames/kitti15_all.txt'
        elif mode == 'val':
            list_filename = './filenames/kitti15_val.txt'
            # list_filename = './filenames/kitti15_train.txt'
        elif mode == 'test':
            list_filename = './filenames/kitti15_test.txt'
        # self.list_filename='/home/boyu.pan/work/code/mobilestereonet/filenames/kitti15_train.txt'
        self.dict = {
            '18.16': '/data2/kitti/data_scene_flow',
            '17.17': '/data/cv/baiyu.pan/kitti/data_scene_flow',
            'guiyang': '/CV_team_data_01/pby_data/Dataset',
            '18.15': '/data/pby/dataset/kitti/data_scene_flow',
            'local': '/data2/dataset/kitti/data_scene_flow',
            '18.18':'/data3/pby/dataset/kitti/data_scene_flow',
        }
        self.datapath = self.dict[server_name]
        
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(
            list_filename)
        self.training = training
        self.wantsize = want_size
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None

        if self.training:
            w, h = left_img.size
            crop_h, crop_w = self.wantsize
            
            x1 = random.randint(0, w - crop_w if (w - crop_w >= 0) else 0)
            y1 = random.randint(0, h - crop_h if (h - crop_h >= 0) else 0)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            disparity = ToTensorV2()(image=disparity)['image']

            return (left_img.unsqueeze(0), right_img.unsqueeze(0), disparity.unsqueeze(0))
        else:
            # w, h = left_img.size
            # crop_h, crop_w = self.wantsize
            # x1 = random.randint(0, w - crop_w if (w - crop_w >= 0) else 0)
            # y1 = random.randint(0, h - crop_h if (h - crop_h >= 0) else 0)

            # normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            if disparity is not None:
                disparity = ToTensorV2()(image=disparity)['image']
            # pad to size 1248x384
            # top_pad = 384 - h
            # right_pad = 1248 - w
            # assert top_pad > 0 and right_pad > 0
            # # pad images
            # left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            # right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
            #                        constant_values=0)
            # pad disparity gt
            # if disparity is not None:
            #     assert len(disparity.shape) == 2
            #     disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                return (left_img.unsqueeze(0), right_img.unsqueeze(0), disparity, self.left_filenames[index])

            else:
                return (left_img.unsqueeze(0), right_img.unsqueeze(0), self.left_filenames[index], self.right_filenames[index])





class KITTIDataset_1215(Dataset):
    def __init__(self,  want_size, training, datapath=None, server_name='18.16', mode='val'):
        if training is True:
            # list_filename = './filenames/kitti15_train.txt'
            list_filename = './filenames/kitti_12_15_train.txt'
        elif mode == 'val':
            list_filename = './filenames/kitti15_val.txt'
            # list_filename = './filenames/kitti15_train.txt'
        elif mode == 'test':
            # list_filename = './filenames/kitti15_test.txt'
            list_filename = './filenames/kitti12_test.txt'
        # self.list_filename='/home/boyu.pan/work/code/mobilestereonet/filenames/kitti15_train.txt'
        self.dict = {
            '18.16': '/data2/kitti/data_scene_flow',
            '17.17': '/data/cv/baiyu.pan/kitti/data_scene_flow',
            'guiyang': '/CV_team_data_01/pby_data/Dataset',
            '18.15': '/data/pby/dataset/kitti/data_scene_flow',
            'local': '/data2/dataset/kitti/data_scene_flow',
            '18.18':'/data3/pby/dataset/kitti/data_scene_flow',
        }
        self.datapath = self.dict[server_name]
        
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(
            list_filename)
        self.training = training
        self.wantsize = want_size
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_name = self.left_filenames[index].split('/')[1]
        if left_name.startswith('image'):
            datapath = self.datapath
        else:
            datapath = "/data2/dataset/kitti/data_stereo_flow"

        left_img = self.load_image(os.path.join(datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(datapath, self.right_filenames[index]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(datapath, self.disp_filenames[index]))
        else:
            disparity = None

        if self.training:
            w, h = left_img.size
            crop_h, crop_w = self.wantsize
            
            x1 = random.randint(0, w - crop_w if (w - crop_w >= 0) else 0)
            y1 = random.randint(0, h - crop_h if (h - crop_h >= 0) else 0)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            disparity = ToTensorV2()(image=disparity)['image']

            return (left_img.unsqueeze(0), right_img.unsqueeze(0), disparity.unsqueeze(0))
        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            if disparity is not None:
                disparity = ToTensorV2()(image=disparity)['image']
            # pad to size 1248x384
            # top_pad = 384 - h
            # right_pad = 1248 - w
            # assert top_pad > 0 and right_pad > 0
            # # pad images
            # left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            # right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
            #                        constant_values=0)
            # pad disparity gt
            # if disparity is not None:
            #     assert len(disparity.shape) == 2
            #     disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                return (left_img.unsqueeze(0), right_img.unsqueeze(0), disparity, self.left_filenames[index])
                # {"imgL": left_img,
                #         "imgR": right_img,
                #         "disp_L": disparity,
                #         "top_pad": top_pad,
                #         "right_pad": right_pad,
                #         "left_filename": self.left_filenames[index]}
            else:
                return (left_img.unsqueeze(0), right_img.unsqueeze(0), self.left_filenames[index], self.right_filenames[index])
                # {"imgL": left_img,
                #         "imgR": right_img,
                #         "top_pad": top_pad,
                #         "right_pad": right_pad,
                #         "left_filename": self.left_filenames[index],
                #         "right_filename": self.right_filenames[index]}


def find_all_file(dir, Filelist,ends='.png'):
    newDir = dir
    if os.path.isfile(dir):
        if dir.endswith(ends):
            Filelist.append(dir)
        # # 若只是要返回文件文，使用这个
        # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            if s == 'right':
                continue
            newDir = os.path.join(dir, s)
            find_all_file(newDir, Filelist)
    return Filelist


class KITTIDataset_1215(Dataset):
    def __init__(self,  want_size, training, dataset, list_filename=None, datapath=None, server_name='18.16', mode='val'):
        print('train on kitti{}'.format(dataset))
        assert dataset in ['12','15'], 'error'
        if dataset =='15':
            if training is True:
                list_filename = './filenames/kitti15_train.txt'
                # list_filename = './filenames/kitti_12_15_train.txt'
            elif mode == 'val':
                list_filename = './filenames/kitti15_val.txt'
                # list_filename = './filenames/kitti15_train.txt'
            elif mode == 'test':
                list_filename = './filenames/kitti15_test.txt'
                # list_filename = './filenames/kitti12_test.txt'
        elif dataset =='12':
            if training is True:
                # list_filename = './filenames/kitti12_train.txt'
                list_filename = './filenames/kitti_12_15_train.txt'
            elif mode == 'val':
                list_filename = './filenames/kitti12_val.txt'
                # list_filename = './filenames/kitti15_train.txt'
            elif mode == 'test':
                list_filename = './filenames/kitti12_test.txt'
            
        # elif mode == 'test12':
        #     list_filename = './filenames/kitti12_test.txt'
        # self.list_filename='/home/boyu.pan/work/code/mobilestereonet/filenames/kitti15_train.txt'
        self.dict = {
            '18.16': '/data2/kitti/data_scene_flow',
            '17.17': '/data/cv/baiyu.pan/kitti/data_scene_flow',
            'guiyang': '/CV_team_data_01/pby_data/Dataset',
            '18.15': '/data/pby/dataset/kitti/data_scene_flow',
            'local': '/data2/dataset/kitti/data_scene_flow',
            '18.18':'/data3/pby/dataset/kitti/data_scene_flow',
            'LARGE':'/data/datasets/KITTI/data_scene_flow'
        }
        self.datapath = self.dict[server_name]
        
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(
            list_filename)
        self.training = training
        self.wantsize = want_size
        self.processed = get_transform()
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_name = self.left_filenames[index].split('/')[1]
        if left_name.startswith('image'):
            datapath = self.datapath
        else:
            datapath = self.datapath.replace('data_scene_flow','data_stereo_flow')


        left_img = self.load_image(os.path.join(datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(datapath, self.right_filenames[index]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(datapath, self.disp_filenames[index]))
        else:
            disparity = None

        if self.training:
            w, h = left_img.size
            crop_h, crop_w = self.wantsize
            
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            # processed = get_transform()
            left_img = self.processed(left_img)
            right_img = self.processed(right_img)
            disparity = ToTensorV2()(image=disparity)['image']

            return (left_img, right_img, disparity)
        else:
            # w, h = left_img.size
            # crop_h, crop_w = self.wantsize
            
            # x1 = random.randint(0, w - crop_w if (w - crop_w >= 0) else 0)
            # y1 = random.randint(0, h - crop_h if (h - crop_h >= 0) else 0)

            # normalize
            # processed = get_transform()
            left_img = self.processed(left_img)
            right_img = self.processed(right_img)
            if disparity is not None:
                disparity = ToTensorV2()(image=disparity)['image']
            # pad to size 1248x384
            # top_pad = 384 - h
            # right_pad = 1248 - w
            # assert top_pad > 0 and right_pad > 0
            # # pad images
            # left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            # right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
            #                        constant_values=0)
            # pad disparity gt
            # if disparity is not None:
            #     assert len(disparity.shape) == 2
            #     disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                return (left_img, right_img, disparity, self.left_filenames[index])
                # {"imgL": left_img,
                #         "imgR": right_img,
                #         "disp_L": disparity,
                #         "top_pad": top_pad,
                #         "right_pad": right_pad,
                #         "left_filename": self.left_filenames[index]}
            else:
                return (left_img, right_img, self.left_filenames[index], self.right_filenames[index])
                # {"imgL": left_img,
                #         "imgR": right_img,
                #         "top_pad": top_pad,
                #         "right_pad": right_pad,
                #         "left_filename": self.left_filenames[index],
                #         "right_filename": self.right_filenames[index]}


def find_all_file(dir, Filelist):
    newDir = dir
    if os.path.isfile(dir):
        if dir.endswith('.png'):
            Filelist.append(dir)
        # # 若只是要返回文件文，使用这个
        # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            if s == 'right':
                continue
            newDir = os.path.join(dir, s)
            find_all_file(newDir, Filelist)
    return Filelist

if __name__ == '__main__':
    rgb = sorted(find_all_file('/data2/dataset/Sceneflow/frames_finalpass',[]))
    depth = sorted(find_all_file('/data2/dataset/Sceneflow/flyingthings3d__disparity/disparity',[]))
    # import os
    with open('files.txt', 'a+') as f:
        for l in rgb:
            f.writelines(l+"\n")
        for l in depth:
            f.writelines(l+"\n")
    # print(left[0])
