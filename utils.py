import torch
import torchvision
from torch.utils import data
import os
import cv2
import csv
import numpy as np
import shutil
from random import randint, shuffle

class Dataset(data.Dataset):
    def __init__(self, dirs, resize=64, scale_by=4, also_valid=False):
        self.resize = resize
        self.scale_by = scale_by
        self.train_size = self.resize // self.scale_by
        self.img_list = []
        self.also_valid = also_valid
        self.valid_list = []

        for dir in dirs:
            if os.path.isfile(dir):
                if dir.endswith('.csv'):
                    with open(dirs, 'r') as file:
                        reader = csv.reader(file)
                        images = list(reader)
                        for path in images:
                            self.img_list.append(path[0])
            else:
                getFiles(dir, self.img_list)

        if self.also_valid: # For visualization
            shuffle(self.img_list)
            n_data = len(self.img_list)

            if n_data * 0.01 > 100:
                n_valid = 100
            elif n_data * 0.01 > 10:
                n_valid = 10
            else:
                n_valid = 1

            self.valid_list = self.img_list[-n_valid:]
            self.img_list = self.img_list[:-n_valid]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index): # Transformation in CPU (Make LR using inter-cubic interpolation)
        img_path = self.img_list[index]
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=(self.resize, self.resize), interpolation=cv2.INTER_CUBIC)
        img = augmentation(img)
        lr_img, gt_img = downsample(img, size=(self.train_size,))
        lr_img = normalization(lr_img)
        gt_img = normalization(gt_img)
        to_tensor = torchvision.transforms.ToTensor()
        lr_img = to_tensor(lr_img)
        gt_img = to_tensor(gt_img)

        name = os.path.basename(img_path)
        name = name[:-4]

        return lr_img, gt_img, name

    def clone_for_validation(self):
        if not self.also_valid:
            raise AttributeError("Dataset only for training, not for validation")
        else:
            valid_dataset = evaluation_dataset(dirs=None, resize=self.resize, scale_by=self.scale_by)
            valid_dataset.img_list = self.valid_list

            return valid_dataset


class evaluation_dataset(data.Dataset):
    def __init__(self, dirs, scale_by, resize=64, down_grade=True):
        self.scale_by = scale_by
        self.size = resize
        self.down_grade = down_grade
        self.img_list = []
        dirs = [dirs]
        for d in dirs:
            getFiles(d, self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_file = self.img_list[index]
        img = cv2.imread(img_file)
        if self.size is not None:
            img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_CUBIC)
        img_name = os.path.basename(img_file)
        img_name = img_name[:-4]
        to_tensor = torchvision.transforms.ToTensor()

        if self.down_grade:
            h = img.shape[0]
            w = img.shape[1]

            lr_h = h // 4
            lr_w = w // 4

            lr_img, gt_img = downsample(img, size=(lr_h, lr_w))
            lr_img = normalization(lr_img)
            gt_img = normalization(gt_img)
            lr_img = to_tensor(lr_img)
            gt_img = to_tensor(gt_img)
        else:
            lr_img = normalization(img)
            lr_img = to_tensor(lr_img)
            gt_img = lr_img

        return lr_img, gt_img, img_name


def getFiles(dir, dataList):
    if dir is None:
        return None
    if os.path.isdir(dir):
        temp_dataList = os.listdir(dir)
        for directory in temp_dataList:
            directory = os.path.join(dir, directory)
            getFiles(directory, dataList)
    elif os.path.isfile(dir):
        if dir.endswith('.png') or dir.endswith('.jpeg') or dir.endswith('.jpg'):
            dataList.append(dir)

    return dataList


def crop_img(image, size):
    h = randint(0, image.shape[0] - size)
    w = randint(0, image.shape[1] - size)

    cropped_img = image[h: h + size, w: w + size]
    return cropped_img


def augmentation(image):
    # flipping
    flip_flag = randint(0, 1)
    if flip_flag == 1:
        image = cv2.flip(image, 1)

    # rotation
    rot = randint(0, 359)
    if rot < 90:
        rot = 0
    elif rot < 180:
        rot = 90
    elif rot < 270:
        rot = 180
    else:
        rot = 270

    w = image.shape[1]
    h = image.shape[0]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, rot, scale=1.0)
    if rot == 90 or rot == 270:
        image = cv2.warpAffine(image, M, (h, w))

    elif rot == 180:
        image = cv2.warpAffine(image, M, (w, h))

    # random blur
    return image


def downsample(image, size):
    if len(size) == 1:
        h = size[0]
        w = size[0]
    elif len(size) == 2:
        h = size[0]
        w = size[1]
    else:
        raise ValueError('Size should be a single value or a pair of values')

    lr_img = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    return lr_img, image


def normalization(image, _from=(0, 255), _to=(-1, 1)):
    from_min, from_max = _from
    to_min, to_max = _to

    from_range = from_max - from_min
    to_range = to_max - to_min

    if from_range < 0 or to_range < 0:
        raise ValueError('wrong range input: form should be (min, max)')

    scale = from_range / to_range
    image = image / scale - to_range / 2

    return image


def renormalization(image, _from=(-1, 1), _to=(0, 255)):
    from_min, from_max = _from
    to_min, to_max = _to

    from_range = from_max - from_min
    to_range = to_max - to_min

    if from_range < 0 or to_range < 0:
        raise ValueError('wrong range input: form should be (min, max)')

    scale = to_range / from_range
    image = (image + from_range / 2) * scale

    return image


def sample_from_dataset(directory, n_samples):
    list_of_data = []
    getFiles(directory, list_of_data)
    sample_list = []
    for i in range(n_samples):
        data = list_of_data[i]
        sample_list.append(data)

    return sample_list


def save_sample(dst, sample_list):
    for img in sample_list:
        name = os.path.basename(img)
        copied_img = os.path.join(dst, name)
        shutil.copy(img, copied_img)

###############################################################################################

from skimage.measure import compare_psnr as psnr_metric
from skimage.measure import compare_ssim as ssim_metric
from scipy import misc

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1)
    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size())
    if torch.cuda.is_available():
        fake = fake.cuda()

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def eval(gt, pred):
    '''
    image input is (BS, C, H, W)
    '''

    gt = (gt + 1.) * 127.5
    pred = (pred + 1.) * 127.5

    gt = np.clip(gt, 0., 255.).astype(np.uint8)
    pred = np.clip(pred, 0., 255.).astype(np.uint8)

    batch_size = np.shape(gt)[0]
    ssim = np.zeros([batch_size])
    psnr = np.zeros([batch_size])
    mse = np.zeros([batch_size])

    for bs in range(batch_size):
        for c in range(gt[bs].shape[0]):

            ssim[bs] += ssim_metric(gt[bs][c], pred[bs][c])
            psnr[bs] += psnr_metric(gt[bs][c], pred[bs][c])
        mse[bs] += mse_metric(gt[bs], pred[bs])

    psnr /= 3
    ssim /= 3

    return mse, ssim, psnr


def save_images(images, size, image_path):
    '''
    image input is (BS, C, H, W)
    '''
    images = (np.transpose(images, [0, 2, 3, 1]) + 1.) * 127.5
    images = np.clip(images, 0., 255.).astype(np.uint8)
    images = merge(images, size)

    misc.imsave(image_path, images)


def merge(images, size):

    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    cnt = 0

    for i in range(size[0]):
        for j in range(size[1]):

            img[i*h:(i+1)*h, j*w:(j+1)*w, :] = images[cnt]
            cnt += 1

    return img


def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err
