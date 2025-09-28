import glob
import os
import random
import cv2
import PIL.Image as Image
import numpy as np
import rasterio
from torch.utils.data import Dataset
import torch.nn.functional as F


def load_image(fname, mode='RGB',  return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
        out_img = img.astype('float32') / 255
    else:
        out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img


def load_tif(fname, return_orig=False):
    # 判断文件扩展名
    if fname.endswith('.png'):
        # 读取PNG图像
        img = np.array(Image.open(fname).convert("L"))
    elif fname.endswith('.tif'):
        # 读取TIF图像
        with rasterio.open(fname) as src:
            # 读取RGB波段，假设是3个波段
            img = src.read([1, 2, 3, 4])  # 读取前3个波段
    else:
        raise ValueError(f"Unsupported file format: {fname}")

    # 将图像转换为float32类型并归一化
    if img.ndim == 3:
        out_img = img.astype('float32') / 10000
    else:
        out_img = img.astype('float32') / 255

    # 返回原始图像和处理后的图像
    if return_orig:
        return out_img, img
    else:
        return out_img


class InpaintingDataset(Dataset):
    def __init__(self, datadir, img_suffix='.png'):
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.region_dict = self._create_region_dict()

    def __len__(self):
        return len(self.mask_filenames)

    def _create_region_dict(self):
        region_dict = {}
        for file in self.img_filenames:
            region_id = os.path.basename(file).split('_')[:2]
            region_id = '_'.join(region_id)
            if region_id in region_dict:
                region_dict[region_id].append(file)
            else:
                region_dict[region_id] = [file]
        return region_dict

    def __getitem__(self, i):
        image = load_image(self.img_filenames[i], mode='RGB')
        mask = load_image(self.mask_filenames[i], mode='L')

        # Extract region id from the file name
        region_id = os.path.basename(self.img_filenames[i]).split('_')[:2]
        region_id = '_'.join(region_id)
        # If there are multiple images for the same region, calculate the average
        if len(self.region_dict[region_id]) > 1:
            ref_imgs = self.region_dict[region_id]
            # Randomly select 1 to 5 images from the same region
            # num_samples = min(7, len(self.region_dict[region_id]))
            # sample_imgs = random.sample(ref_imgs, num_samples)

            # test use all the same region images
            sample_imgs = ref_imgs

            avg_img = np.zeros_like(image).astype(np.float32)  # 初始化平均图像数组
            count = 0
            for ref_img_path in sample_imgs:
                ref_img = cv2.imread(ref_img_path)
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                ref_img = ref_img.astype(np.float32)/255
                ref_img = np.transpose(ref_img, (2, 0, 1))  # 转换为float32
                avg_img += ref_img
                count += 1
            # 计算平均值并转换回uint8
            avg_img /= count
        else:
            avg_img = image
        result = dict(image=image, ref_image=avg_img, mask=mask[None, ...])
        return result


class Sentinel_InpaintingDataset(Dataset):
    def __init__(self, datadir, img_suffix='.tif'):
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.region_dict = self._create_region_dict()

    def __len__(self):
        print(len(self.mask_filenames))
        return len(self.mask_filenames)

    def _create_region_dict(self):
        region_dict = {}
        for file in self.img_filenames:
            region_id = os.path.basename(file).split('_')[:2]
            region_id = '_'.join(region_id)
            if region_id in region_dict:
                region_dict[region_id].append(file)
            else:
                region_dict[region_id] = [file]
        return region_dict

    def __getitem__(self, i):
        print(self.img_filenames[i])
        image = load_tif(self.img_filenames[i])
        mask = load_tif(self.mask_filenames[i])

        # Extract region id from the file name
        region_id = os.path.basename(self.img_filenames[i]).split('_')[:2]
        region_id = '_'.join(region_id)

        # If there are multiple images for the same region, calculate the average
        if len(self.region_dict[region_id]) > 1:
            ref_imgs = self.region_dict[region_id]
            # Randomly select 1 to 5 images from the same region
            # num_samples = min(7, len(self.region_dict[region_id]))
            # sample_imgs = random.sample(ref_imgs, num_samples)

            # test use all the same region images
            sample_imgs = ref_imgs

            avg_img = np.zeros_like(image).astype(np.float32)  # 初始化平均图像数组
            count = 0
            for ref_img_path in sample_imgs:
                ref_img = load_tif(ref_img_path)
                ref_img = ref_img.astype(np.float32)
                # ref_img = np.transpose(ref_img, (2, 0, 1))  # 转换为float32
                avg_img += ref_img
                count += 1
            # 计算平均值并转换回uint8
            avg_img /= count
        else:
            avg_img = image

        result = dict(image=image, ref_image=avg_img, mask=mask[None, ...])
        return result