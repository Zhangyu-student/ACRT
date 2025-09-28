import glob
import logging
import os
import random

import albumentations as A
import cv2
import numpy as np
from omegaconf import open_dict, OmegaConf
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import rasterio
from saicinpainting.evaluation.data import InpaintingDataset as InpaintingEvaluationDataset
from saicinpainting.evaluation.data import Sentinel_InpaintingDataset as Sentinel_InpaintingEvaluationDataset
from saicinpainting.training.data.masks import get_mask_generator

LOGGER = logging.getLogger(__name__)


#  训练集Training Dataloader
class InpaintingTrainDataset(Dataset):
    def __init__(self, indir, mask_generator, transform):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.png'), recursive=True))
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0
        self.region_dict = self._create_region_dict()

    def __len__(self):
        return len(self.in_files)

    def _create_region_dict(self):
        region_dict = {}
        for file in self.in_files:
            region_id = os.path.basename(file).split('_')[:2]
            region_id = '_'.join(region_id)
            if region_id in region_dict:
                region_dict[region_id].append(file)
            else:
                region_dict[region_id] = [file]
        return region_dict

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))  # 转换为float32以避免溢出

        # Extract region id from the file name
        region_id = os.path.basename(path).split('_')[:2]
        region_id = '_'.join(region_id)

        # If there are multiple images for the same region, calculate the average
        if len(self.region_dict[region_id]) > 1:
            ref_imgs = self.region_dict[region_id]
            # Randomly select 1 to 5 images from the same region
            num_samples = random.randint(1, min(5, len(ref_imgs)))
            sample_imgs = random.sample(ref_imgs, num_samples)

            avg_img = np.zeros_like(img).astype(np.float32)  # 初始化平均图像数组
            count = 0
            for ref_img_path in sample_imgs:
                ref_img = cv2.imread(ref_img_path)
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                ref_img = self.transform(image=ref_img)['image']
                ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)  # 转换为float32
                ref_mask = self.mask_generator(ref_img)
                ref_img = np.multiply(ref_img, 1 - ref_mask)
                # ref_img_bgr = np.transpose(ref_img, (1, 2, 0)).astype(np.float32)
                # cv2.imwrite(f"ref_img_bgr_{item}.png", ref_img_bgr * 255)
                avg_img += ref_img
                count += 1
            avg_img /= count
        else:
            num_samples = 0
            avg_img = img
        ref_img_bgr = np.transpose(avg_img, (1, 2, 0)).astype(np.float32)
        # cv2.imwrite(f"ref_img_bgr_{item}.png", ref_img_bgr * 255)
        # print(f"ref_img_bgr_{item}.png has been written down!")
        # avg_img_bgr = np.transpose(avg_img, (1, 2, 0)).astype(np.float32)
        # cv2.imwrite(f"ref_img_transformed_{item}.png", avg_img_bgr * 255)
        # mask的维度(1,512,512)
        mask = self.mask_generator(img)  # Use the average image to generate the mask
        return dict(image=img,
                    ref_image=avg_img,
                    mask=mask)


class SentinelTrainDataset(Dataset):
    def __init__(self, indir, mask_generator, transform):
        # 获取目录下所有.tif文件
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.tif'), recursive=True))
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0
        self.region_dict = self._create_region_dict()

    def __len__(self):
        return len(self.in_files)

    def _create_region_dict(self):
        region_dict = {}
        for file in self.in_files:
            region_id = os.path.basename(file).split('_')[:2]
            region_id = '_'.join(region_id)
            if region_id in region_dict:
                region_dict[region_id].append(file)
            else:
                region_dict[region_id] = [file]
        return region_dict

    def __getitem__(self, item):
        path = self.in_files[item]

        # 使用 rasterio 读取四波段的TIF图像
        with rasterio.open(path) as dataset:
            img = dataset.read([1, 2, 3, 4])  # 读取红、绿、蓝、第四波段
        # 应用图像预处理
        img = self.transform(image=img)['image']/10000

        # 提取区域ID（从文件名获取）
        region_id = os.path.basename(path).split('_')[:2]
        region_id = '_'.join(region_id)

        # 如果一个区域有多个图像，计算这些图像的平均值
        if len(self.region_dict[region_id]) > 1:
            ref_imgs = self.region_dict[region_id]
            num_samples = random.randint(1, min(5, len(ref_imgs)))
            sample_imgs = random.sample(ref_imgs, num_samples)

            avg_img = np.zeros_like(img).astype(np.float32)  # 初始化平均图像数组
            count = 0
            for ref_img_path in sample_imgs:
                with rasterio.open(ref_img_path) as ref_dataset:
                    ref_img = ref_dataset.read([1, 2, 3, 4])/10000  # 读取四波段图像

                ref_img = self.transform(image=ref_img)['image']
                ref_mask = self.mask_generator(ref_img)
                ref_img = np.multiply(ref_img, 1 - ref_mask)
                avg_img += ref_img
                count += 1
            avg_img /= count
        else:
            num_samples = 0
            avg_img = img

        ref_img_bgr = np.transpose(avg_img, (1, 2, 0)).astype(np.float32)

        # 使用平均图像生成掩码
        mask = self.mask_generator(img)  # 生成掩码

        return dict(image=img,
                    ref_image=avg_img,
                    mask=mask)


def get_transforms(transform_variant, out_size):
    if transform_variant == 'default':
        transform = A.Compose([
            A.RandomScale(scale_limit=0.2),  # +/- 20%
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'no_augs':
        transform = A.Compose([
            A.ToFloat()
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform


def make_default_train_dataloader(indir, kind='Landsat', out_size=512, mask_gen_kwargs=None, transform_variant='default',
                                  mask_generator_kind="mixed", dataloader_kwargs=None, ddp_kwargs=None, **kwargs):
    LOGGER.info(f'Make train dataloader {kind} from {indir}. Using mask generator={mask_generator_kind}')

    mask_generator = get_mask_generator(kind=mask_generator_kind, kwargs=mask_gen_kwargs)
    transform = get_transforms(transform_variant, out_size)

    if kind == 'Landsat':
        logging.info("train")
        dataset = InpaintingTrainDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         **kwargs)
    elif kind == 'Sentinel':
        logging.info("train")
        dataset = SentinelTrainDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         **kwargs)
    else:
        raise ValueError(f'Unknown train dataset kind {kind}')

    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    is_dataset_only_iterable = kind in ('default_web',)

    if ddp_kwargs is not None and not is_dataset_only_iterable:
        dataloader_kwargs['shuffle'] = False
        dataloader_kwargs['sampler'] = DistributedSampler(dataset, **ddp_kwargs)

    if is_dataset_only_iterable and 'shuffle' in dataloader_kwargs:
        with open_dict(dataloader_kwargs):
            del dataloader_kwargs['shuffle']

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_default_val_dataset(indir, kind='Landsat', **kwargs):

    LOGGER.info(f'Make val dataloader {kind} from {indir}')
    if kind == 'Landsat':
        dataset = InpaintingEvaluationDataset(indir, **kwargs)
    elif kind == 'Sentinel':
        dataset = Sentinel_InpaintingEvaluationDataset(indir, **kwargs)
    else:
        raise ValueError(f'Unknown val dataset kind {kind}')

    return dataset


def make_default_val_dataloader(*args, dataloader_kwargs=None, **kwargs):
    dataset = make_default_val_dataset(*args, **kwargs)

    if dataloader_kwargs is None:
        dataloader_kwargs = {}
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader



