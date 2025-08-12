import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

datainfo_path = "dataset_info.json"

# transform = 


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    elif ext in ['.png', '.gif']:
        return Image.open(filename).convert("L")
    else:
        assert ext == '.jpg', "We are loading a jpg file"
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))

    # only have one label: 255 means 1
    mask = mask / 255
    # print(mask_file)
    # print(len(np.unique(mask)))
    # exit()

    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, p_aug: float = 0.1, mask_suffix: str = '', split: str = 'val'): 
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.split = split
        self.p = p_aug

        self.transform = A.Compose([
            # A.RandomCrop(width=256, height=256), # no need to random crop in the spider dataset
            # A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=self.p),
            A.RandomBrightnessContrast(p=self.p),
            A.ElasticTransform(p=self.p),
            A.GaussianBlur(p=self.p),
            A.GaussNoise(p=self.p),
            A.ColorJitter(p=self.p),
            # A.Normalize(mean=(0.5,), std=(0.5,)),  # for grayscale ? ######## solution, after normalize, why it become black?
            # ToTensorV2()
        ])

        # self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]

        with open(datainfo_path, "r") as f:
            dataset_info = json.load(f)
        
        self.ids = dataset_info[f"{self.split}_filenames"] 
        

        if not self.ids:
            # raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
            raise RuntimeError(f'No file name found in {datainfo_path}, make sure you put your images names there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)

            # only have one label: 255 means 1
            img = img / 255
            # print(np.unique(img))

            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = sorted(list(self.mask_dir.glob(name + self.mask_suffix + '.*'))) # sort the dataset
        img_file = sorted(list(self.images_dir.glob(name + '.*')))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        # Apply transformations (augmentation + preprocessing)
        if self.split == 'train':
            # print("img.size(), mask.size():",img.shape, mask.shape)
            img = np.squeeze(img, axis=0).astype(np.float32)
            mask = mask.astype(np.float32)

            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            img = np.expand_dims(img, axis=0)
            # print("img.size(), mask.size():",img.shape, mask.shape)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix ='_mask')

class PerfusionDataset(BasicDataset):
    def __init__(self, images_dir: str, mask_dir: str, scale=1, p_aug = 0.1, mask_suffix = '', split = 'val'):
        super().__init__(images_dir, mask_dir, scale, p_aug, mask_suffix, split)