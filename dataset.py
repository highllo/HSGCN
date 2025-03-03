import libtiff
#libtiff.libtiff_ctypes.suppress_warnings()
from libtiff import TIFF
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
import torchvision.datasets as dset
import torchvision.datasets.folder as folder
from PIL import Image
import torch
from torch.utils import data as tudata
from conf import settings

def toNomr(image,isNum:bool, im_max, im_min = 0):
    if(image.max() == image.min()):
        im = image/im_max
    elif(isNum):
        im = image/im_max
    else:
        im = (image-im_min)/(im_max-im_min)
    im[im > 1] = 1
    im[im < 0] = 0
    return im

def SHSC_loader(path: str, bands) -> Any:
    if(bands=='rgb'):
        bands_range = settings.SHSC_RGB_BANDS
    elif(bands=='tif'):
        bands_range = settings.SHSC_TIFF_BANDS
    else:
        bands = bands.split(',')
        bands_range = (int(bands[0]),int(bands[1]),int(bands[2]))
    #print(path)
    tif = TIFF.open(path, mode='r')
    im = tif.read_image().astype('float32')
    imshape = im.shape
    re_im = np.zeros([imshape[1], imshape[2], len(bands_range)]).astype('float32')
    for i in range(len(bands_range)):
        re_im[:,:,i] = toNomr(im[bands_range[i],:,:], True, 10000)
    return re_im.astype('float32')

def OHS_loader(path: str, bands) -> Any:
    if(bands=='rgb'):
        bands_range = settings.SHSC_RGB_BANDS
    elif(bands=='tif'):
        bands_range = settings.SHSC_TIFF_BANDS
    else:
        bands = bands.split(',')
        bands_range = (int(bands[0]),int(bands[1]),int(bands[2]))
    #print(path)
    tif = TIFF.open(path, mode='r')
    im = tif.read_image().astype('float32')
    imshape = im.shape
    re_im = np.zeros([imshape[0], imshape[1], len(bands_range)]).astype('float32')
    for i in range(len(bands_range)):
        re_im[:,:,i] = toNomr(im[:,:,bands_range[i]], True, 10000)
    return re_im.astype('float32')

def eurosat_loader(path: str, bands) -> Any:
    if(bands=='rgb'):
        bands_range = settings.EUROSAT_RGB_BANDS
    elif(bands=='tif'):
        bands_range = settings.EUROSAT_TIFF_BANDS
    else:
        bands = bands.split(',')
        bands_range = (int(bands[0]),int(bands[1]),int(bands[2]))
    #print(path)
    tif = TIFF.open(path, mode='r')
    im = tif.read_image().astype('float32')
    imshape = im.shape
    re_im = np.zeros([imshape[0], imshape[1], len(bands_range)]).astype('float32')
    for i in range(len(bands_range)):
        re_im[:,:,i] = toNomr(im[:,:,bands_range[i]], True, 10000)
    return re_im.astype('float32')


def nasc_tg2_loader(path: str, bands) -> Any:
    if(bands=='rgb'):
        bands_range = settings.NASC_TG2_RGB_BANDS
    elif(bands=='tif'):
        bands_range = settings.NASC_TG2_TIFF_BANDS
    else:
        bands = bands.split(',')
        bands_range = (int(bands[0]),int(bands[1]),int(bands[2]))
    #print(path)
    tif = TIFF.open(path, mode='r')
    im = tif.read_image().astype('float32')
    imshape = im.shape
    re_im = np.zeros([imshape[0], imshape[1], len(bands_range)]).astype('float32')
    for i in range(len(bands_range)):
        re_im[:,:,i] = toNomr(im[:,:,bands_range[i]], True, 10000)
    return re_im.astype('float32')

class MyImageFolder(dset.VisionDataset):

    def __init__(self,root: str,setname: str,type: str,
        loader: Callable[[str], Any] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, settings.IMG_EXTENSIONS, is_valid_file)
        self.type = type
        # 针对不同数据集设置不同的数据加载方法
        if(setname == 'shsc'):
            self.loader = SHSC_loader
        elif(setname == 'eurosat'):
            self.loader = eurosat_loader
        elif(setname == 'nasc_tg2'):
            self.loader = nasc_tg2_loader
        elif(setname == 'ohs'):
            self.loader = OHS_loader

        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    # 读取特定文件夹内的数据集
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return folder.make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
    # 获取数据集的类名、数量
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    # 由模型调用，返回图像与标签
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        if self.type == 'rgb':
            sample = self.loader(path, self.type)
        else:
            sample = self.loader(path, self.type)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

def split_datasets(dataset: tudata.Dataset,frac: float, seed = 345235):
    train_size = int(frac*dataset.__len__())
    print(train_size)
    test_size = dataset.__len__()-train_size
    return tudata.random_split(dataset=dataset,lengths=[train_size, test_size])










    


