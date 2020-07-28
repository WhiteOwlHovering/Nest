import os
from collections import OrderedDict
from typing import Tuple, List, Dict, Union, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from nest import register


@register
def image_transform(
    image_size: Union[int, List[int]],
    augmentation: dict,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]) -> Callable:
    """Image transforms.
    """

    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    else:
        image_size = tuple(image_size)

    # data augmentations
    horizontal_flip = augmentation.pop('horizontal_flip', None)
    if horizontal_flip is not None:
        assert isinstance(horizontal_flip, float) and 0 <= horizontal_flip <= 1

    vertical_flip = augmentation.pop('vertical_flip', None)
    if vertical_flip is not None:
        assert isinstance(vertical_flip, float) and 0 <= vertical_flip <= 1

    random_crop = augmentation.pop('random_crop', None)
    if random_crop is not None:
        assert isinstance(random_crop, dict)

    center_crop = augmentation.pop('center_crop', None)
    if center_crop is not None:
        assert isinstance(center_crop, (int, list))

    if len(augmentation) > 0:
        raise NotImplementedError('Invalid augmentation options: %s.' % ', '.join(augmentation.keys()))
    
    t = [
        transforms.Resize(image_size) if random_crop is None else transforms.RandomResizedCrop(image_size[0], **random_crop),
        transforms.CenterCrop(center_crop) if center_crop is not None else None,
        transforms.RandomHorizontalFlip(horizontal_flip) if horizontal_flip is not None else None,
        transforms.RandomVerticalFlip(vertical_flip) if vertical_flip is not None else None,
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]
    
    return transforms.Compose([v for v in t if v is not None])


@register
def fetch_data(
    dataset: Callable[[str], Dataset],
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    train_splits: List[str] = [],
    test_splits: List[str] = [],
    train_shuffle: bool = True,
    test_shuffle: bool = False,
    train_augmentation: dict = {},
    test_augmentation: dict = {},
    batch_size: int = 1,
    test_batch_size: Optional[int] = None) -> Tuple[List[Tuple[str, DataLoader]], List[Tuple[str, DataLoader]]]:
    """Return data loader list.
    """

    # fetch training data
    train_transform = transform(augmentation=train_augmentation) if transform else None
    train_loader_list = []
    for split in train_splits:
        train_loader_list.append((split, DataLoader(
            dataset = dataset(
                split = split, 
                transform = train_transform,
                target_transform = target_transform),
            batch_size = batch_size,
            num_workers = num_workers,
            pin_memory = pin_memory,
            drop_last=drop_last,
            shuffle = train_shuffle)))
    
    # fetch testing data
    test_transform = transform(augmentation=test_augmentation) if transform else None
    test_loader_list = []
    for split in test_splits:
        test_loader_list.append((split, DataLoader(
            dataset = dataset(
                split = split, 
                transform = test_transform,
                target_transform = target_transform),
            batch_size = batch_size if test_batch_size is None else test_batch_size,
            num_workers = num_workers,
            pin_memory = pin_memory,
            drop_last=drop_last,
            shuffle = test_shuffle)))

    return train_loader_list, test_loader_list


@register
def mnist(
    split: str,
    data_dir: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = True) -> Dataset:
    """MNIST dataset.
    """

    if split == 'train':
        return datasets.MNIST(data_dir, True, transform, target_transform, download)
    elif split == 'test':
        return datasets.MNIST(data_dir, False, transform, target_transform, download)
    else:
        raise NotImplementedError('Invalid "%s" split for MNIST dataset.' % split)


@register
def pascal_voc_object_categories(query: Optional[Union[int, str]] = None) -> Union[int, str, List[str]]:
    """PASCAL VOC dataset class names.
    """

    categories = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor']
        
    if query is None:
        return categories
    else:
        for idx, val in enumerate(categories):
            if isinstance(query, int) and idx == query:
                return val
            elif val == query:
                return idx


class VOC_Classification(Dataset):
    """Dataset for PASCAL VOC classification.
    """

    def __init__(self, data_dir, dataset, split, classes, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.dataset = dataset
        self.split = split
        self.image_dir = os.path.join(data_dir, dataset, 'JPEGImages')
        assert os.path.isdir(self.image_dir), 'Could not find image folder "%s".' % self.image_dir
        self.gt_path = os.path.join(self.data_dir, self.dataset, 'ImageSets', 'Main')
        assert os.path.isdir(self.gt_path), 'Could not find ground truth folder "%s".' % self.gt_path
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
        self.image_labels = self._read_annotations(self.split)

    def _read_annotations(self, split):
        class_labels = OrderedDict()
        num_classes = len(self.classes)
        if os.path.exists(os.path.join(self.gt_path, split + '.txt')):
            for class_idx in range(num_classes):
                filename = os.path.join(
                    self.gt_path, self.classes[class_idx] + '_' + split + '.txt')
                with open(filename, 'r') as f:
                    for line in f:
                        name, label = line.split()
                        if name not in class_labels:
                            class_labels[name] = np.zeros(num_classes)
                        class_labels[name][class_idx] = int(label)
        else:
            raise NotImplementedError(
                'Invalid "%s" split for PASCAL %s classification task.' % (split, self.dataset))

        return list(class_labels.items())

    def __getitem__(self, index):
        filename, target = self.image_labels[index]
        target = torch.from_numpy(target).float()
        img = Image.open(os.path.join(
            self.image_dir, filename + '.jpg')).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.image_labels)
    

@register
def pascal_voc_classification(
    split: str,
    data_dir: str,
    year: int = 2007,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None) -> Dataset:
    """PASCAL VOC dataset.
    """

    object_categories = pascal_voc_object_categories()
    dataset = 'VOC' + str(year)
    return VOC_Classification(data_dir, dataset, split, object_categories, transform, target_transform)


class Virtual_OR_Classification(Dataset):
    """Dataset for COCO
    """

    def __init__(self, data_dir,split, year, dataloader_flag, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.split = split
        ## data_dir: /media/rao/Data/Datasets/MSCOCO/coco/
        self.image_dir = os.path.join(data_dir,'images')
        assert os.path.isdir(self.image_dir), 'Could not find image folder "%s".' % self.image_dir
        self.gt_path = os.path.join(self.data_dir, 'annotations')
        assert os.path.isdir(self.gt_path), 'Could not find ground truth folder "%s".' % self.gt_path
        self.transform = transform
        self.target_transform = target_transform
        if dataloader_flag=='counting':
            ## use coco 2017 train data
            self.image_labels = self._read_annotations(split,year)
        else:
            print('error, dataloader_flag should be counting')
        if split=='val':
            index=int(len(self.image_labels)/2)
            self.image_labels=self.image_labels[:index]
        print(len(self.image_labels))

    def _read_annotations(self,split,year):
        gt_file=os.path.join(self.gt_path,'virtual_OR_'+split+'.json')
        cocoGt=COCO(gt_file)
        catids=cocoGt.getCatIds()
        num_classes=len(catids)
        catid2index={}
        for i,cid in enumerate(catids):
            catid2index[cid]=i
        annids=cocoGt.getAnnIds()
        class_labels = OrderedDict()
        for id in annids:
            anns=cocoGt.loadAnns(id)
            for i in range(len(anns)):
                ann=anns[i]
                name=ann['image_id']
                if name not in class_labels:
                    class_labels[name]=np.zeros(num_classes)
                category_id=ann['category_id']
                class_labels[name][catid2index[category_id]]+=1
        return list(class_labels.items())            

    def __getitem__(self, index):
        filename, target = self.image_labels[index]
        target0=target
        target1=np.array([1])
        target0 = torch.from_numpy(target0).float()
        # print(type(1*target1))
        target1 = torch.from_numpy(1*target1).float()
        # target = torch.from_numpy(target).float()
        # 000000291625.jpg
        filename='0'*(12-len(str(filename)))+str(filename) #os.path.join(
        img = Image.open( self.image_dir+'/'+ filename + '.png').convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target0,target1

    def __len__(self):
        return len(self.image_labels)

@register
def virtual_OR_classification(
    split: str,
    data_dir: str,
    year: int = 2019,
    dataloader_flag: str = 'counting',
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None) -> Dataset:
    """ Virtual OR dataset
    """

    return Virtual_OR_Classification(data_dir, split,year,dataloader_flag, transform, target_transform)


