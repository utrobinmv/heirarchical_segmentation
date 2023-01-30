import os
from PIL import Image
import numpy as np

from base import BaseDataSet, BaseDataLoader
from utils import palette

class PascalDataset(BaseDataSet):
    """
    """
    def __init__(self, **kwargs):
        self.num_classes = 7
        self.palette = palette.get_voc_palette(self.num_classes)
        super(PascalDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in  ["train", "val"]:
            self.image_dir = os.path.join(self.root, 'JPEGImages')
            self.label_dir = os.path.join(self.root, 'gt_masks')
            
            with open(os.path.join(self.root, f'{self.split}_id.txt'), 'r', encoding="utf-8-sig") as f:
                self.files = f.read().splitlines()
                
            #self.files = [path for path in list_files]
        else: raise ValueError(f"Invalid split name {self.split}")
    
    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.npy')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.load(label_path) # from -1 to 149
        return image, label, image_id


class PascalDataloader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, mean=None, std=None, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = mean
        self.STD = std

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = PascalDataset(**kwargs)
        super(PascalDataloader, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
