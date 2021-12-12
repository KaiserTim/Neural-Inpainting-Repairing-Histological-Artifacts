import os
import numpy as np
import random
import torch
import h5py
import warnings
from torch.utils.data import Dataset
from demics import Tensor
from .ops import RandomRotateFlip


class TrainSetIntact(Dataset):
    """A PyTorch Dataset class. Inputs:
        length: The total number of examples to generate
        crop_dim: The square dimension of the generated crops
        test: Whether to return data from the test split as default
        n_test: How many cutouts to use as test data
    """
    def __init__(self, length, crop_dim, test=False, n_test=0):
        """"""
        
        self.test = test
        self.len = length
        self.crop_dim = crop_dim
        datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/")
        data_ids = ["B21_3037_LE06_28000_34000_4000_10000_Slice15.tif",
                    "B21_1959_LE06_30000_38000_36000_46000_Slice15.tif",
                    "B21_1959_LE06_83000_99000_50000_60000_Slice15.tif",
                    "B21_3037_LE06_68000_85000_7000_25000_Slice15.tif",
                    "B21_3037_LE06_74000_80000_65000_73000_Slice15.tif",
                    "B21_3614_LE06_5000_20000_25000_31000_Slice15.tif",
                    "B21_3870_Amunts_LE07_20180820_Slice15.tif"]
        
        self.imgs = []
        for data_id in data_ids:
            path = os.path.join(datapath, "example_volume/")+data_id
            self.imgs.append(Tensor.from_file(path, delimiter="#"))
            
        self.train_ids = np.arange(len(self.imgs))
        self.test_ids = np.arange(0)
        for i in range(n_test):
            self.test_ids = np.append(self.test_ids, self.train_ids[i])
            self.train_ids = np.delete(self.train_ids, i)
        
        anno_ids = os.listdir(os.path.join(datapath, "new_data"))
        
        self.annos = []
        for i, img_id in enumerate(anno_ids):
            with h5py.File(os.path.join(datapath, "new_data/")+img_id, 'r') as file:
                img = file["inputs"][:]
                shape = img.shape
                if shape[0] < crop_dim or shape[1] < crop_dim:
                    continue
                self.annos.append(file["foreground0"][:,:,3].astype("bool"))  
                
        self.transforms = RandomRotateFlip()
        
        print(f"Loaded {len(data_ids)} intact crops ({len(self.train_ids)} train, {len(self.test_ids)} test).")
        
    def __getitem__(self, item, coords=None, seed=None, test=None):
        """Returns a [crop_dim, crop_dim] crop and a mask from a random annotation."""
        if test is not None:  # Check test first and self.test second
            assert type(test) == bool, "test needs to be bool."
            ids = self.test_ids if test else self.train_ids
        else:
            ids = self.test_ids if self.test else self.train_ids
        
        if seed is not None:
            randint = np.random.randint(1000)
            np.random.seed(seed)
            random.seed(seed)
        
        idx = np.random.choice(ids)
        img = self.imgs[idx]
        img_height, img_width = img.shape[:2]
        
        if coords is not None:
            x, y = coords
            crop = img[x:x + self.crop_dim, y:y + self.crop_dim].numpy().astype("float32")
        else:
            while True:
                x = np.random.randint(img_height - self.crop_dim)
                y = np.random.randint(img_width - self.crop_dim)
                # crop = img.lazy_load((slice(x,x+self.crop_dim), slice(y,y+self.crop_dim))).numpy().astype("float32")
                crop = img[x:x + self.crop_dim, y:y + self.crop_dim].numpy().astype("float32")
                if crop.sum() < 170*self.crop_dim**2:
                    break
        
        crop /= 127.5
        crop -= 1
        
        # Select a random annotation as a mask
        while True:
            anno = random.choice(self.annos)
            shape = anno.shape            
            x = np.random.randint(shape[0] - self.crop_dim)
            y = np.random.randint(shape[1] - self.crop_dim)

            mask = anno[x:x + self.crop_dim, y:y + self.crop_dim]
            if mask.sum() > 1000 and mask.sum() < self.crop_dim**2/3:  # Ensure that some pixels are masked
                break   
            
        crop = torch.as_tensor(crop).unsqueeze(0)
        mask = torch.as_tensor(mask).unsqueeze(0)
        crop, mask = self.transforms([crop, mask])  # Data augmentation
        
        if seed is not None:
            np.random.seed(randint)
            random.seed(randint)
        
        return crop, mask 

    def __len__(self):
        return self.len


class TrainSetArtefacts(Dataset):
    """A PyTorch Dataset class. Inputs:
        length: The total number of examples to generate
        crop_dim: The square dimension of the generated crops
        test: Whether to return data from the test split as default
        n_test: How many cutouts to use as test data
    """
    def __init__(self, length, crop_dim, test=False, n_test=0):
        self.test = test
        
        datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/")
        self.data_ids = os.listdir(os.path.join(datapath, "new_data"))
        
        self.imgs = []
        self.annos = []
        for i, img_id in enumerate(self.data_ids):
            with h5py.File(os.path.join(datapath, "new_data/")+img_id, 'r') as file:
                img = file["inputs"][:]
                shape = img.shape
                if shape[0] < crop_dim or shape[1] < crop_dim:
                    warnings.warn(f"Image {img_id} is smaller than crop_dim in at least one dimension")
                    continue
                self.imgs.append(img[:,:,14])
                self.annos.append(file["foreground0"][:,:,3].astype("bool"))  
                
        self.train_ids = np.arange(len(self.imgs))
        self.test_ids = np.arange(0)
        for img_id in self.data_ids[:n_test]:
            idx = self.data_ids.index(img_id)
            self.test_ids = np.append(self.test_ids, self.train_ids[idx])
            self.train_ids = np.delete(self.train_ids, idx)
        
        self.len = length
        self.crop_dim = crop_dim
        
        self.transforms = RandomRotateFlip()
                                              
        print(f"{len(self.imgs)} images loaded into the dataset ({len(self.train_ids)} train, {len(self.test_ids)} test).")

    def __getitem__(self, item, test=False, seed=None):
        """Returns a [crop_dim, crop_dim] crop and an artifact target for that crop."""
        if seed is not None:
            randint = np.random.randint(1000)
            np.random.seed(seed)
            random.seed(seed)
        
        ids = self.test_ids if test or self.test else self.train_ids
        
        idx = np.random.choice(ids)
        img = self.imgs[idx]
        anno = self.annos[idx]
        shape = img.shape
        
        idx = np.random.choice(ids)
        img = self.imgs[idx]
        anno = self.annos[idx]
        shape = img.shape            
        x = np.random.randint(shape[0] - self.crop_dim)
        y = np.random.randint(shape[1] - self.crop_dim)

        anno_crop = anno[x:x + self.crop_dim, y:y + self.crop_dim]
#             if anno_crop_tmp.sum() > 0 and anno_crop_tmp.sum() < 0.7*self.crop_dim**2 and b:
        img_crop = img[x:x + self.crop_dim, y:y + self.crop_dim].astype("float32")
           
        img_crop /= 127.5
        img_crop -= 1
        
        img_crop = torch.from_numpy(img_crop).unsqueeze(0)
        anno_crop = torch.from_numpy(anno_crop).unsqueeze(0)
        img_crop, anno_crop = self.transforms([img_crop, anno_crop])  # Data augmentation
        
        if seed is not None:
            np.random.seed(randint)
            random.seed(randint)

        return img_crop, anno_crop
    
    def __len__(self):
        return self.len
