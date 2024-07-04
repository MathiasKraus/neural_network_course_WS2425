import os
from pathlib import Path
from PIL import Image
from typing import Union, Optional
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from lightning.pytorch import LightningDataModule
from lightning.pytorch.demos.mnist_datamodule import MNIST
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder, VOCDetection
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms.functional import resize
from transformers import GPT2Tokenizer

import warnings
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

import logging
log = logging.getLogger(__name__)

###############################
#### CATS vs. DOGS DATASET ####
###############################

class CatDogDataModule(LightningDataModule):
    def __init__(
        self,
        dl_path: Union[str, Path] = "data",
        class_names: list = ["cat", "dog"],
        batch_size: int = 8,
        image_size: tuple = (224, 224),
    ):
        """GeneralImageDataModule.

        Args:
            dataset_name: name of the dataset folder
            dl_path: root directory where to download the data
            class_names: Names of the classes in the dataset
            batch_size: number of samples in a batch
            image_size: size to resize images
        """
        super().__init__()

        self._dl_path = dl_path
        self._batch_size = batch_size
        self.dataset_name = "cats_and_dogs_filtered"
        self.class_names = class_names
        self.image_size = image_size
        self.data_path = Path(dl_path).joinpath(self.dataset_name)
        self.DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"

        self.prepare_data()
        self.setup()

    def prepare_data(self):
        """Download images and prepare image datasets."""
        if self.DATA_URL:
            download_and_extract_archive(url=self.DATA_URL, download_root=self._dl_path, remove_finished=False)

    def setup(self, *args, **kwargs):
        pass

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = ImageFolder(self.data_path.joinpath("train"), transform)
        return DataLoader(dataset=dataset, batch_size=self._batch_size, shuffle=True)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = ImageFolder(self.data_path.joinpath("validation"), transform)
        return DataLoader(dataset=dataset, batch_size=self._batch_size, shuffle=False)


###############################
####### MNIST DATASET #########
###############################


class MNISTDataModule(LightningDataModule):
    def __init__(self, 
                 dl_path: Union[str, Path] = "data",
                 class_names: list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                 batch_size: int = 32):
        super().__init__()
        
        self.dl_path = dl_path
        self.class_names = class_names
        self.batch_size = batch_size

        self.prepare_data()
        self.setup()

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        # Transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Custom transform to repeat channels
        ])
        
        # Datasets
        dataset = datasets.MNIST(self.dl_path, train=True, download=True, transform=transform)
        self.mnist_test = datasets.MNIST(self.dl_path, train=False, download=True, transform=transform)
        self.mnist_train, self.mnist_val = random_split(
            dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


############################
######## VOCData ###########
############################

class VOCDataModule(LightningDataModule):
    def __init__(self, data_dir='data', year='2012', image_set='train', batch_size=16, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.year = year
        self.image_set = image_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.prepare_data()
        self.setup()

    def collate_fn(self, batch):
        images, targets = list(zip(*batch))
        class_to_index = self.get_class_to_idx()
        
        # Resize all images to a fixed size (e.g., 224x224)
        target_size = (224, 224)
        resized_images = []
        adjusted_targets = []
        
        for img, target in zip(images, targets):
            original_width, original_height = img.shape[2], img.shape[1]
            resized_img = resize(img, target_size)
            resized_images.append(resized_img)
            
            # Adjust bounding boxes
            adjusted_boxes = []
            labels = []
            for obj in target['annotation']['object']:
                bbox = obj['bndbox']
                xmin = float(bbox['xmin']) * (target_size[0] / original_width)
                ymin = float(bbox['ymin']) * (target_size[1] / original_height)
                xmax = float(bbox['xmax']) * (target_size[0] / original_width)
                ymax = float(bbox['ymax']) * (target_size[1] / original_height)
                adjusted_boxes.append([xmin, ymin, xmax, ymax])

                # Get the correct label index
                class_name = obj['name']
                label_idx = class_to_index[class_name]
                labels.append(label_idx)
            
            # Convert to tensor
            adjusted_boxes = torch.tensor(adjusted_boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            
            adjusted_targets.append({'boxes': adjusted_boxes, 'labels': labels})
        
        images = torch.stack(resized_images)
        
        return images, adjusted_targets

    
    def get_class_to_idx(self):
        return {
            'background': 0, 'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 
            'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11, 
            'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16, 
            'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20
        }

    def get_idx_to_class(self):
        class_to_idx = self.get_class_to_idx()
        return dict(zip(class_to_idx.values(), class_to_idx.keys()))

    def get_classes(self):
        return self.get_class_to_idx().keys()
    
    def prepare_data(self):
        VOCDetection(root=self.data_dir, year=self.year, image_set=self.image_set, download=True)

    def setup(self, stage=None):
        self.voc_train = VOCDetection(root=self.data_dir, year=self.year, image_set='train', transform=self.transform)
        self.voc_val = VOCDetection(root=self.data_dir, year=self.year, image_set='val', transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.voc_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.voc_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.voc_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)


#######################
#### JokesDataset #####
#######################

class JokesDataset(Dataset):
    def __init__(self, file_path, max_length=512):
        self.jokes = pd.read_csv(file_path)['Joke'].tolist()
        self.max_length = max_length
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Add padding token

    def __len__(self):
        return len(self.jokes)
    
    def __getitem__(self, idx):
        joke = self.jokes[idx]
        encoding = self.tokenizer(joke, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        input_ids = encoding['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)  # Remove batch dimension
        return input_ids, attention_mask

class TextCollate:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = [item[0] for item in batch]
        attention_masks = [item[1] for item in batch]
        
        # Pad sequences to the same length
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        
        return input_ids, attention_masks

class JokesDataModule(LightningDataModule):
    def __init__(self, data_dir='data', batch_size=8, max_length=512, train_val_test_split=[0.8, 0.1, 0.1]):
        super().__init__()
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, 'shortjokes.csv')
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_val_test_split = train_val_test_split
        
        os.makedirs(data_dir, exist_ok=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Add padding token

        self.prepare_data()
        self.setup()

    def prepare_data(self):
        # Ensure the data file exists
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"{self.data_file} not found. Please download the dataset from Kaggle and place it in the data directory.")

    def setup(self, stage=None):
        dataset = JokesDataset(file_path=self.data_file, max_length=self.max_length)
        train_size = int(self.train_val_test_split[0] * len(dataset))
        val_size = int(self.train_val_test_split[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])
        self.collate_fn = TextCollate(self.tokenizer, max_length=self.max_length)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

