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
        dl_path="data",
        class_names=None,
        batch_size=8,
        image_size=(224, 224),
        data_url="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    ):
        """CatDogDataModule.

        Args:
            dl_path: Root directory where to download the data.
            class_names: Names of the classes in the dataset.
            batch_size: Number of samples in a batch.
            image_size: Size to resize images.
            data_url: URL to download the dataset.
        """
        super().__init__()

        self.dl_path = dl_path
        self.batch_size = batch_size
        self.dataset_name = "cats_and_dogs_filtered"
        self.class_names = class_names or ["cat", "dog"]
        self.image_size = image_size
        self.data_path = Path(dl_path).joinpath(self.dataset_name)
        self.data_url = data_url

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def prepare_data(self):
        """Download images and prepare image datasets."""
        if not self.data_path.exists():
            download_and_extract_archive(url=self.data_url, download_root=self.dl_path, remove_finished=False)

    def setup(self, stage=None):
        """Setup the datasets and transformations."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"The dataset directory {self.data_path} does not exist.")

        full_dataset = datasets.ImageFolder(self.data_path.joinpath("train"), transform=self.transform)

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        self.test_dataset = datasets.ImageFolder(self.data_path.joinpath("validation"), transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )


###############################
####### MNIST DATASET #########
###############################


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        dl_path="data",
        class_names=None,
        batch_size=32,
        image_size=(28, 28),
        normalize_mean=[0.1307],
        normalize_std=[0.3081]
    ):
        """MNISTDataModule.

        Args:
            dl_path: Root directory where to download the data.
            class_names: Names of the classes in the dataset.
            batch_size: Number of samples in a batch.
            image_size: Size to resize images (MNIST is originally 28x28).
            normalize_mean: Normalization mean values.
            normalize_std: Normalization standard deviation values.
        """
        super().__init__()

        self.dl_path = dl_path
        self.class_names = class_names or [str(i) for i in range(10)]
        self.batch_size = batch_size
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Custom transform to repeat channels
        ])

    def prepare_data(self):
        """Download images and prepare image datasets."""
        datasets.MNIST(self.dl_path, train=True, download=True)
        datasets.MNIST(self.dl_path, train=False, download=True)

    def setup(self, stage=None):
        """Setup the datasets and transformations."""
        if not Path(self.dl_path).exists():
            raise FileNotFoundError(f"The dataset directory {self.dl_path} does not exist.")

        dataset = datasets.MNIST(self.dl_path, train=True, transform=self.transform)
        self.mnist_test = datasets.MNIST(self.dl_path, train=False, transform=self.transform)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.mnist_train, self.mnist_val = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False
        )



############################
######## VOCData ###########
############################

class VOCDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir="data",
        year="2007",
        image_set="train",
        batch_size=16,
        image_size=(224, 224),
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225]
    ):
        """VOCDataModule.

        Args:
            data_dir: Root directory where to download the data.
            year: The year of the dataset to be downloaded (2021 is the larger version).
            image_set: The image set to be used (train, val, test).
            batch_size: Number of samples in a batch.
            image_size: Size to resize images.
            normalize_mean: Normalization mean values.
            normalize_std: Normalization standard deviation values.
        """
        super().__init__()

        self.data_dir = data_dir
        self.year = year
        self.image_set = image_set
        self.batch_size = batch_size
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),
        ])

    def prepare_data(self):
        """Download images and prepare image datasets."""
        VOCDetection(root=self.data_dir, year=self.year, image_set=self.image_set, download=True)

    def setup(self, stage=None):
        """Setup the datasets and transformations."""
        self.voc_train = VOCDetection(root=self.data_dir, year=self.year, image_set="train", transform=self.transform)
        self.voc_val = VOCDetection(root=self.data_dir, year=self.year, image_set="val", transform=self.transform)

    def collate_fn(self, batch):
        images, targets = list(zip(*batch))
        class_to_index = self.get_class_to_idx()

        resized_images = []
        adjusted_targets = []

        for img, target in zip(images, targets):
            original_width, original_height = img.shape[2], img.shape[1]
            resized_img = resize(img, self.image_size)
            resized_images.append(resized_img)

            adjusted_boxes = []
            labels = []
            for obj in target['annotation']['object']:
                bbox = obj['bndbox']
                xmin = float(bbox['xmin']) * (self.image_size[0] / original_width)
                ymin = float(bbox['ymin']) * (self.image_size[1] / original_height)
                xmax = float(bbox['xmax']) * (self.image_size[0] / original_width)
                ymax = float(bbox['ymax']) * (self.image_size[1] / original_height)
                adjusted_boxes.append([xmin, ymin, xmax, ymax])

                class_name = obj['name']
                label_idx = class_to_index[class_name]
                labels.append(label_idx)

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
        return list(self.get_class_to_idx().keys())

    def train_dataloader(self):
        return DataLoader(
            self.voc_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.voc_val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.voc_val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )


#######################
#### JokesDataset #####
#######################

class JokesDataset(Dataset):
    def __init__(self, 
                 file_path, 
                 max_length=512):
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
    def __init__(self, 
                 tokenizer, 
                 max_length=512):
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
    def __init__(self, 
                 data_dir='data', 
                 batch_size=8, 
                 max_length=512, 
                 train_val_test_split=[0.8, 0.1, 0.1]):
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

