import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import List
from dataclasses import field
from .datasets import *
from .utils import Augmentation, Processing

# Steps to add a new dataset:
# 1) add name and composition to train_names/train_composition/val_names
# 2) in list each dataset, add name: Name = Name()
# 3) add dataset in datasets.py

@dataclass 
class Datasets:

    # image processing
    processing: Processing = Processing()

    # training data
    train_names: List[str] = field(default_factory=lambda: ['behave'])
    train_composition: List[float] = field(default_factory=lambda: [1.0])
    augmentation: Augmentation = Augmentation(
        use=True, mirror=0.5, noise=0.4, rotation=30, scale=0.25
    )

    # validation data
    val_names: List[str] = field(default_factory=lambda: ['behave'])
    
    # test data 
    test_names: List[str] = field(default_factory=lambda: [])
    
    # list all datasets here
    behave: Behave = Behave()


conf = OmegaConf.structured(Datasets)