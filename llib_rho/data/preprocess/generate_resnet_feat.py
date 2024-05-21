import numpy as np
import torch
import torchvision
import cv2
import os
import warnings
from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image
import multiprocessing
import random
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# Set a few constants related to data loading.
BATCH_SIZE = 128
NUM_WORKERS = multiprocessing.cpu_count()//2
PATH = '/home/ubuntu'

if torch.cuda.is_available():
	DEVICE = torch.device("cuda")
else:
	DEVICE = torch.device("cpu")

def get_data(dataset_name:str='behave'):

	train_dataset = CVPR_Dataset(
		PATH, "train",
		download=False,
		base_folder=dataset_name
	)

	val_dataset = CVPR_Dataset(
		PATH, "val",
		download=False,
		base_folder=dataset_name
	)

	test_dataset = CVPR_Dataset(
		PATH, "test",
		download=False,
		base_folder=dataset_name
	)
	
	return train_dataset, val_dataset, test_dataset

class CVPR_Dataset(Dataset):
		
	def __init__(
		self, 
		root: str,
		split: str = 'train', 
		download: bool = False,
		base_folder: str = 'behave',
		) -> None:
		assert split in ['train', 'val', 'test']
		
		self.root = root
		self.split = split
		self.base_folder = base_folder
		self.dataset_dir = os.path.join(self.root, self.base_folder)         
		self.resolution = [224, 224]

		self.all_list = self.parse_behave()
		self.shuffle()

		## SMALL SUBSET for Debugging
		# self.all_list = self.all_list[:100]

		print("Initialized BEHAVE Dataset!")

	def parse_behave(self,):
		""" Parse the BEHAVE dataset"""
		data_dir = os.path.join(self.dataset_dir, self.split)
		rgb_path = os.path.join(data_dir, "cropped_images")
   
		rgb_list = os.listdir(rgb_path)
		rgb_list.sort()

		# Make all list
		all_list=[]
		print("READING DATA.................")
		for rgb_file in tqdm(rgb_list):
			obj_sample={}
			obj_sample['rgb_path'] = os.path.join(rgb_path, rgb_file)
			all_list.append(obj_sample)

		return all_list
			
	def __len__(self):
		return len(self.all_list)

	def preprocess_image(self, img_cropped: np.ndarray):
		""" Preprocess image for ResNet18.
			1. Apply transforms to cropped image
		 
		 Returns torch.Tensor of shape (3, 224, 224)"""
		
		INPUT_IMG_SIZE = 224

		assert img_cropped.shape == (INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3), f"Image shape is {img_cropped.shape}"

		# torchvision transforms
		transforms = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

		img_tensor = transforms(img_cropped)
		return img_tensor
	
	def __getitem__(self, idx):
 
		obj_sample = self.all_list[idx]
		rgb_path = obj_sample['rgb_path']
	
		# Read images
		rgb = Image.open(rgb_path)
		rgb = np.array(rgb)
		img_tensor = self.preprocess_image(rgb)

		#output
		data_dict = {}
		data_dict['rgb'] = img_tensor
		data_dict['data_idx'] = torch.tensor(int(idx))

		return data_dict

	def shuffle(self):
		random.shuffle(self.all_list)

def generate_resnet_feat(args, save_dir_name:str='resnet_feat'):
	""" Generate ResNet18 features for the images in the dataset."""

	train_dataset, val_dataset, test_dataset = get_data(args.dataset_name)
	datasets = {
		'train': train_dataset,
		'val': val_dataset,
		'test': test_dataset
	}

	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
	val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

	resnet18 = torchvision.models.resnet18(pretrained=True)
	resnet18 = resnet18.to(DEVICE)
	resnet18.eval()
	feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
	

	for loader, split in zip([train_loader, val_loader, test_loader], ['train', 'val', 'test']):
		data_list = datasets[split].all_list
		print(f"Generating ResNet18 features for {split} set....")
		for idx, data_dict in tqdm(enumerate(loader)):
			rgb = data_dict['rgb'].to(DEVICE)
			rgb_paths=[data_list[int(i)]['rgb_path'] for i in data_dict['data_idx']]
			fileIDs = [os.path.basename(rgb_path).split('.')[0] for rgb_path in rgb_paths]

			with torch.no_grad():
				feat = feature_extractor(rgb)
				feat = feat.cpu().numpy()

			for i in range(feat.shape[0]):
				# Save in pickle format
				save_path = os.path.join(args.data_dir, split, save_dir_name, fileIDs[i] + '.pkl')
				with open(save_path, 'wb') as f:
					pickle.dump(feat[i].reshape(512,), f)

if __name__ == '__main__':

	# parse arguments
	parser = ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='~/behave/', help='Path to the  dataset directory containing train/ val/ test/')
	parser.add_argument('--dataset_name', type=str, default='behave', help='Dataset name', choices=['behave', 'intercap'])
	args = parser.parse_args()
	assert os.path.exists(args.data_dir), f"{args.data_dir} does not exist."

	# Generate save dirs
	for split in ['train', 'val', 'test']:
		for save_dir_name in ['resnet_feat']:
			save_dir = os.path.join(args.data_dir, split, save_dir_name)
			os.makedirs(save_dir, exist_ok=True)
	
	# Generate ResNet18 features
	generate_resnet_feat(args, save_dir_name='resnet_feat')