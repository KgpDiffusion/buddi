import numpy as np
import torch
import cv2
import os
import warnings
from tqdm import tqdm
from argparse import ArgumentParser

def preprocess_image(img: np.ndarray, mask:np.ndarray, INFLATION_RATIO:float=1.25):
    """ Preprocess image for ResNet18.
        1. Fit a rectangle around the mask
        2. Crop image around rectangle with some scale and inflation
        3. Resize image to 224x224
        
        Returns torch.Tensor of shape (3, 224, 224)"""
    
    INPUT_IMG_SIZE = 224

    mask = mask.astype(np.uint8)
    x,y,w,h = cv2.boundingRect(mask)
    x_c, y_c = x + w//2, y + h//2

    dim = max(w,h)

    if(dim < INPUT_IMG_SIZE):
        scale = INPUT_IMG_SIZE / dim
    else:
        scale=1

    if(w>h):
        w = scale * w * INFLATION_RATIO
        h = w
    else:
        h = scale * h * INFLATION_RATIO
        w = h
    
    # sanity check
    xmin = max(0, int(x_c - w//2))
    ymin = max(0, int(y_c - h//2))
    xmax = min(img.shape[1], int(x_c + w//2))
    ymax = min(img.shape[0], int(y_c + h//2))

    img_cropped = img[ymin:ymax, xmin:xmax]
    img_cropped = cv2.resize(img_cropped, (INPUT_IMG_SIZE, INPUT_IMG_SIZE))

    return img_cropped

def generate_cropped_images(behave_dir, save_dir_name='cropped_images'):
    """ Generate cropped images from the original images in the behave dataset."""

    assert os.path.exists(behave_dir), f"{behave_dir} does not exist."
    
    sets = ['train', 'val', 'test']

    for set_name in sets:

        if not os.path.exists(os.path.join(behave_dir, set_name)):
            warnings.warn(f"{os.path.join(behave_dir, set_name)} does not exist. Skipping...")
            continue

        img_dir = os.path.join(behave_dir, set_name, 'images')
        mask_dir = os.path.join(behave_dir, set_name, 'masks')
        img_list = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

        if not os.path.exists(os.path.join(behave_dir, save_dir_name)):
            os.makedirs(os.path.join(behave_dir, set_name, save_dir_name), exist_ok=True)

        for idx, img_name in tqdm(enumerate(img_list)):
            
            if idx % 100 == 0:
                print(f"Processing {idx}th image for set: {set_name}..............")

            fileID = img_name.split('.')[0]
            img = cv2.imread(os.path.join(img_dir, img_name))

            mask_obj = cv2.imread(os.path.join(mask_dir, fileID + '_obj.png'), cv2.IMREAD_GRAYSCALE)
            mask_human = cv2.imread(os.path.join(mask_dir, fileID + '_hum.png'), cv2.IMREAD_GRAYSCALE)
            mask_combined = np.maximum(mask_obj, mask_human)
            img_cropped = preprocess_image(img, mask_combined)

            assert img_cropped.shape == (224, 224, 3), f"Image shape is {img_cropped.shape} instead of (224, 224, 3)."
            cv2.imwrite(os.path.join(behave_dir, set_name, save_dir_name, img_name), img_cropped)

        print("Finished cropping {} images for set: {}".format(len(img_list), set_name))


if __name__ == '__main__':

    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('--behave_dir', type=str, default='~/behave/', help='Path to the behave dataset directory containing train/ val/')
    args = parser.parse_args()

    assert os.path.exists(args.behave_dir), f"{args.behave_dir} does not exist."
    generate_cropped_images(args.behave_dir, save_dir_name='cropped_images')