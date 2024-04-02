#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:12:04 2022

convert instance labels to three class labels:
0: background
1: interior
2: boundary
@author: jma
"""
import torch
import os
join = os.path.join
import argparse
import torch
import torchvision.transforms as transforms
import random
from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
import numpy as np
import cv2

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

def create_interior_map(inst_map):
    """
    Parameters
    ----------
    inst_map : (H,W), np.int16
        DESCRIPTION.

    Returns
    -------
    interior : (H,W), np.uint8 
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
    # create interior-edge map
    boundary = segmentation.find_boundaries(inst_map, mode='inner')
    boundary = morphology.binary_dilation(boundary, morphology.disk(1))

    interior_temp = np.logical_and(~boundary, inst_map > 0)
    # interior_temp[boundary] = 0
    interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
    interior = np.zeros_like(inst_map, dtype=np.uint8)
    interior[interior_temp] = 1
    interior[boundary] = 2
    return interior

encoding_map = {
    0: [1, 0, 0],  # background
    1: [0, 1, 0],  # interior
    2: [0, 0, 1]   # boundary
}

def one_hot_encode(mask, encoding_map):
    num_classes = len(encoding_map)
    encoded_mask = np.zeros((mask.shape[0], mask.shape[1], num_classes), dtype=np.uint8)
    for class_id, encoding in encoding_map.items():
        encoded_mask[mask == class_id] = encoding
    return  np.transpose(encoded_mask, (2, 0, 1))

import torch
import torchvision.transforms as transforms
import random
from PIL import Image

class RandomCropOrResize(object):
    def __init__(self, output_size,test=False):
        self.test = test
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        img, interior_map = sample['image'], sample['interior_map']
        
        # Convert numpy arrays to PIL Images
        img = Image.fromarray(img)
        interior_map = Image.fromarray(interior_map)
        
        # Get image size
        w, h = img.size
        
        new_h, new_w = self.output_size

        positionTransform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        #if self.test:
        img = positionTransform(img)
        resizeMask = transforms.Resize((new_h, new_w),interpolation=Image.NEAREST)
        # Convert tensor to image (assuming RGB)
        img = cv2.cvtColor(np.transpose(img.numpy(), (1, 2, 0)), cv2.COLOR_RGB2BGR)
        # Upscale image using Lanczos interpolation
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        #resize = transforms.Resize((new_h, new_w), interpolation=Image.LANCZOS)
        img = np.transpose(img, (2, 0, 1))
        interior_map = resizeMask(interior_map)
    
        # Convert PIL Images back to numpy arrays
        img = np.array(img)
        interior_map = np.array(interior_map)
        return img, interior_map
def main():
    parser = argparse.ArgumentParser('Preprocessing for microscopy image segmentation', add_help=False)
    parser.add_argument('-i', '--input_path', default='./Train_Labeled', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='./Train_Pre_3class', type=str, help='preprocessing data path') 
    parser.add_argument('-l','--length', default=256, type=int, help='length of the cropped image')   
    args = parser.parse_args()
    
    source_path = args.input_path
    target_path = args.output_path
    length = args.length
    test = True
    if test:
        source_path = 'Test_Labeled'
        target_path = 'Test_Pre_3class'
    img_path = join(source_path, 'images')
    gt_path =  join(source_path, 'labels')
    
    img_names = sorted(os.listdir(img_path))
    gt_names = [img_name.split('.')[0]+'_label.tiff' for img_name in img_names]
    
    pre_img_path = join(target_path, 'images')
    pre_gt_path = join(target_path, 'labels')
    os.makedirs(pre_img_path, exist_ok=True)
    os.makedirs(pre_gt_path, exist_ok=True)
    
    for img_name, gt_name in zip(tqdm(img_names), gt_names):
        if img_name.endswith('.tif') or img_name.endswith('.tiff'):
            img_data = tif.imread(join(img_path, img_name),ioworkers=8)
        else:
            img_data = io.imread(join(img_path, img_name))
        gt_data = tif.imread(join(gt_path, gt_name),ioworkers=8)
        
        # normalize image data
        if len(img_data.shape) == 2:
            img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
        elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
            img_data = img_data[:,:, :3]
        else:
            pass
        pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
        for i in range(3):
            img_channel_i = img_data[:,:,i]
            if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
        
        # conver instance bask to three-class mask: interior, boundary
        #print(gt_data)
        interior_map = create_interior_map(gt_data.astype(np.int16))
        pre_img_data, interior_map = RandomCropOrResize((length,length),test=test)({'image':pre_img_data, 'interior_map':interior_map})
        if not test:
            interior_map = one_hot_encode(interior_map, encoding_map)
        np.save(join(target_path, 'images', img_name.split('.')[0]+'.npy'), pre_img_data)
        np.save(join(target_path, 'labels', gt_name.split('.')[0]+'.npy'), interior_map)
if __name__ == "__main__":
    main()