import torch.utils.data as data
import torchvision.transforms as transforms
import torch

import os
from PIL import Image
import pandas as pd
# from torchvision.transforms.transforms import CenterCrop

IMG_EXTENSIONS = ['.png', '.jpg']

class NIPS_Norm(object):
    """Normalize to -1..1 in Google Inception style
    """
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor
    
# def ImageNet_trans(img_size):
#     tf = transforms.Compose([
#         transforms.Resize(img_size),
#         transforms.CenterCrop(img_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#     return tf

def NIPS_trans(img_size):
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        NIPS_Norm()
    ])
    return tf

def find_inputs(folder, true_label=None, types=IMG_EXTENSIONS):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            _, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                label = true_label[rel_filename.split('.')[0]] if true_label else 0
                inputs.append((abs_filename, label, rel_filename))
    return inputs


class Dataset(data.Dataset):
    
    def __init__(self, root, target_file='images.csv', transform=None):
        
        if target_file:  
            target_file_path = target_file
            # target_file_path = os.path.join(root, target_file)
            target_df = pd.read_csv(target_file_path)#, header=None)
            target_df["TrueLabel"] = target_df["TrueLabel"].apply(int)
            true_label = dict(zip(target_df["ImageId"], target_df["TrueLabel"] - 1))  
        else:
            true_label = dict()

        imgs = find_inputs(root, true_label)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path, target, filename = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target, filename

    def __len__(self):
        return len(self.imgs)

    
