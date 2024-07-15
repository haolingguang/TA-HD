import os
import argparse
import random
import numpy as np

import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

import timm
from utils.Dataset import Dataset,NIPS_trans
import scipy.stats as st
import numpy as np
import torch_dct as dct
from sklearn import metrics
import torch.nn.functional as F
from utils.U_Net import Unet

from torchvision.utils import save_image

def seed_torch(seed=20):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
seed_torch()

parser = argparse.ArgumentParser(description="ours")
parser.add_argument('--attack_name', default='FGSM_MI',type=str)
parser.add_argument('--HD_name', default='HD_MI',type=str)
parser.add_argument('--defense_name', default='resize',type=str)
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 4)')
parser.add_argument('--image_size', default=224, type=int, help='the size of the image')
parser.add_argument('--epoch', default=1, type=int, help = 'the numbers of train epoch')
parser.add_argument('--gpu', default='1', type=str, help='number of gpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda')
parser.add_argument('--local_model', default='inception_v3',type=str, help='train model name')
parser.add_argument('--remote_models',  default=['inception_v3'], type=list, help='test model name')
parser.add_argument('--Clean_NIPS', default='/raid/haolingguang/dataset/archive',type=str, help='input directory')
parser.add_argument('--Clean_ImageNet', default='./newDataset_1k',type=str, help='input directory')
parser.add_argument('--categories', default='/raid/haolingguang/dataset/archive/images.csv', type=str, help='label file directory')
parser.add_argument('--workers', default = 4,type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--U_Net', default = 'Unet', type=str)
parser.add_argument('--test_dataset', default = 'NIPS',type=str)



args = parser.parse_args()
print ("args", args)

#GPU number
if not args.gpu == 'None':
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
# writer = SummaryWriter(os.path.join(args.method_dir,'Tensorboard'))

RESULT = 'result'
METHOD_DIR=os.path.join(RESULT,args.attack_name)
DATASET_DIR=os.path.join(METHOD_DIR,args.test_dataset)
MODEL_DIR=os.path.join(DATASET_DIR,args.local_model)
IMAGE_DIR=os.path.join(MODEL_DIR, 'image')
OUTPUT_DENOISE=os.path.join(MODEL_DIR,'denoise_image')
DEFENSE_ = os.path.join(OUTPUT_DENOISE, args.defense_name)


HD_PRARMETERS_DIR=os.path.join('result',args.HD_name, args.test_dataset, args.local_model, 'HD_parameter')


def mkdir_path(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        
mkdir_path(RESULT)
mkdir_path(METHOD_DIR)
mkdir_path(DATASET_DIR)
mkdir_path(MODEL_DIR)                                                                                                                                                                                                                                                                                                   
mkdir_path(IMAGE_DIR)
mkdir_path(DEFENSE_)
mkdir_path(HD_PRARMETERS_DIR)
mkdir_path(OUTPUT_DENOISE) 



def freeze(model):
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False

def load_model(model_name):
    model = Unet(3,3)
    model = torch.nn.DataParallel(model).cuda()
    model_para = torch.load(os.path.join(HD_PRARMETERS_DIR, 
        (args.test_dataset + '_' + args.U_Net + '_' + args.local_model + '_' + str(args.epoch) + '.pth')))
    model.load_state_dict(model_para)
    model.eval()
    freeze(model)
    return model

class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, _ = self.samples[index]
        image, target = super().__getitem__(index)
        filename = path.split('/')[-1]  # 获取文件名
        return image, target, filename
    
def main():

    # 加载所有的模型
    HDN=load_model(args.local_model)
    # 定义对抗样本的transform，仅仅需要转换为-1,1
    transform = transforms.Compose([
        transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    # 测试所有模型在对抗样本上的精度
    dir = os.path.join(IMAGE_DIR,str(0))
    if args.test_dataset == 'NIPS':
        Adv_dataset = Dataset(root=dir, target_file=args.categories, transform=transform)
        Adv_loader = DataLoader(Adv_dataset, batch_size=1,  shuffle=False,pin_memory=True)
    
            
    input_HD = HDN
    input_transform = SIA()
    for batch_idx, (input, true_label,filename) in enumerate(Adv_loader): 
        input = input.cuda()
        
        if args.defense_name == 'HD':
            input_denoise = input_HD(input)
        elif args.defense_name == 'scale':
            input_denoise = input_transform.scale(input)
        elif args.defense_name == 'resize':
            input_denoise = input_transform.resize(input)
        elif args.defense_name == 'horizontal_shift':
            input_denoise = input_transform.horizontal_shift(input)
            
        # input_flip = input_transform.horizontal_flip(input)
        Save_NIPS(input_denoise, filename, os.path.join(DEFENSE_, str(args.epoch)))
        # Save_NIPS(input_scale, filename, os.path.join(OUTPUT_DENOISE,'input_scale'))
        # Save_NIPS(input_resize, filename, os.path.join(OUTPUT_DENOISE,'input_resize'))
        # Save_NIPS(input_shift, filename, os.path.join(OUTPUT_DENOISE,'input_shift'))
        # Save_NIPS(input_flip, filename, os.path.join(OUTPUT_DENOISE,'input_flip'))
        
            
                
def Save_NIPS(image, filename, output_dir):
    # 反归一化图像
    if not os.path.exists(output_dir):
            os.makedirs(output_dir) 
    for index in range(len(image)):
        image[index] = image[index].div_(2).add(0.5)
        image_path = os.path.join(output_dir,filename[index])
        save_image(image[index], image_path)
            
class SIA(torch.nn.Module):
    
    def __init__(self, num_copies=20, num_block=3):
        super(SIA,self).__init__()
        self.num_copies = num_copies
        self.num_block = num_block
        self.op = [self.resize, self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip, self.rotate180, self.scale, self.add_noise,self.dct,self.drop_out]
        

    def forward(self, x):
        return torch.cat([self.blocktransform(x) for _ in range(self.num_copies)])
    
        
    def vertical_shift(self, x):
        _, _, w, _ = x.shape
        step = np.random.randint(low = 0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)


    def horizontal_shift(self, x):
        _, _, _, h = x.shape
        step = 50
        return x.roll(step, dims=3)


    def vertical_flip(self, x):
        return x.flip(dims=(2,))


    def horizontal_flip(self, x):
        return x.flip(dims=(3,))


    def rotate180(self, x):
        return x.rot90(k=2, dims=(2,3))
    
    
    def scale(self, x):
        return 0.5 * x
    

    
    def resize(self, x):
        # if torch.rand(1)<0.5:
        _, _, w, h = x.shape
        scale_factor = 0.8
        new_h = int(h * scale_factor)+1
        new_w = int(w * scale_factor)+1
        # rnd = torch.randint(224, 254, ()).item()
        h_rem = 254 - new_h
        w_rem = 254 - new_w
        pad_top = torch.randint(0, h_rem, ())
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0 , w_rem, ())
        pad_right = w_rem - pad_left
        rescaled = F.interpolate(x, size=[new_h, new_w], mode='bilinear', align_corners=False)
        x = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom], value=-1)

        return x
    
    def dct(self, x):
        """
        Discrete Fourier Transform
        """
        dctx = dct.dct_2d(x) #torch.fft.fft2(x, dim=(-2, -1))
        _, _, w, h = dctx.shape
        low_ratio = 0.4
        low_w = int(w * low_ratio)
        low_h = int(h * low_ratio)
        # dctx[:, :, -low_w:, -low_h:] = 0
        dctx[:, :, -low_w:,:] = 0
        dctx[:, :, :, -low_h:] = 0
        dctx = dctx # * self.mask.reshape(1, 1, w, h)
        idctx = dct.idct_2d(dctx)
        return idctx
    
    
    def add_noise(self, x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16/255,16/255), -1, 1)


    def drop_out(self, x):
        return F.dropout2d(x, p=0.1, training=True)


    def blocktransform(self, x, choice=-1):
        _, _, w, h = x.shape
        y_axis = [0,] + np.random.choice(list(range(1, h)), self.num_block-1, replace=False).tolist() + [h,]
        x_axis = [0,] + np.random.choice(list(range(1, w)), self.num_block-1, replace=False).tolist() + [w,]
        y_axis.sort()
        x_axis.sort()
        
        x_copy = x.clone()
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = self.op[chosen](x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])
        return x_copy

        

    
if __name__ =='__main__':
    main()



