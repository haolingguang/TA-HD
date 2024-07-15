import os
import datetime
import argparse
import random
import numpy as np

import torch
from torch import autograd
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import timm
from utils.Dataset import Dataset,NIPS_trans
from FGSM_ATTACK.FGSM_MI import FGSM_MI
from FGSM_ATTACK.FGSM_DI import FGSM_DI
from FGSM_ATTACK.FGSM_TI import FGSM_TI
from FGSM_ATTACK.FGSM_SI import FGSM_SI
from FGSM_ATTACK.FGSM_PGN import FGSM_PGN
from FGSM_ATTACK.FGSM_GRA import FGSM_GRA
from FGSM_ATTACK.FGSM_SIA import FGSM_SIA


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
parser.add_argument('--method', default='FGSM_MI',type=str)
parser.add_argument('--max_epsilon', default=16, type=int, help='max perturbation value')
parser.add_argument('--norm', default= 1, type=int, help='lp norm type')
parser.add_argument('--num_steps', default=10, type=int, help='number of step')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 4)')
parser.add_argument('--image_size', default=224, type=int, help='the size of the image')
parser.add_argument('--epoch', default=10, type=int, help = 'the numbers of train epoch')
parser.add_argument('--gpu', default='0', type=str, help='number of gpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda')
parser.add_argument('--local_model', default='inception_v3',type=str, help='train model name')
parser.add_argument('--NIPS_data', default='/data/archive',type=str, help='input directory')
parser.add_argument('--New_data1w', default='./data/newDataset_1w',type=str, help='input directory')
parser.add_argument('--New_data1k', default='./data/newDataset_1k',type=str, help='input directory')
parser.add_argument('--ImageNet_data', metavar='DIR',default='/raid/datasets/ImageNet/ILSVRC/Data/CLS-LOC',
                    help='path to dataset')
parser.add_argument('--ImageNetv2', default='/raid/datasets/',type=str, help='input directory')
parser.add_argument('--categories', default='/raid/haolingguang/dataset/archive/images.csv', type=str, help='label file directory')
parser.add_argument('--workers', default = 4,type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_dataset', default = 'NIPS',type=str)



args = parser.parse_args()
print ("args", args)

#GPU number
if not args.gpu == 'None':
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


RESULT = 'result'
METHOD_DIR=os.path.join(RESULT,args.method)
DATASET_DIR=os.path.join(METHOD_DIR,args.test_dataset)
MODEL_DIR=os.path.join(DATASET_DIR,args.local_model)
IMAGE_DIR=os.path.join(MODEL_DIR, 'image')


def mkdir_path(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        
mkdir_path(RESULT)
mkdir_path(METHOD_DIR)
mkdir_path(DATASET_DIR)
mkdir_path(MODEL_DIR)                                                                                                                                                                                                                                                                                                   
mkdir_path(IMAGE_DIR)


for i in range(args.epoch):
    mkdir_path(os.path.join(IMAGE_DIR,str(i)))



def Save_NIPS(image, filename, output_dir,epoch):
    # 反归一化图像
    for index in range(len(image)):
        image[index] = image[index].div_(2).add(0.5)
        image_path = os.path.join(output_dir,str(epoch),filename[index])
        save_image(image[index], image_path)
            

def Save_ImageNet(image, filename, output_dir, epoch):  
    # 建立类文件夹
    for index in range(len(image)):
        dst = os.path.join(output_dir, str(epoch), filename[index].split('_')[0])
        if not os.path.exists(dst):
            os.makedirs(dst)  
            
        # 反归一化图像
        image[index] = image[index].div_(2).add(0.5)
        image_path = os.path.join(dst, filename[index])
        save_image(image[index], image_path)

# 冻住模型参数
def freeze(model):
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False

# 通过timm库加载模型
def load_model(model_name):
    model = timm.create_model(model_name, num_classes=1000, pretrained=True)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    freeze(model)
    return model

# Dataset for new_data
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, _ = self.samples[index]
        image, target = super().__getitem__(index)
        filename = path.split('/')[-1]  # 获取文件名
        return image, target, filename
    
    
def Attack(args, attack):   
    
    # load 
    New_data1k = CustomImageFolder(root=args.New_data1k, transform=NIPS_trans(args.image_size))
    New_data1k_loader = DataLoader(New_data1k, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    # load NIPS 2017 adversarial dataset    
    NIPS_data = Dataset(root=args.NIPS_data, target_file=args.categories, transform=NIPS_trans(args.image_size))
    NIPS_loader = DataLoader(NIPS_data, batch_size=args.batch_size,  shuffle=False,pin_memory=True)
    
    
    local_model = load_model(args.local_model)

    if args.test_dataset == 'NIPS':
        val_loader = NIPS_loader
    else:
        val_loader = New_data1k_loader
    
    for epoch in range(args.epoch):
        for _, (input, true_label, filename) in enumerate(val_loader):    
            input = input.cuda()
            true_label = true_label.cuda()
            non_target_adv = attack.non_target_attack(local_model, input, true_label)
            if args.test_dataset == 'NIPS':
                Save_NIPS(non_target_adv.clone(), filename, IMAGE_DIR, epoch)
            else:
                Save_ImageNet(non_target_adv, filename, IMAGE_DIR,epoch)
                
    
def main():
    
     # 选择攻击方式
    if args.method == 'FGSM_MI':
        attack = FGSM_MI(args.max_epsilon,args.norm,args.num_steps )
    elif args.method == 'FGSM_DI':
        attack = FGSM_DI(args.max_epsilon,args.norm,args.num_steps )
    elif args.method == 'FGSM_TI':
        attack = FGSM_TI(args.max_epsilon,args.norm,args.num_steps )
    elif args.method == 'FGSM_SI':
        attack = FGSM_SI(args.max_epsilon,args.norm,args.num_steps )
    elif args.method == 'FGSM_SIA':
        attack = FGSM_SIA(args.max_epsilon,args.norm,args.num_steps )
    elif args.method == 'FGSM_PGN':
        attack = FGSM_PGN(args.max_epsilon,args.norm,args.num_steps )
    elif args.method == 'FGSM_GRA':
        attack = FGSM_GRA(args.max_epsilon,args.norm,args.num_steps )
        
        
    start = datetime.datetime.now()
    Attack(args, attack)
    end = datetime.datetime.now()
    print ("time:", end-start)
    
if __name__ =='__main__':
    main()



