import os
import argparse
import random
import numpy as np
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

import timm
from utils.Dataset import Dataset, NIPS_trans


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
parser.add_argument('--method', default='HD_MI',type=str)
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 4)')
parser.add_argument('--image_size', default=224, type=int, help='the size of the image')
parser.add_argument('--epoch', default=10, type=int, help = 'the numbers of train epoch')
parser.add_argument('--gpu', default='1', type=str, help='number of gpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda')
parser.add_argument('--local_model', default='inception_v3',type=str, help='train model name')
parser.add_argument('--remote_models',  default=['inception_v3', 'inception_v4', 'inception_resnet_v2', 'resnet101', 
        'xception', 'dpn98', 'vgg19','convnext_base','vit_base_patch16_224','efficientnetv2_rw_m','mixer_b16_224',
        'ens_adv_inception_resnet_v2','adv_inception_v3'], type=list, help='test model name')
parser.add_argument('--Clean_NIPS', default='./data/archive',type=str, help='input directory')
parser.add_argument('--Clean_ImageNet', default='./data/newDataset_1k',type=str, help='input directory')
parser.add_argument('--categories', default='/raid/haolingguang/dataset/archive/images.csv', type=str, help='label file directory')
parser.add_argument('--workers', default = 4,type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--U_Net', default = 'Unet', type=str)
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
LOG_DIR=os.path.join(MODEL_DIR, 'log')


def mkdir_path(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        
mkdir_path(RESULT)
mkdir_path(METHOD_DIR)
mkdir_path(DATASET_DIR)
mkdir_path(MODEL_DIR)                                                                                                                                                                                                                                                                                                   
mkdir_path(IMAGE_DIR)
mkdir_path(LOG_DIR)


# write log
LOG_FOUT = open(os.path.join(LOG_DIR, args.local_model +'.txt'), 'a')


def log_string(out_str):
    LOG_FOUT.write(out_str+' | ')
    LOG_FOUT.flush()
    print(out_str)


def freeze(model):
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False

def load_model(model_name):
    model = timm.create_model(model_name, num_classes=1000, pretrained=True)
    model = torch.nn.DataParallel(model).cuda()
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
    test_models=[]
    for model in args.remote_models:
        test_models.append(load_model(model))
    
    
    # 加载清洁样本
    if args.test_dataset == 'NIPS':
        # load NIPS 2017 adversarial dataset    
        Clean_data = Dataset(root=args.Clean_NIPS, target_file=args.categories, transform=NIPS_trans(args.image_size))
        Clean_loader = DataLoader(Clean_data, batch_size=args.batch_size,  shuffle=False,pin_memory=True)
    else:
        # load ImageNet dataset  
        Clean_data = CustomImageFolder(root=args.Clean_ImageNet, transform=NIPS_trans(args.image_size))
        Clean_loader = DataLoader(Clean_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    
    # Correct是一个列表，长度为模型列表长度，记录每个模型在清洁数据集上的正确识别的样本个数
    total=0
    correct = [0]*len(test_models)  
    correct_rate = [0]*len(test_models)
    log_string('\n')
    for batch_idx, (input, true_label,_) in enumerate(Clean_loader): 
        input = input.cuda()
        true_label = true_label.cuda()
        for i in range(len(test_models)):
            _,output = test_models[i](input).max(1)
            correct[i]+=torch.sum(output== true_label)
        total += true_label.size(0)
    log_string('')
    for i in range(len(test_models)):    
        correct_rate[i] =  correct[i]/total
        log_string( '%.3f'%correct_rate[i])
    
    
    # 定义对抗样本的transform，仅仅需要转换为-1,1
    transform = transforms.Compose([
        transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    
    # 测试所有模型在对抗样本上的精度
    for epoch in range(args.epoch):
        log_string('\n')
        dir = os.path.join(IMAGE_DIR,str(epoch))
        if args.test_dataset == 'NIPS':
            Adv_dataset = Dataset(root=dir, target_file=args.categories, transform=transform)
            Adv_loader = DataLoader(Adv_dataset, batch_size=args.batch_size,  shuffle=False,pin_memory=True)
        else:
            Adv_dataset = CustomImageFolder(root=dir, transform=transform)
            Adv_loader = DataLoader(Adv_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
        success = [0]*len(test_models)    
        success_rate = [0]*len(test_models) 
        for batch_idx, (input, true_label,_) in enumerate(Adv_loader): 
            input = input.cuda()
            true_label = true_label.cuda()
            for i in range(len(test_models)):
                _,output = test_models[i](input).max(1)
                success[i]+=torch.sum(output!= true_label)
        for i in range(len(test_models)):    
            success_rate[i] =  1-(total-success[i])/correct[i]
            log_string( '%.3f'%success_rate[i])
            
    
if __name__ =='__main__':
    main()



