import os
import datetime
import argparse
import random
import numpy as np

import torch
from torch import optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import timm


from utils.Dataset import Dataset,NIPS_trans


from utils.U_Net import Unet, Unet_5, Unet_4, Unet_3, Unet_2
from utils.ATTA_Net import Conv_Net


from HD_ATTACK.HD_MI import HD_MI
from HD_ATTACK.HD_DI import HD_DI
from HD_ATTACK.HD_TI import HD_TI
from HD_ATTACK.HD_SI import HD_SI
from HD_ATTACK.HD_SIA import HD_SIA
from HD_ATTACK.HD_PGN import HD_PGN
from HD_ATTACK.HD_GRA import HD_GRA

# import xlrd
import matplotlib.pyplot as plt
# from matplotlib import font_manager
# from matplotlib.pyplot import MultipleLocator


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

parser = argparse.ArgumentParser(description="HD ATTACK")
parser.add_argument('--method', default='HD_MI',type=str)
parser.add_argument('--max_epsilon', default=16, type=int, help='max perturbation value')
parser.add_argument('--norm', default= 1, type=int, help='lp norm type')
parser.add_argument('--num_steps', default=10, type=int, help='number of step')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 4)')
parser.add_argument('--image_size', default=224, type=int, help='the size of the image')
parser.add_argument('--epoch', default=10, type=int, help = 'the numbers of train epoch')
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='number of gpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda')
parser.add_argument('--local_model', default='inception_v3',type=str, help='train model name')
parser.add_argument('--NIPS_data', default='/data/archive',type=str, help='input directory')
parser.add_argument('--New_data1w', default='./data/newDataset_1w',type=str, help='input directory')
parser.add_argument('--New_data1k', default='./data/newDataset_1k',type=str, help='input directory')
parser.add_argument('--ImageNet_data', metavar='DIR',default='./data/ImageNet/ILSVRC/Data/CLS-LOC',
                    help='path to dataset')
parser.add_argument('--ImageNetv2', default='./data',type=str, help='input directory')
parser.add_argument('--categories', default='./data/archive/images.csv', type=str, help='label file directory')
parser.add_argument('--workers', default = 4,type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_dataset', default = 'NIPS',type=str)
parser.add_argument('--scheduler', default = True,type=bool)
parser.add_argument('--U_Net', default = 'Unet', type=str)
parser.add_argument('--loss_mode', default = 'LGD_CGD', type=str)

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
LOSS_DIR=os.path.join(MODEL_DIR, 'loss')
PRARMETERS_DIR=os.path.join(MODEL_DIR, 'HD_parameter')

def mkdir_path(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        
mkdir_path(RESULT)
mkdir_path(METHOD_DIR)
mkdir_path(DATASET_DIR)
mkdir_path(MODEL_DIR)                                                                                                                                                                                                                                                                                                   
mkdir_path(IMAGE_DIR)
mkdir_path(LOSS_DIR)
mkdir_path(PRARMETERS_DIR)


for i in range(args.epoch):
    mkdir_path(os.path.join(IMAGE_DIR,str(i)))
writer = SummaryWriter(os.path.join(MODEL_DIR,'Tensorboard'))


class Attack:
    def __init__(self, atta, train_loader, val_loader, local_model,scheduler):
        '''initialize the parameters'''
        self.attack = atta
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_model = local_model
        self.scheduler = scheduler
        
        
    def train(self,train_model,train_loader,epoch,loss):  
        l = len(train_loader)     
        for batch_id, (input, true_label, filename) in enumerate(train_loader):    
            input = input.cuda()
            true_label = true_label.cuda()
            self.attack.train_T(train_model, input, true_label, writer, args.loss_mode, args.U_Net, global_step=epoch * l + batch_id) 
        torch.cuda.empty_cache()
        return loss
    
    
    def test(self,local_model,val_loader,epoch):
        for _, (input, true_label, filename) in enumerate(val_loader):    
            input = input.cuda()
            true_label = true_label.cuda()
            non_target_adv = self.attack.test_T(local_model, input, true_label)
            if args.test_dataset == 'NIPS':
                Save_NIPS(non_target_adv, filename, IMAGE_DIR, epoch)
            else:
                Save_ImageNet(non_target_adv, filename, IMAGE_DIR,epoch)
              
               
    def iter(self,epochs):
        max_success_rate = 0
        loss=[[] for i in range(3)]
        for epoch in range(epochs):
            # for name, parameters in self.attack.T_net.named_parameters():
            #     print(name)
            self.attack.T_net.train() 
            # print(self.attack.T_net.state_dict()['module.inc.double_conv.0.weight']) 
            # print(self.attack.T_net.state_dict()['module.inc.double_conv.0.bias'])
            # print(self.attack.T_net.state_dict()['module.inc.double_conv.1.weight'])
            # print(self.attack.T_net.state_dict()['module.inc.double_conv.1.bias'])
            loss=self.train(self.local_model, self.train_loader,epoch,loss)
            # print(self.attack.T_net.state_dict()['module.inc.double_conv.1.weight']) 
            self.attack.T_net.eval()
            self.test(self.local_model,self.val_loader,epoch)
            # print(self.attack.T_net.state_dict()['module.inc.double_conv.1.weight'])  
            # log_string('| epoch: %d | Tar_Acc: %.3f | Ori_Acc: %.3f | Succ_rate: %.3f'%(epoch, Tar_Acc, Ori_Acc, Succ_rate)) 
            
            if args.scheduler:
                self.scheduler.step()
            

            torch.save(self.attack.T_net.state_dict(), (PRARMETERS_DIR + '/{}_{}_{}_{}.pth').format(args.test_dataset, args.U_Net, args.local_model,str(epoch)))

            # if epoch==1:
            #     torch.save(self.attack.T_net.state_dict(), os.path.join(OUTPUT_MODEL,'model_para.pth'))
            print(epoch)
        plot(loss[0],os.path.join(LOSS_DIR,'Adv_loss.pdf'),'Adv Loss')
        plot(loss[1],os.path.join(LOSS_DIR,'Clean_loss.pdf'),'Clean Loss')
        plot(loss[2],os.path.join(LOSS_DIR,'L2-norm.pdf'),'L2 Norm')
        return max_success_rate    
     

def Save_NIPS(image, filename, output_dir,epoch):
    # Denormalized image
    for index in range(len(image)):
        image[index] = image[index].div_(2).add(0.5)
        image_path = os.path.join(output_dir,str(epoch),filename[index])
        save_image(image[index], image_path)
            
def Save_ImageNet(image, filename, output_dir, epoch):  
    
    # Create folders
    for index in range(len(image)):
        dst = os.path.join(output_dir, str(epoch), filename[index].split('_')[0])
        if not os.path.exists(dst):
            os.makedirs(dst)  
            
        # Denormalized image
        image[index] = image[index].div_(2).add(0.5)
        image_path = os.path.join(dst, filename[index])
        save_image(image[index], image_path)

# Freeze model parameters
def freeze(model):
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False

# Load the model through the timm library
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
        filename = path.split('/')[-1]  
        return image, target, filename
    
    
def main():
    

    if args.U_Net=='Unet_5':
        t_net = Unet_5(3,3)
    elif args.U_Net=='Unet_4':
        t_net = Unet_4(3,3)
    elif args.U_Net=='Unet_3':
        t_net = Unet_3(3,3)
    elif args.U_Net=='Unet_2':
        t_net = Unet_2(3,3)
    elif args.U_Net=='Unet':
        t_net = Unet(3,3)
    elif args.U_Net=='Conv_Net':
        t_net = Conv_Net()
 
 
    t_net = torch.nn.DataParallel(t_net).cuda()
    optimizer = optim.Adam(t_net.parameters(),lr=0.001) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=0, last_epoch=-1)
    
        

    if args.method == 'HD_MI':
        atta = HD_MI(args.max_epsilon,args.norm,args.num_steps,t_net,optimizer)
    elif args.method == 'HD_DI':
        atta = HD_DI(args.max_epsilon,args.norm,args.num_steps,t_net,optimizer)
    elif args.method == 'HD_TI':
        atta = HD_TI(args.max_epsilon,args.norm,args.num_steps,t_net,optimizer)
    elif args.method == 'HD_SI':
        atta = HD_SI(args.max_epsilon,args.norm,args.num_steps,t_net,optimizer)
    elif args.method == 'HD_SIA':
        atta = HD_SIA(args.max_epsilon,args.norm,args.num_steps,t_net,optimizer)
    elif args.method == 'HD_PGN':
        atta = HD_PGN(args.max_epsilon,args.norm,args.num_steps,t_net,optimizer)
    elif args.method == 'HD_GRA':
        atta = HD_GRA(args.max_epsilon,args.norm,args.num_steps,t_net,optimizer)

    
    # load ImageNet_v2 dataset
    ImageNetv2_data = CustomImageFolder(root=args.ImageNetv2+'ImageNetV2-matched-frequency', transform=NIPS_trans(args.image_size))
    ImageNetv2_loader = DataLoader(ImageNetv2_data,batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    New_data1w = CustomImageFolder(root=args.New_data1w, transform=NIPS_trans(args.image_size))
    New_data1w_loader = DataLoader(New_data1w, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    New_data1k = CustomImageFolder(root=args.New_data1k, transform=NIPS_trans(args.image_size))
    New_data1k_loader = DataLoader(New_data1k, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    # load NIPS 2017 adversarial dataset    
    NIPS_data = Dataset(root=args.NIPS_data, target_file=args.categories, transform=NIPS_trans(args.image_size))
    NIPS_loader = DataLoader(NIPS_data, batch_size=args.batch_size,  shuffle=False,pin_memory=True)
    
    
    local_model = load_model(args.local_model)

    
    if args.test_dataset == 'NIPS':
        attack =Attack(atta, ImageNetv2_loader, NIPS_loader, local_model, scheduler)
    elif args.test_dataset == 'New_data':
        attack = Attack(atta, New_data1w_loader, New_data1k_loader, local_model, scheduler)
    
    # start attack
    start = datetime.datetime.now()
    attack.iter(args.epoch)
    end = datetime.datetime.now()
    print ("time:", end-start)
    

def plot(data, path, label):
    

    plt.rcParams['font.sans-serif'] = ['Times New Roman']  
    plt.rcParams['axes.unicode_minus'] = False 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    t = data
    xAxis1 = range(0,len(t))
    plt.plot(xAxis1, t, color='blue', linestyle='-',linewidth=1)


    plt.xlabel('Iterations',fontsize=15)
    plt.ylabel(label,fontsize=15)
    # plt.title("InceptionResnet-v2",fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    ax=plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(200))
    # ax.yaxis.set_major_locator(MultipleLocator(y_locator))
    # # plt.xlim(0,11)
    # plt.ylim(-y_clim,y_clim)

    # plt.legend()

    # plt.grid(axis="y", linewidth=0.1)
    plt.savefig(path,dpi=500,bbox_inches = 'tight')
    plt.show()
    plt.clf()
    
    
if __name__ =='__main__':
    main()
