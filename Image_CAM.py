import os
import os.path
import datetime
import argparse
from PIL import Image

from torchvision.utils import save_image
import torch
# import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp, LayerCAM
from torchcam.utils import overlay_mask

import timm
from utils.Dataset import Dataset, NIPS_trans


parser = argparse.ArgumentParser(description="ours")
parser.add_argument('--test_model', default='inception_v3',type=str, help='train model name')
parser.add_argument('--test_method', default='FGSM_DI', type=str, help='test method name')
parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size (default: 4)')
parser.add_argument('--NIPS_data', default='./data/archive',type=str, help='input directory')
parser.add_argument('--categories', default='./data/archive/images.csv', type=str, help='label file directory')
parser.add_argument('--dataset_dir', default='./result/FGSM_DI/NIPS/inception_v3/image/0',type=str, help='input directory')
parser.add_argument('--method_dir', default='./Clean-image-CAM',type=str, help='path to method_result')
# parser.add_argument('--log_dir', default='./Clean-image-CAM/log', type=str, help='path to log')
parser.add_argument('--output_dir', default='./Clean-image-CAM/output', type=str, help='output directory')
parser.add_argument('--image_size', default=224, type=int, help='the size of the image')
args = parser.parse_args()
print ("args", args)

os.environ["CUDA_VISIBLE_DEVICES"]='1'

# create log file
if not os.path.exists(args.method_dir):
    os.mkdir(args.method_dir)
    
# if not os.path.exists(args.log_dir):
#     os.mkdir(args.log_dir)
    
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
        
OUTPUT_DIR = os.path.join(args.output_dir, args.test_method)
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
 


def run_iterative_attack():
    # NIPS_data = Dataset(root=args.dataset_dir, target_file=args.categories, transform=NIPS_trans(args.image_size))
    # dataset = Dataset(args.dataset_dir, transform=NIPS_trans(299))
    # loader = data.DataLoader(NIPS_data, batch_size=args.batch_size, shuffle=False)
    # dataset = Dataset(args.dataset_dir, transform=default_inception_transform())
    # loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    NIPS_data = Dataset(root=args.dataset_dir, target_file=args.categories, transform=NIPS_trans(args.image_size))
    loader = DataLoader(NIPS_data, batch_size=args.batch_size,  shuffle=False,pin_memory=True)
    
    remote_model = timm.create_model(args.test_model, num_classes=1000, pretrained=True)
    remote_model = torch.nn.DataParallel(remote_model).cuda()
    remote_model.eval()
    # print(remote_model)
    cam_extractor = LayerCAM(remote_model)
    
    for epoch in range(1):
        correct = 0
        for batch_idx, (input, true_label,filename) in enumerate(loader):    
            input = input.cuda()
            # true_label = true_label.cuda()

            out = remote_model(input)
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
            result = overlay_mask(to_pil_image(input.div_(2).add(0.5).squeeze(0)), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
            result.save(os.path.join(OUTPUT_DIR, filename[0]+'_adv.png'))
            # save_image(result,str(batch_idx),args.output_dir)
            # save_image(result,str(batch_idx),args.output_dir)


# def save_img(image, filename, output_dir):
#     image = image.div_(2).add(0.5)
#     image_path = os.path.join(output_dir, filename+'.png')
#     save_image(image, image_path)
    
def main():

    start = datetime.datetime.now()
    run_iterative_attack()
    end = datetime.datetime.now()
    print ("time:", end-start)

if __name__ =='__main__':
    main()


