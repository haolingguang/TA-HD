#!/bin/bash
python TA_HD_test.py --U_Net 'Unet' --local_model 'inception_v3'  
# python TA_HD_test.py --U_Net 'Conv_Net' --local_model 'inception_v3'
python TA_HD_test.py --U_Net 'Unet' --local_model 'inception_v4' 
# python TA_HD_test.py --U_Net 'Conv_Net' --local_model 'inception_v4' 
python TA_HD_test.py --U_Net 'Unet' --local_model 'inception_resnet_v2'
# python TA_HD_test.py --U_Net 'Conv_Net' --local_model 'inception_resnet_v2' 
python TA_HD_test.py --U_Net 'Unet' --local_model 'resnet101' 
