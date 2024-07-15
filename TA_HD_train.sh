#!/bin/bash


python TA_HD_train.py --U_Net 'Unet' --local_model 'inception_v3'  --test_dataset 'NIPS'
# python TA_HD_train.py --U_Net 'Conv_Net' --local_model 'inception_v3'    --test_dataset 'NIPS'
python TA_HD_train.py --U_Net 'Unet' --local_model 'inception_v4'    --test_dataset 'NIPS' 
# python TA_HD_train.py --U_Net 'Conv_Net' --local_model 'inception_v4'    --test_dataset 'NIPS' 
python TA_HD_train.py --U_Net 'Unet' --local_model 'inception_resnet_v2' --test_dataset 'NIPS'
# python TA_HD_train.py --U_Net 'Conv_Net' --local_model 'inception_resnet_v2' --test_dataset 'NIPS' 
python TA_HD_train.py --U_Net 'Unet' --local_model 'resnet101'   --test_dataset 'NIPS' 
# python TA_HD_train.py --U_Net 'Conv_Net' --local_model 'resnet101'   --test_dataset 'NIPS' 


python TA_HD_train.py --U_Net 'Unet' --local_model 'inception_v3'  --test_dataset 'New_data'
# python TA_HD_train.py --U_Net 'Conv_Net' --local_model 'inception_v3'    --test_dataset 'New_data'
python TA_HD_train.py --U_Net 'Unet' --local_model 'inception_v4'    --test_dataset 'New_data' 
# python TA_HD_train.py --U_Net 'Conv_Net' --local_model 'inception_v4'    --test_dataset 'New_data' 
python TA_HD_train.py --U_Net 'Unet' --local_model 'inception_resnet_v2' --test_dataset 'New_data'
# python TA_HD_train.py --U_Net 'Conv_Net' --local_model 'inception_resnet_v2' --test_dataset 'New_data' 
python TA_HD_train.py --U_Net 'Unet' --local_model 'resnet101'   --test_dataset 'New_data' 
# python TA_HD_train.py --U_Net 'Conv_Net' --local_model 'resnet101'   --test_dataset 'New_data' 

