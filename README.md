# A Hypothetical Defenses-based Training Framework for GeneratingTransferable Adversarial Examples （TA-HD）

## Introduction    
Transfer-based attacks utilize the proxy model to craft adversarial examples against the target model and make significant advancements in the realm of black-box attacks. Recent research suggests that these attacks can be enhanced by incorporating adversarial defenses into the training process of adversarial examples. Specifically, adversarial defenses supervise the training process, forcing the attacker to overcome greater challenges and produce more robust adversarial examples with enhanced transferability. However, current methods mainly rely on limited input transformation defenses, which apply only linear affine changes. These defenses are insufficient for effectively removing harmful content from adversarial examples, resulting in restricted improvements in their transferability. To address this issue, we propose a novel training framework named Transfer-based Attacks through Hypothesis Defense (TA-HD). This framework enhances the generalization of adversarial examples by integrating a hypothesis defense mechanism into the proxy model. Specifically, we propose an input denoising network as the hypothesis defense to remove harmful noise from adversarial examples effectively. Furthermore, we introduce an adversarial training strategy and design specific adversarial loss functions to optimize the input denoising network’s parameters. The visualization of the training process demonstrates not only the effective denoising capability of the hypothesized defense mechanism but also the stability of the training process. Extensive experiments show that the proposed training framework significantly improves the success rate of transfer-based attacks up to 19.9%. 

## Implement   
download dataset [Baidu_pan](https://pan.baidu.com/s/1qRHLwirC_MeFKJvGDAphQQ?pwd=3xrz)    [Google drive](https://drive.google.com/file/d/1M922wnWCA5Ro3_xd5jC5z8FBVkzAfNoR/view?usp=drive_link)  

## Main Environment  
> Ubuntu 20.04  
> CUDA 11.8   
> cudnn8   
> python==3.10.13   
> pytorch==2.0.1   
> torchvision==0.15.2   
> timm == 1.0.3    


## RUN  

### TA-HD   
- generate adversarial examples   
> python TA_HD_train.py  
     
- test ASR    
> python TA_HD_test.py 




### FGSM   
- generate adversarial examples    
> python FGSM_train.py
     
- test ASR    
> python FGSM_test.py 

  
### test defense performance   
> python Defense_visual.py 

  
### CAM visual (hot map)    
> python Image_CAM.py  



