#这是一个使用PyTorch训练卷积自编码器的简单示例代码。
#请注意，这只是一个简单的示例，你可能需要根据你的数据和需求进行调整。

import torch
from torch import nn
import torch.nn.functional as F
class T_net(nn.Module):
    def __init__(self):
        super(T_net, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 224
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112
            nn.Conv2d(16, 4, 3, padding=1), #112
            nn.ReLU(),
            nn.MaxPool2d(2) # 56
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 2, stride=2),                       
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# class T_net(nn.Module):
#     def __init__(self):
#         super(T_net, self).__init__()

#         # 编码器
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding=1),  # 299
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # 149
#             nn.Conv2d(16, 4, 3, padding=1), #149
#             nn.ReLU(),
#             nn.MaxPool2d(2) # 74
#         )

#         # 解码器
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(4, 16, 3, stride=2,),  
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 3, 3, stride=2),
#             nn.Sigmoid()
#         )

#     def forward(self,x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

class Conv_Net(torch.nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=15, padding=7)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.conv2(x) 
        return x