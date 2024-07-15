import torch
import torch.nn as nn
# import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    ''' up path
        conv_transpose => double_conv
    '''
    def __init__(self, in_channels, out_channels, Transpose=True):
        super(Up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if Transpose:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels//2)

    def forward(self, x1, x2):
        ''' 
            conv output shape = (input_shape - Filter_shape + 2 * padding)/stride + 1
        '''

        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2,
                                    diffY // 2, diffY - diffY//2))
        x = torch.cat([x2,x1], dim=1)
        x = self.conv(x)
        return x

    # @staticmethod
    # def init_weights(m):
    #     if type(m) == nn.Conv2d:
    #         init.xavier_normal(m.weight)
    #         init.constant(m.bias,0)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class Unet(nn.Module):
    def __init__(self, in_ch, out_ch, Transporse=True):
        super(Unet, self).__init__()
        
        self.inc = (DoubleConv(in_ch, 64)) 
        self.down1 = (Down(64, 128))
        # print(list(self.down1.parameters()))
        self.down2 = (Down(128, 256)) 
        self.down3 = (Down(256, 512)) 
        # self.drop3 = nn.Dropout2d(0.5)
        factor = 1 if Transporse else 2
        self.down4 = (Down(512, 1024//factor)) 
        # self.drop4 = nn.Dropout2d(0.5)
        self.up1 = (Up(1024, 512//factor, Transporse))
        self.up2 = (Up(512, 256//factor, Transporse))
        self.up3 = (Up(256, 128//factor, Transporse))
        self.up4 = (Up(128, 64, Transporse))
        self.outc = (OutConv(64, out_ch))
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

    def forward(self, x):
        x1 = self.inc(x)  # 224
        x2 = self.down1(x1) # 112
        x3 = self.down2(x2) # 56
        x4 = self.down3(x3) # 28
        # x4 = self.drop3(x4)
        x5 = self.down4(x4) # 14
        # x5 = self.drop4(x5)
        x_d = self.up1(x5, x4) 
        x_d = self.up2(x_d, x3)
        x_d = self.up3(x_d, x2)
        x_d = self.up4(x_d, x1)
        x_d = self.outc(x_d)
        return x_d
        # self.pred_y = nn.functional.sigmoid(x)
        

class Unet_5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_5, self).__init__()
        
        self.inc = DoubleConv(in_ch, 64) 
        self.down1 = Down(64, 128) 
        # print(list(self.down1.parameters()))
        self.down2 = Down(128, 256) 
        self.down3 = Down(256, 512) 
        self.drop3 = nn.Dropout2d(0.5)
        # self.down4 = Down(512, 1024) 
        # self.drop4 = nn.Dropout2d(0.5)
        # self.up1 = Up(1024, 512, True)
        self.up2 = Up(512, 256, True)
        self.up3 = Up(256, 128, True)
        self.up4 = Up(128, 64, True)
        self.outc = OutConv(64, out_ch)

    def forward(self, x):
        x1 = self.inc(x)  # 224
        x2 = self.down1(x1) # 112
        x3 = self.down2(x2) # 56
        x4 = self.down3(x3) # 28
        x4 = self.drop3(x4)
        # x5 = self.down4(x4) # 28
        # x5 = self.drop4(x5)
        # x_d = self.up1(x5, x4) 
        x_d = self.up2(x4, x3)
        x_d = self.up3(x_d, x2)
        x_d = self.up4(x_d, x1)
        x_d = self.outc(x_d)
        return x+x_d
    
class Unet_4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_4, self).__init__()
        
        self.inc = DoubleConv(in_ch, 64) 
        self.down1 = Down(64, 128) 
        # print(list(self.down1.parameters()))
        self.down2 = Down(128, 256) 
        # self.down3 = Down(256, 512) 
        # self.drop3 = nn.Dropout2d(0.5)
        # self.down4 = Down(512, 1024) 
        # self.drop4 = nn.Dropout2d(0.5)
        # self.up1 = Up(1024, 512, True)
        # self.up2 = Up(512, 256, True)
        self.up3 = Up(256, 128, True)
        self.up4 = Up(128, 64, True)
        self.outc = OutConv(64, out_ch)

    def forward(self, x):
        x1 = self.inc(x)  # 224
        x2 = self.down1(x1) # 112
        x3 = self.down2(x2) # 56
        # x4 = self.down3(x3) # 28
        # x4 = self.drop3(x4)
        # x5 = self.down4(x4) # 28
        # x5 = self.drop4(x5)
        # x_d = self.up1(x5, x4) 
        # x_d = self.up2(x_d, x3)
        x_d = self.up3(x3, x2)
        x_d = self.up4(x_d, x1)
        x_d = self.outc(x_d)
        return x+x_d
    
class Unet_3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_3, self).__init__()
        
        self.inc = DoubleConv(in_ch, 64) 
        self.down1 = Down(64, 128) 
        # print(list(self.down1.parameters()))
        # self.down2 = Down(128, 256) 
        # self.down3 = Down(256, 512) 
        # self.drop3 = nn.Dropout2d(0.5)
        # self.down4 = Down(512, 1024) 
        # self.drop4 = nn.Dropout2d(0.5)
        # self.up1 = Up(1024, 512, True)
        # self.up2 = Up(512, 256, True)
        # self.up3 = Up(256, 128, True)
        self.up4 = Up(128, 64, True)
        self.outc = OutConv(64, out_ch)

    def forward(self, x):
        x1 = self.inc(x)  # 224
        x2 = self.down1(x1) # 112
        # x3 = self.down2(x2) # 56
        # x4 = self.down3(x3) # 28
        # x4 = self.drop3(x4)
        # x5 = self.down4(x4) # 28
        # x5 = self.drop4(x5)
        # x = self.up1(x5, x4) 
        # x = self.up2(x4, x3)
        # x = self.up3(x, x2)
        x_d = self.up4(x2, x1)
        x_d = self.outc(x_d)
        return x+x_d
    
class Unet_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_2, self).__init__()
        
        self.inc = DoubleConv(in_ch, 64) 
        # self.down1 = Down(64, 128) 
        # print(list(self.down1.parameters()))
        # self.down2 = Down(128, 256) 
        # self.down3 = Down(256, 512) 
        # self.drop3 = nn.Dropout2d(0.5)
        # self.down4 = Down(512, 1024) 
        # self.drop4 = nn.Dropout2d(0.5)
        # self.up1 = Up(1024, 512, True)
        # self.up2 = Up(512, 256, True)
        # self.up3 = Up(256, 128, True)
        # self.up4 = Up(128, 64, True)
        self.outc = OutConv(64, out_ch)

    def forward(self, x):
        x1 = self.inc(x)  # 224
        # x2 = self.down1(x1) # 112
        # x3 = self.down2(x2) # 56
        # x4 = self.down3(x3) # 28
        # x4 = self.drop3(x4)
        # x5 = self.down4(x4) # 28
        # x5 = self.drop4(x5)
        # x = self.up1(x5, x4) 
        # x = self.up2(x4, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x2, x1)
        x_d = self.outc(x1)
        return x+x_d
    