import torch.nn as nn
import torch

class RepeatedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.repeatedconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding=1, kernel_size= 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace= True),
            nn.Conv2d(out_channels, out_channels, padding= 1, kernel_size= 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace= True)
        )

    def forward(self, x):
        return self.repeatedconv(x)

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.rc = RepeatedConv(in_channels= in_channels, out_channels=64)
        # Contracting Path
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride= 2)
        self.down1 = nn.Sequential(self.maxpool, RepeatedConv(in_channels= 64, out_channels= 128))
        self.down2 = nn.Sequential(self.maxpool, RepeatedConv(in_channels= 128, out_channels= 256))
        self.down3 = nn.Sequential(self.maxpool, RepeatedConv(in_channels=256, out_channels= 512))

        # Bottleneck layer
        self.down4 = RepeatedConv(in_channels=512, out_channels= 1024)

        # Expansive Path

        self.up1 = nn.ConvTranspose2d(in_channels= 1024, out_channels= 512, kernel_size=2)
        self.rc1 = RepeatedConv(in_channels= 1024, out_channels=512)

        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2)
        self.rc2 = RepeatedConv(in_channels= 512, out_channels=256)

        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2)
        self.rc3 = RepeatedConv(in_channels= 256, out_channels=128)

        self.up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2)
        self.rc4 = RepeatedConv(in_channels= 128, out_channels=64)

        self.final_layer = nn.Conv2d(in_channels= 64, out_channels= out_channels, kernel_size= 1)

    def forward(self, x):
        out = self.rc(x)
        out1 = self.down1(out)
        out2 = self.down2(out1)
        out3 = self.down3(out2)

        # Bottleneck layer
        out4 = self.down4(out3)

        out5 = self.up1(out4)
        concat5 = torch.cat([out5, out3], dim= 1)
        rc5 = self.rc1(concat5)

        out6 = self.up2(rc5)
        concat6 = torch.cat([out6, out2], dim= 1)
        rc6 = self.rc2(concat6)

        out7 = self.up3(rc6)
        concat7 = torch.cat([out7, out1], dim= 1)
        rc7 = self.rc3(concat7)

        out8= self.up4(rc7)
        concat8 = torch.cat([out8, out], dim= 1)
        rc8 = self.rc4(concat8)

        return self.final_layer(rc8)



        

