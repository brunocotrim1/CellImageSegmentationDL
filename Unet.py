import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_class):
        super(UNet, self).__init__()
        self.n_class = n_class
        # Encoder
        self.ee11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 3x256x256 -> 64x256x256
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 64x256x256 -> 64x256x256
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x256x256 -> 64x128x128

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64x128x128 -> 128x128x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 128x128x128 -> 128x128x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x128x128 -> 128x64x64
        
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 128x64x64 -> 256x64x64
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 256x64x64 -> 256x64x64
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256x64x64 -> 256x32x32

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 256x32x32 -> 512x32x32
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # 512x32x32 -> 512x32x32
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512x32x32 -> 512x16x16

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # 512x16x16 -> 1024x16x16
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)  # 1024x16x16 -> 1024x16x16

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 1024x16x16 -> 512x32x32
        self.d41 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)  # 1024x32x32 -> 512x32x32
        self.d42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # 512x32x32 -> 512x32x32

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 512x32x32 -> 256x64x64
        self.d31 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # 512x64x64 -> 256x64x64
        self.d32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 256x64x64 -> 256x64x64

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 256x64x64 -> 128x128x128
        self.d21 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 256x128x128 -> 128x128x128
        self.d22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 128x128x128 -> 128x128x128

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 128x128x128 -> 64x256x256
        self.d11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 128x256x256 -> 64x256x256
        self.d12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 64x256x256 -> 64x256x256

        self.conv1x1 = nn.Conv2d(64, n_class, kernel_size=1)  # 64x256x256 -> n_classx256x256
        

    def forward(self,x):
        xe11 = F.relu(self.ee11(x))
        xe12 = F.relu(self.e12(xe11))
        ep1 = self.pool1(xe12)

        xe21 = F.relu(self.e21(ep1))
        xe22 = F.relu(self.e22(xe21))
        ep2 = self.pool2(xe22)

        xe31 = F.relu(self.e31(ep2))
        xe32 = F.relu(self.e32(xe31))
        ep3 = self.pool3(xe32)

        xe41 = F.relu(self.e41(ep3))
        xe42 = F.relu(self.e42(xe41))
        ep4 = self.pool4(xe42)

        xe51 = F.relu(self.e51(ep4))
        xe52 = F.relu(self.e52(xe51))
        up4 = self.upconv4(xe52)
        cat4 = torch.cat((xe42, up4), dim=1)
        xd41 = F.relu(self.d41(cat4))
        xd42 = F.relu(self.d42(xd41))

        up3 = self.upconv3(xd42)
        cat3 = torch.cat((xe32, up3), dim=1)
        xd31 = F.relu(self.d31(cat3))
        xd32 = F.relu(self.d32(xd31))

        up2 = self.upconv2(xd32)
        cat2 = torch.cat((xe22, up2), dim=1)
        xd21 = F.relu(self.d21(cat2))
        xd22 = F.relu(self.d22(xd21))

        up1 = self.upconv1(xd22)
        cat1 = torch.cat((xe12, up1), dim=1)
        xd11 = F.relu(self.d11(cat1))
        xd12 = F.relu(self.d12(xd11))

        out = self.conv1x1(xd12)
        return  out




