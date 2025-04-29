import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Кодувальні шари (downsampling)
        self.encoder = nn.Sequential(
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
            self.conv_block(512, 1024)
        )

        # Декодувальні шари (upsampling)
        self.decoder = nn.Sequential(
            self.upconv_block(1024, 512),
            self.upconv_block(512, 256),
            self.upconv_block(256, 128),
            self.upconv_block(128, 64)
        )

        # Вихідний шар
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Прохід через кодувальні шари
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[1](enc1)
        enc3 = self.encoder[2](enc2)
        enc4 = self.encoder[3](enc3)
        enc5 = self.encoder[4](enc4)

        # Прохід через декодувальні шари
        dec4 = self.decoder[0](enc5)
        dec3 = self.decoder[1](dec4)
        dec2 = self.decoder[2](dec3)
        dec1 = self.decoder[3](dec2)

        # Переведемо до потрібного розміру (256x256)
        output = self.output_conv(dec1)

        # Якщо розмір не збігається з 256x256, зробимо фіксоване відновлення
        return F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
