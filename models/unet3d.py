import torch
import torch.nn as nn
from .common_blocks import DoubleConv3d, Conv3dBNReLU
import torch.nn.functional as F

class UNet3D(nn.Module):
    """
    3D U-Net for BraTS Brain Tumor Segmentation
    Input: (batch_size, 4, 128, 128, 128) - [T1, T1ce, T2, FLAIR]
    Output: (batch_size, 3, 128, 128, 128) - [WT, TC, ET]
    """

    def __init__(self, in_channels=4, out_channels=3, features=[32, 64, 128, 256, 512]):
        super(UNet3D, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder (Contracting Path)
        for feature in features:
            self.encoder.append(DoubleConv3d(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv3d(features[-1], features[-1] * 2)

        # Decoder (Expansive Path)
        features_reversed = features[::-1]
        for i, feature in enumerate(features_reversed):
            self.decoder.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            if i < len(features_reversed) - 1:
                self.decoder.append(DoubleConv3d(feature * 2, feature))
            else:
                self.decoder.append(DoubleConv3d(feature * 2, feature))

        # Final convolution
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Transpose conv
            skip_connection = skip_connections[idx // 2]

            # Ensure spatial dimensions match (handle odd dimensions)
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        return self.final_conv(x)


class UNet3D_Simple(nn.Module):
    """
    Simplified 3D U-Net for faster training
    """

    def __init__(self, in_channels=4, out_channels=3, features=[16, 32, 64, 128]):
        super(UNet3D_Simple, self).__init__()

        # Encoder
        self.enc1 = DoubleConv3d(in_channels, features[0])
        self.enc2 = DoubleConv3d(features[0], features[1])
        self.enc3 = DoubleConv3d(features[1], features[2])
        self.enc4 = DoubleConv3d(features[2], features[3])

        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = DoubleConv3d(features[3], features[3] * 2)

        # Decoder
        self.up4 = nn.ConvTranspose3d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv3d(features[3] * 2, features[3])

        self.up3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv3d(features[2] * 2, features[2])

        self.up2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv3d(features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv3d(features[0] * 2, features[0])

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)