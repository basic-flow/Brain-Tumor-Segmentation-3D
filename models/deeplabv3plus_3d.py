import torch
import torch.nn as nn
import torch.nn.functional as F
from .common_blocks import Conv3dBNReLU, ASPP3D, ResidualBlock3D


class DeepLabV3Plus3D(nn.Module):
    """
    3D DeepLabV3+ for BraTS Brain Tumor Segmentation
    """

    def __init__(self, in_channels=4, out_channels=3, backbone_channels=64, aspp_channels=256):
        super(DeepLabV3Plus3D, self).__init__()

        # Initial convolution
        self.initial_conv = Conv3dBNReLU(in_channels, backbone_channels, kernel_size=7, stride=2, padding=3)
        self.initial_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Backbone with residual blocks
        self.layer1 = self._make_layer(backbone_channels, backbone_channels, 2, stride=1)
        self.layer2 = self._make_layer(backbone_channels, backbone_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(backbone_channels * 2, backbone_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(backbone_channels * 4, backbone_channels * 8, 2, stride=2)

        # ASPP module
        self.aspp = ASPP3D(backbone_channels * 8, aspp_channels)

        # Low-level features (from layer1)
        self.low_level_conv = Conv3dBNReLU(backbone_channels, 48, kernel_size=1, padding=0)

        # Decoder
        self.decoder = nn.Sequential(
            Conv3dBNReLU(aspp_channels + 48, 256, kernel_size=3, padding=1),
            Conv3dBNReLU(256, 256, kernel_size=3, padding=1),
            nn.Dropout3d(0.1),
            nn.Conv3d(256, out_channels, kernel_size=1)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock3D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial features
        x_initial = self.initial_conv(x)
        x_pool = self.initial_pool(x_initial)

        # Backbone features
        x_low = self.layer1(x_pool)  # Low-level features
        x2 = self.layer2(x_low)
        x3 = self.layer3(x2)
        x_high = self.layer4(x3)  # High-level features

        # ASPP
        x_aspp = self.aspp(x_high)
        x_aspp = F.interpolate(x_aspp, scale_factor=4, mode='trilinear', align_corners=True)

        # Low-level features
        x_low_level = self.low_level_conv(x_low)
        x_low_level = F.interpolate(x_low_level, scale_factor=2, mode='trilinear', align_corners=True)

        # Ensure spatial dimensions match
        if x_aspp.shape[2:] != x_low_level.shape[2:]:
            x_aspp = F.interpolate(x_aspp, size=x_low_level.shape[2:], mode='trilinear', align_corners=True)

        # Concatenate and decode
        x_cat = torch.cat([x_aspp, x_low_level], dim=1)
        x_out = self.decoder(x_cat)

        # Final upsampling to input size
        x_out = F.interpolate(x_out, scale_factor=4, mode='trilinear', align_corners=True)

        return x_out


class DeepLabV3Plus3D_Simple(nn.Module):
    """
    Simplified DeepLabV3+ for faster training
    """

    def __init__(self, in_channels=4, out_channels=3, features=32):
        super(DeepLabV3Plus3D_Simple, self).__init__()

        # Smaller backbone
        self.encoder1 = Conv3dBNReLU(in_channels, features, kernel_size=3, stride=2, padding=1)
        self.encoder2 = Conv3dBNReLU(features, features * 2, kernel_size=3, stride=2, padding=1)
        self.encoder3 = Conv3dBNReLU(features * 2, features * 4, kernel_size=3, stride=2, padding=1)
        self.encoder4 = Conv3dBNReLU(features * 4, features * 8, kernel_size=3, stride=2, padding=1)

        # Simplified ASPP
        self.aspp = nn.Sequential(
            Conv3dBNReLU(features * 8, features * 4, kernel_size=1, padding=0),
            Conv3dBNReLU(features * 4, features * 4, kernel_size=3, padding=2, dilation=2),
            Conv3dBNReLU(features * 4, features * 4, kernel_size=3, padding=4, dilation=4),
        )

        # Low-level features
        self.low_level_conv = Conv3dBNReLU(features, features // 2, kernel_size=1, padding=0)

        # Decoder
        self.decoder = nn.Sequential(
            Conv3dBNReLU(features * 4 + features // 2, features * 2, kernel_size=3, padding=1),
            Conv3dBNReLU(features * 2, features * 2, kernel_size=3, padding=1),
            nn.Conv3d(features * 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        x_low = self.encoder1(x)  # Low-level features
        x2 = self.encoder2(x_low)
        x3 = self.encoder3(x2)
        x_high = self.encoder4(x3)  # High-level features

        # ASPP
        x_aspp = self.aspp(x_high)
        x_aspp = F.interpolate(x_aspp, scale_factor=8, mode='trilinear', align_corners=True)

        # Low-level features
        x_low_level = self.low_level_conv(x_low)
        x_low_level = F.interpolate(x_low_level, scale_factor=2, mode='trilinear', align_corners=True)

        # Ensure spatial dimensions match
        if x_aspp.shape[2:] != x_low_level.shape[2:]:
            x_aspp = F.interpolate(x_aspp, size=x_low_level.shape[2:], mode='trilinear', align_corners=True)

        # Concatenate and decode
        x_cat = torch.cat([x_aspp, x_low_level], dim=1)
        x_out = self.decoder(x_cat)

        # Final upsampling
        x_out = F.interpolate(x_out, scale_factor=2, mode='trilinear', align_corners=True)

        return x_out

