import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3dBNReLU(nn.Sequential):
    """3D Convolution + InstanceNorm + ReLU - works with any batch size"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(Conv3dBNReLU, self).__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.InstanceNorm3d(out_channels),  # Use InstanceNorm3d for 3D data
            nn.ReLU(inplace=True)
        )

class DoubleConv3d(nn.Sequential):
    """Double 3D convolution block"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv3d, self).__init__(
            Conv3dBNReLU(in_channels, mid_channels),
            Conv3dBNReLU(mid_channels, out_channels)
        )


class ASPP3D(nn.Module):
    """3D Atrous Spatial Pyramid Pooling for DeepLabV3+"""

    def __init__(self, in_channels, out_channels, rates=[1, 2, 4, 6]):
        super(ASPP3D, self).__init__()

        # 1x1x1 convolution
        self.conv1x1 = Conv3dBNReLU(in_channels, out_channels, 1, padding=0)

        # 3x3x3 convolutions with different dilation rates
        self.conv3x3_1 = Conv3dBNReLU(in_channels, out_channels, 3, padding=rates[0], dilation=rates[0])
        self.conv3x3_2 = Conv3dBNReLU(in_channels, out_channels, 3, padding=rates[1], dilation=rates[1])
        self.conv3x3_3 = Conv3dBNReLU(in_channels, out_channels, 3, padding=rates[2], dilation=rates[2])

        # Global average pooling - FIXED: Use simple conv + relu without normalization
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=True),  # Use bias instead of norm
            nn.ReLU(inplace=True)
        )

        self.conv1x1_out = Conv3dBNReLU(out_channels * 5, out_channels, 1, padding=0)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)

        # Global average pooling
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='trilinear', align_corners=True)

        # Concatenate all features
        x_out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.conv1x1_out(x_out)


class ResidualBlock3D(nn.Module):
    """Simple 3D Residual Block for the backbone"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = Conv3dBNReLU(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels)  # InstanceNorm without ReLU for second conv
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv3dBNReLU(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        return F.relu(out)