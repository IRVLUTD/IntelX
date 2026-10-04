# flexible_unet.py
"""
Flexible U-Net implementation with full architectural control
Supports various encoder backbones including ResNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connections"""

    def __init__(self, in_channels, skip_channels, out_channels,
                 upsampling_type='bilinear', skip_type='concat'):
        super().__init__()
        self.upsampling_type = upsampling_type
        self.skip_type = skip_type

        # Upsampling
        if upsampling_type == 'bilinear':
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upsample_out_channels = in_channels
        elif upsampling_type == 'transposed_conv':
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            upsample_out_channels = in_channels
        else:
            raise ValueError(f"Unknown upsampling type: {upsampling_type}")

        # Skip connection processing
        if skip_type == 'concat':
            conv_in_channels = upsample_out_channels + skip_channels
        elif skip_type == 'add':
            conv_in_channels = upsample_out_channels
            # Project skip to match upsampled channels if needed
            if skip_channels != upsample_out_channels:
                self.skip_proj = nn.Conv2d(skip_channels, upsample_out_channels, 1)
            else:
                self.skip_proj = None
        else:
            raise ValueError(f"Unknown skip type: {skip_type}")

        # Convolution after merge
        self.conv = ConvBlock(conv_in_channels, out_channels)

    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)

        # Handle size mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)

        # Merge with skip connection
        if self.skip_type == 'concat':
            x = torch.cat([x, skip], dim=1)
        elif self.skip_type == 'add':
            if self.skip_proj is not None:
                skip = self.skip_proj(skip)
            x = x + skip

        # Convolution
        x = self.conv(x)
        return x


class ResNetEncoder(nn.Module):
    """ResNet encoder with configurable depth"""

    def __init__(self, backbone='resnet34', pretrained=False, in_channels=1):
        super().__init__()

        # Load backbone
        if backbone == 'resnet18':
            resnet = models.resnet18(weights=None)
            self.out_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights=None)
            self.out_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights=None)
            self.out_channels = [64, 256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            resnet = models.resnet101(weights=None)
            self.out_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Modify first conv for different input channels
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
        else:
            self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        features = []

        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # 1/2 resolution

        x = self.maxpool(x)

        # ResNet blocks
        x = self.layer1(x)
        features.append(x)  # 1/4 resolution

        x = self.layer2(x)
        features.append(x)  # 1/8 resolution

        x = self.layer3(x)
        features.append(x)  # 1/16 resolution

        x = self.layer4(x)
        features.append(x)  # 1/32 resolution

        return features


class FlexibleUNet(nn.Module):
    def __init__(
            self,
            encoder='resnet34',
            encoder_pretrained=False,
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=1,
            out_channels=1,
            upsampling_type='bilinear',
            skip_type='concat',
            center_block=False,
            activation='sigmoid'
    ):
        super().__init__()

        self.encoder_depth = encoder_depth
        self.activation = activation

        # Encoder
        self.encoder = ResNetEncoder(encoder, encoder_pretrained, in_channels)
        encoder_channels = self.encoder.out_channels[:encoder_depth]

        # Optional center block
        if center_block:
            self.center = ConvBlock(encoder_channels[-1], encoder_channels[-1])
        else:
            self.center = None

        # Decoder
        self.decoder_blocks = nn.ModuleList()

        in_ch = encoder_channels[-1]
        for i in range(encoder_depth - 1):
            skip_ch = encoder_channels[-(i + 2)]
            out_ch = decoder_channels[i] if i < len(decoder_channels) else skip_ch

            self.decoder_blocks.append(
                UpBlock(in_ch, skip_ch, out_ch, upsampling_type, skip_type)
            )
            in_ch = out_ch

        # Final conv
        self.final_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)

        # FIX: Add final upsampling to match original input size
        # ResNet encoder has initial stride=2 conv + maxpool = 4x downsampling before blocks
        # So we need 2x upsampling at the end (2^1 = 2, since we've upsampled encoder_depth-1 times)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Activation
        if activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.final_activation = nn.Tanh()
        elif activation is None:
            self.final_activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        # Encoder
        encoder_features = self.encoder(x)
        encoder_features = encoder_features[:self.encoder_depth]

        # Center
        x = encoder_features[-1]
        if self.center is not None:
            x = self.center(x)

        # Decoder
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = encoder_features[-(i + 2)]
            x = decoder_block(x, skip)

        # Final conv
        x = self.final_conv(x)

        # FIX: Final upsample to match input size
        x = self.final_upsample(x)

        # Activation
        x = self.final_activation(x)

        return x

def create_unet_resnet18(in_channels=1, out_channels=1):
    """Lightweight U-Net with ResNet18 encoder"""
    return FlexibleUNet(
        encoder='resnet18',
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
        in_channels=in_channels,
        out_channels=out_channels
    )


def create_unet_resnet34(in_channels=1, out_channels=1):
    """Balanced U-Net with ResNet34 encoder (RECOMMENDED)"""
    return FlexibleUNet(
        encoder='resnet34',
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
        in_channels=in_channels,
        out_channels=out_channels
    )


def create_unet_resnet50(in_channels=1, out_channels=1):
    """Larger U-Net with ResNet50 encoder"""
    return FlexibleUNet(
        encoder='resnet50',
        encoder_depth=5,
        decoder_channels=(512, 256, 128, 64, 32),
        in_channels=in_channels,
        out_channels=out_channels
    )


def create_unet_custom(
        encoder='resnet34',
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
        **kwargs
):
    """Create custom U-Net with full control"""
    return FlexibleUNet(
        encoder=encoder,
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
        **kwargs
    )


if __name__ == '__main__':
    # Test model creation
    model = create_unet_resnet34()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    x = torch.randn(2, 1, 400, 400)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
