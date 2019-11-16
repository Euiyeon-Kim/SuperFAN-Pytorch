import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, scale_by=4):
        super(Generator, self).__init__()
        self.scale_by = scale_by

        self.input_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU()
        )

        self.chunck_1 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),

            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),

            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.chunck_2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.deconv_12_3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        )

        self.deconv_3_2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.final = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = x.float()
        x = x.cuda()
        _in = self.input_block(x)
        chunck1 = self.chunck_1(_in)
        deconv1 = self.deconv_12_3(chunck1 + _in)
        chunck2 = self.chunck_2(deconv1)
        deconv2 = self.deconv_3_2(chunck2 + deconv1)
        block_2 = self.block_2(deconv2)
        final = self.final(block_2)

        return final


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.block0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.blocks = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.final_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Linear(512, 1024),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2),
            # nn.Linear(1024, 1)
            nn.Conv2d(1024, 1, 1)
        )

    def forward(self, x):
        x = x.float()
        x = x.cuda()
        block0 = self.block0(x)
        blocks = self.blocks(block0)
        final_block = self.final_block(blocks)

        return final_block


class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.bn2(res)

        return x + res

class upsample(nn.Module):
    def __init__(self):
        super(upsample, self).__init__()

        self.upsample = torch.nn.UpsamplingBilinear2d([256, 256])

    def forward(self, input):
        return (self.upsample(input) + 1.) / 2
