import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from functools import partial
import math
from thop import profile

# Unet start

# Unet编码器 #
class DoubleConv(nn.Module):
    """(convolution => [InstanceNorm] => LeakyReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
# Unet end

# three branches Unet  start
class LocalBlock(nn.Module):
    def __init__(self, dim):
        super(LocalBlock, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.In = nn.InstanceNorm2d(dim)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        input = x
        x1 = self.dwconv(x)
        x2 = self.In(x1)
        x3 = self.relu(x2)

        x4 = self.conv(x3)
        x5 = self.In(x4)
        x6 = self.relu(x5)

        out = x6 + input
        return out

class GlobalBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # self.act2 = nn.GELU()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # x = self.act2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + x
        return x


class CombineBlock(nn.Module):
    def __init__(self, dim1, dim2):
        super(CombineBlock, self).__init__()
        self.pwconv1 = nn.Conv2d(dim1, dim1, kernel_size=1, padding=0)
        self.pwconv2 = nn.Conv2d(dim2, dim2, kernel_size=1, padding=0)
        self.In = nn.InstanceNorm2d(dim1+dim2)
        self.relu = nn.LeakyReLU(inplace=True)
        self.dwconv = nn.Conv2d(dim1+dim2, dim1+dim2, kernel_size=3, padding=1, groups=dim1+dim2)

        self.fmask1 = nn.Sequential(
            nn.Conv2d(dim1+dim2, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.fmask2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(dim1+dim2, dim1+dim2, kernel_size=3, padding=1, groups=dim1+dim2),
            nn.InstanceNorm2d(dim1+dim2),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x1, x2, mask):
        x11 = self.pwconv1(x1)
        x22 = self.pwconv2(x2)
        x3 = torch.cat((x11, x22), dim=1)
        x4 = self.dwconv(x3)
        x5 = self.In(x4)
        x6 = self.relu(x5)

        fmask = (self.fmask1(x6) > 0.8).type(torch.cuda.FloatTensor)
        m = (self.fmask2(mask) > 0.8).type(torch.cuda.FloatTensor)
        # x7 = x6 * torch.logical_or(fmask, m).type(torch.cuda.FloatTensor)
        x7 = x6 + torch.logical_or(fmask, m).type(torch.cuda.FloatTensor)
        x8 = self.conv(x7)
        # x9 = x6 + x8
        # x = self.conv(x9)

        return x8


class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            LayerNorm(in_dim, eps=1e-6,  data_format="channels_first"),
            nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel//2, 1, 1)
        self.pwconv = nn.Conv2d(channel//2, channel//2, kernel_size=1, padding=0)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv = nn.Conv2d(channel//2, channel//2, kernel_size=3, padding=1, groups=channel//2)

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        out1 = self.layer(up)
        skip1 = self.conv(feature_map)
        skip2 = self.relu(skip1)
        skip3 = self.relu(self.pwconv(skip2))
        skip4 = skip3 + feature_map
        out2 = torch.cat((out1, skip4), dim=1)
        out3 = self.layer(out2)
        return out3

class Out(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Out, self).__init__()
        self.up = nn.ConvTranspose2d(in_dim, in_dim//2, 4, 4)
        self.layer = nn.Conv2d(in_dim//2, out_dim, 1, 1)

    def forward(self, x):
        out = self.up(x)
        return self.layer(out)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class TBUNET(nn.Module):
    def __init__(self, in_chans=3, num_classes=1, dims=[96, 192, 384, 768]):
        super().__init__()
        self.mask = UNet(in_chans, num_classes, True)
        self.pool = nn.MaxPool2d(2)

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        self.t1 = nn.Conv2d(dims[0], dims[0]//4, kernel_size=3, padding=1)
        self.t11 = nn.Conv2d(dims[0], dims[0]*3//4, kernel_size=3, padding=1)
        self.c1 = GlobalBlock(dims[0]//4)
        self.c11 = LocalBlock(dims[0]*3//4)
        self.c111 = CombineBlock(dims[0]//4, dims[0]*3//4)
        self.d1 = DownSample(dims[0], dims[1])

        self.t2 = nn.Conv2d(dims[1], dims[1] // 2, kernel_size=3, padding=1)
        self.t22 = nn.Conv2d(dims[1], dims[1] // 2, kernel_size=3, padding=1)
        self.c2 = GlobalBlock(dims[1] // 2)
        self.c22 = LocalBlock(dims[1] // 2)
        self.c222 = CombineBlock(dims[1] // 2, dims[1] // 2)
        self.d2 = DownSample(dims[1], dims[2])

        self.t3 = nn.Conv2d(dims[2], dims[2]*3//4, kernel_size=3, padding=1)
        self.t33 = nn.Conv2d(dims[2], dims[2]//4, kernel_size=3, padding=1)
        self.c3 = GlobalBlock(dims[2]*3//4)
        self.c33 = LocalBlock(dims[2]//4)
        self.c333 = CombineBlock(dims[2]*3//4, dims[2]//4)
        self.d3 = DownSample(dims[2], dims[3])

        self.t4 = nn.Conv2d(dims[3], dims[3] * 11 // 12, kernel_size=3, padding=1)
        self.t44 = nn.Conv2d(dims[3], dims[3] // 12, kernel_size=3, padding=1)
        self.c4 = GlobalBlock(dims[3] * 11 // 12)
        self.c44 = LocalBlock(dims[3] // 12)
        self.c444 = CombineBlock(dims[3] * 11 // 12, dims[3] // 12)
        self.u1 = UpSample(dims[3])

        self.c5 = LocalBlock(dims[2])
        self.u2 = UpSample(dims[2])

        self.c6 = LocalBlock(dims[1])
        self.u3 = UpSample(dims[1])

        self.c7 = LocalBlock(dims[0])
        self.out = Out(dims[0], num_classes)

    def forward(self, x):
        m = self.mask(x)
        m1 = self.pool(self.pool(m))
        R1 = self.stem(x)
        R2 = self.c1(self.t1(R1))
        R22 = self.c11(self.t11(R1))
        R222 = self.c111(R2, R22, m1)
        R3 = self.d1(R222)

        m2 = self.pool(m1)
        R4 = self.c2(self.t2(R3))
        R44 = self.c22(self.t22(R3))
        R444 = self.c222(R4, R44, m2)
        R5 = self.d2(R444)

        m3 = self.pool(m2)
        R6 = self.c3(self.t3(R5))
        R66 = self.c33(self.t33(R5))
        R666 = self.c333(R6, R66, m3)
        R7 = self.d3(R666)

        m4 = self.pool(m3)
        R8 = self.c4(self.t4(R7))
        R88 = self.c44(self.t44(R7))
        R888 = self.c444(R8, R88, m4)
        O1 = self.u1(R888, R666)

        O2 = self.c5(self.c5(O1))
        O3 = self.u2(O2, R444)

        O4 = self.c6(self.c6(O3))
        O5 = self.u3(O4, R222)

        O6 = self.c7(self.c7(O5))
        O7 = self.out(O6)

        return O7

# if __name__ == '__main__':      # 算法效率测试 #
#     x = torch.randn(1, 3, 512, 512).cuda()
#     # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     # x = torch.tensor(x, device=device)
#     net = UNet().cuda()
#     net(x)
#     flops, params = profile(net, inputs=(x,))
#     print(flops/1e9, params/1e6)


