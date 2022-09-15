# @Time    : 2022/6/10 16:13
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : uranet.py
# @Software: PyCharm
from audioop import bias
import torch
import torch.nn as nn
import torch.nn.functional as F


class CDC_conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, kernel_size=3, padding=1, dilation=1, theta=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                              bias=bias)
        self.theta = theta

    def forward(self, x):
        norm_out = self.conv(x)
        [c_out, c_in, kernel_size, kernel_size] = self.conv.weight.shape
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        diff_out = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0)
        out = norm_out - self.theta * diff_out
        return out


class Layernorm(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_c)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.BatchNorm2d):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
            nn.ReLU(inplace=True),
            CDC_conv(out_c, out_c, kernel_size=3, padding=1, bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
        )
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv_out = self.conv_block(x)
        residual_out = self.residual_block(x)
        out = self.relu(conv_out + residual_out)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, out_c, bilinear=True):
        super().__init__()
        if bilinear:
            self.up_block = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        else:
            self.up_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=2, padding=1,
                                   bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

    def forward(self, x, lateral):
        size = lateral.shape[-2:]
        x = self.up_block(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        out = x + lateral
        return out


class Position_attention(nn.Module):
    def __init__(self, in_c, mid_c=None):
        super().__init__()
        mid_c = mid_c or in_c // 8
        self.q = nn.Conv2d(in_c, mid_c, kernel_size=1)
        self.k = nn.Conv2d(in_c, mid_c, kernel_size=1)
        self.v = nn.Conv2d(in_c, in_c, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, _, h, w = x.shape
        q = self.q(x).view(b, -1, h * w).permute(0, 2, 1)  # bs, hw, c
        k = self.k(x).view(b, -1, h * w)  # bs, c ,hw
        v = self.v(x).view(b, -1, h * w)  # bs, c, hw
        att = self.softmax(q @ k)
        out = (v @ att.permute(0, 2, 1)).view(b, -1, h, w)
        out = self.gamma * out + x

        return out


class Channel_attention(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.in_c = in_c
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.shape
        q = x.view(b, -1, h * w)  # bs, c ,hw
        k = x.view(b, -1, h * w).permute(0, 2, 1)  # bs, hw, c
        v = x.view(b, -1, h * w)  # bs, c, hw
        att = self.softmax(q @ k)  # b, c, c
        out = att @ v
        out = out.view(b, -1, h, w)
        out = self.gamma * out + x
        return out


class Double_attention(nn.Module):
    def __init__(self, in_c, mid_c=None):
        super().__init__()
        self.pam = Position_attention(in_c, mid_c)
        self.cam = Channel_attention(in_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        pam_out = self.pam(x)
        cam_out = self.cam(x)
        return pam_out + cam_out


class URANet(nn.Module):
    def __init__(self, in_c=3, base_dim=32, class_num=1, bilinear=True, use_da=True, norm=nn.BatchNorm2d):
        super(URANet, self).__init__()
        self.norm = norm
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_c, base_dim, kernel_size=3, padding=1, bias=False),
            CDC_conv(in_c, base_dim, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            # nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, bias=False),
            CDC_conv(base_dim, base_dim, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(base_dim, base_dim * 2, norm=self.norm)
        self.layer2 = ResidualBlock(base_dim * 2, base_dim * 4, self.norm)
        self.layer3 = ResidualBlock(base_dim * 4, base_dim * 8, self.norm)
        self.layer4 = ResidualBlock(base_dim * 8, base_dim * 16, self.norm)
        self.da = Double_attention(base_dim * 16, None) if use_da else nn.Identity()
        self.up3 = UpsampleBlock(base_dim * 16, base_dim * 8, bilinear=bilinear)
        self.up2 = UpsampleBlock(base_dim * 8, base_dim * 4, bilinear=bilinear)
        self.up1 = UpsampleBlock(base_dim * 4, base_dim * 2, bilinear=bilinear)
        self.up0 = UpsampleBlock(base_dim * 2, base_dim, bilinear=bilinear)
        self.last_conv = nn.Conv2d(base_dim, class_num, kernel_size=1, stride=1)

    def forward(self, x):
        out_0 = self.conv1(x)
        out_0 = self.conv2(out_0)
        out_1 = self.layer1(out_0)
        out_2 = self.layer2(self.maxpool(out_1))
        out_3 = self.layer3(self.maxpool(out_2))
        out_4 = self.layer4(self.maxpool(out_3))
        out_da = self.da(out_4)
        up_3 = self.up3(out_da, out_3)
        up_2 = self.up2(up_3, out_2)
        up_1 = self.up1(up_2, out_1)
        up_0 = self.up0(up_1, out_0)
        out = self.last_conv(up_0)
        return out


def main():
    x = torch.rand(2, 3, 512, 512)
    net = ura_net()
    out = net(x)
    print(out.shape)


if __name__ == '__main__':
    main()
