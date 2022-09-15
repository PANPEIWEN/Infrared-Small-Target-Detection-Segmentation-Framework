# @Time    : 2022/6/8 10:20
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : DSTransUNet.py
# @Software: PyCharm
import torch
import torch.nn as nn
import einops
import numpy as np
from model.DSTransUNet.swin import SwinTransformer, Swin
from model.DSTransUNet.vit import ViT


class TIF(nn.Module):
    def __init__(self, f1, f2, dims):
        super(TIF, self).__init__()
        self.pools1 = nn.ModuleList([
            nn.AvgPool1d(kernel_size=i)
            for i in f1])
        self.pools2 = nn.ModuleList([
            nn.AvgPool1d(kernel_size=i)
            for i in f2])
        self.trans1 = nn.ModuleList([
            ViT(dim=i + 1)
            for i in f1])
        self.trans2 = nn.ModuleList([
            ViT(dim=i + 1)
            for i in f2])
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            )
            for dim in dims])
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x1, x2):
        outs = []
        for i, x in enumerate(zip(x1, x2)):
            h1, h2 = x1[i].size(2), x2[i].size(2)
            x1[i] = einops.rearrange(x1[i], 'b c h w -> b c (h w)')
            x2[i] = einops.rearrange(x2[i], 'b c h w -> b c (h w)')
            t_x1, t_x2 = x1[i], x2[i]
            t_x1 = self.pools1[i](t_x1)
            t1 = torch.cat((t_x1, x2[i]), dim=2)
            out1 = self.trans2[i](t1)
            t_x2 = self.pools2[i](t_x2)
            t2 = torch.cat((t_x2, x1[i]), dim=2)
            out2 = self.trans1[i](t2)
            out1 = einops.rearrange(out1, 'b c (h w) -> b c h w', h=h2)
            out2 = einops.rearrange(out2, 'b c (h w) -> b c h w', h=h1)
            out1 = self.up(out1)
            out = torch.cat((out1, out2), dim=1)
            out = self.convs[i](out)
            outs.append(out)
        return outs


class DSTransUNet(nn.Module):
    def __init__(self):
        super(DSTransUNet, self).__init__()
        self.primary_swin = Swin()
        self.complement_swin = SwinTransformer(patch_size=8, depths=[2, 2, 6, 2])
        self.tif = TIF(f1=[4096, 1024, 256, 64], f2=[1024, 256, 64, 16], dims=[128, 256, 512, 1024])
        self.swins = nn.ModuleList([
            SwinTransformer(patch_size=1, depths=[2], out_indices=[0], in_chans=channel, embed_dim=embed)
            for channel, embed in zip([256, 512, 1024], [128, 256, 512])])
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pool_conv = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )
            for in_channel, out_channel in zip([3, 64], [64, 128])])
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        out1 = self.primary_swin(x)
        out2 = self.complement_swin(x)
        outs = self.tif(out1, out2)
        end_outs = []
        end_outs.append(outs[3])
        out = outs[3]
        for i in reversed(range(len(outs) - 1)):
            out = self.up(out)
            s_out = self.swins[i](out)
            out = s_out[0] + outs[i]
            end_outs.append(out)
        x1 = self.pool_conv[0](x)
        x2 = self.pool_conv[1](x1)
        x2 = x2 + end_outs[3]
        x2 = self.conv(self.up(x2))
        x1 = x1 + x2
        x1 = self.head(x1)
        x = self.up(x1)
        return x


if __name__ == '__main__':
    x = torch.rand(8, 3, 256, 256)
    model = DSTransUNet()
    outs = model(x)
    print(outs.size())
