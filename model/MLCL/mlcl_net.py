# @Time    : 2022/4/5 21:13
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : mlcl_net.py
# @Software: PyCharm
import torch
import torch.nn as nn


class Resnet1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Resnet1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.layer(x)
        out += identity
        return self.relu(out)


class Resnet2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Resnet2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        identity = self.layer2(identity)
        out += identity
        return self.relu(out)


class Stage(nn.Module):
    def __init__(self):
        super(Stage, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.resnet1_1 = Resnet1(in_channel=16, out_channel=16)
        self.resnet2_1 = Resnet2(in_channel=16, out_channel=32)
        self.resnet2_2 = Resnet1(in_channel=32, out_channel=32)
        self.resnet3_1 = Resnet2(in_channel=32, out_channel=64)
        self.resnet3_2 = Resnet1(in_channel=64, out_channel=64)

    def forward(self, x):
        outs = []
        out = self.layer1(x)
        for i in range(3):
            out = self.resnet1_1(out)
        outs.append(out)
        out = self.resnet2_1(out)
        for i in range(2):
            out = self.resnet2_2(out)
        outs.append(out)
        out = self.resnet3_1(out)
        for i in range(2):
            out = self.resnet3_2(out)
        outs.append(out)
        return outs
    

class MLCL(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLCL, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=3, stride=1, dilation=3),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=5, stride=1, dilation=5),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(in_channels=in_channel * 3, out_channels=out_channel, kernel_size=1)

    def forward(self, x):
        x1 = x
        x2 = x
        x3 = x
        out1 = self.layer1(x1)
        out2 = self.layer2(x2)
        out3 = self.layer3(x3)
        outs = torch.cat((out1, out2, out3), dim=1)
        return self.conv(outs)


class MLCLNet(nn.Module):
    def __init__(self):
        super(MLCLNet, self).__init__()
        self.stage = Stage()
        self.mlcl1 = MLCL(64, 64)
        self.mlcl2 = MLCL(32, 32)
        self.mlcl3 = MLCL(16, 16)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
        )

    def forward(self, x):
        outs = self.stage(x)
        out1 = self.mlcl1(outs[2])
        out2 = self.mlcl2(outs[1])
        out3 = self.mlcl3(outs[0])
        out1 = self.up1(out1)
        out2 = self.conv1(out2)
        out = out1 + out2
        out = self.up2(out)
        out3 = self.conv2(out3)
        out = out + out3
        out = self.layer(out)
        return out


if __name__ == '__main__':
    model = MLCLNet()
    x = torch.rand(8, 3, 256, 256)
    outs = model(x)
    print(outs.size())
