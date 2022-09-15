# @Time    : 2022/9/14 20:10
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : build_model.py
# @Software: PyCharm
from model.MLCL.mlcl_net import MLCLNet
from model.SwinTransformer.SwinTransformer import Swin_Uper
from model.SINet.cod import SINet
from model.P_Swin_ACM.sacm import S_ACM
from model.Swin_Unet.vision_transformer import SwinUnet
from model.P_RSNet.rsnet import RSNet
from model.ResNet.resnet import Res
from model.AGPCNet.apgc import AGPCNet
from model.DNANet.dna_net import DNANet, Res_CBAM_block
from model.SegFormer.segformer import Segformer
from model.ConvSwin.convswin import ConvSwin
from model.DSTransUNet.DSTransUNet import DSTransUNet
from model.URANet.uranet import URANet


def build_model(model_name):
    if model_name == 'MLCL':
        model = MLCLNet()
    elif model_name == 'SwinTransformer':
        model = Swin_Uper()
    elif model_name == 'SINet':
        model = SINet()
    elif model_name == 'Swin_Unet':
        model = SwinUnet()
    elif model_name == 'ConvSwin':
        model = ConvSwin()
    elif model_name == 'DSTransUNet':
        model = DSTransUNet()
    elif model_name == 'URANet':
        model = URANet()
    elif model_name == 'SegFormer':
        model = Segformer(dims=(64, 128, 320, 512), num_layers=(3, 3, 18, 3))
    elif model_name == 'AGPCNet':
        model = AGPCNet(backbone='resnet18', scales=(10, 6, 5, 3), reduce_ratios=(16, 4), gca_type='patch',
                             gca_att='post', drop=0.1)
    elif model_name == 'DNANet':
        model = DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block, num_blocks=[2, 2, 2, 2],
                            nb_filter=[16, 32, 64, 128, 256])
    else:
        assert 'Model: ' + model_name + ' is not defined'
    return model
