# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:19:06 2024

@author: Mona
"""

import torch
from torch import nn
from torchvision import transforms
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d, ReLU, Sigmoid, UpsamplingBilinear2d
from torch import optim
from torch.nn import functional as F
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split

from .swin_v1 import SwinTransformer
import pytorch_grad_cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import argparse
import warnings


class ConvUnit(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 is_resblock: bool = False, pooling: str = None, padding='same', **args):
        '''
        is_resblock: 是否为残差结构
        pooling: 在第一个卷积过后的池化类型，若为None，则不加池化
        padding: Conv2d的参数，这里默认值为'same'不改变卷积前后图像大小
        '''
        super().__init__()
        self.is_resblock = is_resblock
        self.pooling = pooling
        stride = args.get('stride')
        
        if stride is not None and stride > 1 \
            and padding == 'same':
            padding = 0
            
        conv = [Conv2d(in_channels, out_channels, kernel_size, 
                                         padding=padding, **args),
                                  BatchNorm2d(out_channels), ReLU(),
                                  Conv2d(out_channels, out_channels, kernel_size, 
                                         padding=padding, **args),
                                  BatchNorm2d(out_channels), ReLU(),]
        
        if pooling == 'max':
            conv.insert(3, MaxPool2d(2,2))
            self.conv_x = nn.Sequential(Conv2d(in_channels, out_channels, kernel_size,
                                               stride=2, padding=(kernel_size-1)//2),
                                        BatchNorm2d(out_channels), ReLU())
        self.conv = nn.Sequential(*conv)
    def forward(self, X):
        
        if not self.is_resblock:
            return self.conv(X)
        out = self.conv[:-1](X)
        if self.pooling:
            X = self.conv_x(X)
        out = torch.add(out, X)
        out = self.conv[-1](out)
        return out
    
#%%
# 串联多个残差结构的卷积单元ConvUnit
def Res_Block(num_res_cell, in_channels, out_channels, kernel_size, pooling=None, **args):
    
    '''
    num_res_cell: 串联的卷积单元的个数
    '''
    
    blocks = [ConvUnit(out_channels, out_channels, 
                      kernel_size, is_resblock=True) 
             for i in range(num_res_cell-1)]
    
    if pooling:
        blocks.insert(0, ConvUnit(in_channels, out_channels, kernel_size, 
                      is_resblock=True, pooling=pooling))
    else:
        blocks.insert(0, ConvUnit(in_channels, out_channels, kernel_size, 
                                  is_resblock=True) )
    return nn.Sequential(*blocks)


# 1*1卷积
def PointWiseConv(in_channels, out_channels, activation='relu'):
    
    '''
    activation: 卷积后接的激活函数类型，默认为ReLU，可选为Sigmoid
    '''
    
    net = [Conv2d(in_channels, out_channels, 1),
           BatchNorm2d(out_channels)]
    if activation == 'relu':
        net.append(ReLU())
    elif activation == 'sigmoid':
        net.append(Sigmoid())
    
    return nn.Sequential(*net)
#%%

# LCL单元块儿，可用于构成MLCL块儿
def LCL_Unit(channels, kernel_size, dilation_rate):
    
    conv1 = Conv2d(channels, channels, kernel_size=kernel_size, padding='same')
    relu1 = ReLU()
    conv2 = Conv2d(channels, channels, kernel_size=3, padding='same', dilation=dilation_rate)
    relu2 = ReLU()
    return nn.Sequential(conv1, relu1, conv2, relu2)


class MLCBlock(nn.Module):
    
    def __init__(self, num_lcl, channels):
    
        super().__init__()
        self.num_lcl = num_lcl
        self.channels = channels
        self._define_model()
        
    def _define_model(self):
        
        layers = [LCL_Unit(self.channels, 2*i+1, 2*i+1) for i in range(self.num_lcl)]#2*i+1是卷积大小，2*i+1是扩张率
        self.mlc = nn.Sequential(*layers)
        
    def forward(self, X):
        
        concat_X = torch.stack([lcl(X) for lcl in self.mlc], dim=-3)
        return torch.mean(concat_X, dim=-3)
    

#%%
class SBAMUnit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.define_model1(in_channels,out_channels)


    def define_model1(self,in_channels,out_channels):
        self.up=nn.Sequential(UpsamplingBilinear2d(scale_factor=2),
                              PointWiseConv(in_channels,out_channels, 
                                            activation='relu'))
        self.conv_blam_x= PointWiseConv(out_channels, out_channels, 
                                        activation='sigmoid')
    
    def forward(self,high_level,low_level):
        up_high=self.up(high_level)
        blam_low_x=self.conv_blam_x(low_level)
        blam_low_y=up_high*blam_low_x
        blam_out=torch.add(low_level,blam_low_y)
        return blam_out
        
# SBAM 块
class SBAMBlock(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self._define_model()
        
    def _define_model(self):
    
        # up6
        self.sbam6=SBAMUnit(192,96)
        self.sbam5=SBAMUnit(96,64)
        self.sbam4=SBAMUnit(64,32)
        self.sbam3=SBAMUnit(32,16)
  
    def forward(self,l6,l5,l4,l3,l2):
        out=self.sbam6(l6,l5)
        out=self.sbam5(out,l4)
        out=self.sbam4(out,l3)
        out=self.sbam3(out,l2)
        return out
        
        
#%%
class mlcem(nn.Module):
    
    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self._define_model()

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) and m.out_channels != 1:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _define_model(self):
        
        self.conv1 = ConvUnit(in_channels=self.in_channels, out_channels=16, kernel_size=3)
        
        # stage1
        self.stage1 = Res_Block(num_res_cell=3, in_channels=16, out_channels=16, kernel_size=3)
        
        # stage2
        self.stage2 = Res_Block(num_res_cell=3, in_channels=16, out_channels=32, kernel_size=3,
                                pooling='max')
        
        # stage3
        self.stage3 = Res_Block(num_res_cell=3, in_channels=32, out_channels=64, kernel_size=3,
                                pooling='max')
        
        # mlc1
        self.mlc1 = MLCBlock(1,16)
        # mlc2
        self.mlc2 = MLCBlock(1,32)
        # mlc3
        self.mlc3 = MLCBlock(1,64)
        # mlc4
        self.mlc4 = MLCBlock(1,96)
        # mlc5
        self.mlc5 = MLCBlock(1,192)
        
        # sbam
        self.sbam = SBAMBlock()
        
        self.conv4 = PointWiseConv(16, 16, activation='relu')#通道之间做了一个全连接
        self.conv5 = PointWiseConv(16, 1, activation='sigmoid')
        
        # 初始化权重
        self.apply(self._init_weights)

        # stage4-5
        self.stage45 = SwinTransformer(patch_size=4, in_chans=64,
                                      embed_dim=96, depths=[6, 4], num_heads=[3, 6])  # 16 16 192
        

    def forward(self, X):
        
        out = self.conv1(X)
        out = self.stage1(out)
        mlc1 = self.mlc1(out)
        out = self.stage2(out)
        mlc2 = self.mlc2(out)
        out = self.stage3(out)
        mlc3 = self.mlc3(out)

        x, stage_x_list = self.stage45(out)
        stage_x_list[0] = F.interpolate(stage_x_list[0], scale_factor=2, mode='bilinear', align_corners=True)
        mlc4 = self.mlc4(stage_x_list[0])
        stage_x_list[1] = F.interpolate(stage_x_list[1], scale_factor=2, mode='bilinear', align_corners=True)
        mlc5 = self.mlc5(stage_x_list[1])
        
        out = self.sbam(mlc5,mlc4,mlc3,mlc2,mlc1)
        out = self.conv4(out)
        out = self.conv5(out)
        return out
    
    



# 加载模型
def get_net(snapshot):
    net = ALCLnetSwin()  # 替换为你的 SwinUNet 模型
    net.load_state_dict(torch.load(snapshot))
    net = torch.nn.DataParallel(net).cuda()
    net.eval()
    return net

# 类激活函数目标
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output * self.mask).sum()

# 主要脚本
if __name__ == '__main__':
    import numpy as np
    parser = argparse.ArgumentParser(description='Grad-CAM Visualization')
    parser.add_argument('--snapshot', required=True, type=str, help='Path to the model snapshot')
    parser.add_argument('--image', required=True, type=str, help='Path to the input image')
    parser.add_argument('--mask', required=True, type=str, help='Path to the mask image')
    parser.add_argument('--output', required=True, type=str, help='Path to save the output image')
    args = parser.parse_args()

    # 读取输入图像和掩码
    img_path = args.image
    mask_path = args.mask
    image = np.array(Image.open(img_path))
    mask = np.array(Image.open(mask_path))
    rgb_img = np.float32(image) / 255
    rgb_mask = np.float32(mask) / 255

    # 预处理图像和掩码
    tensor_img = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_mask = torch.tensor(rgb_mask).unsqueeze(0)
    input_tensor = torch.cat((tensor_img, tensor_mask), dim=0).unsqueeze(0)

    # 加载模型
    model = get_net(args.snapshot)
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    # 模型输出
    model_output = model(input_tensor)
    normalized_masks = torch.nn.functional.softmax(model_output, dim=1).cpu()
    target_class = 'foreground'
    target_category = 1  # 前景类别的索引，假设是1
    target_mask = normalized_masks[0, target_category, :, :].detach().cpu().numpy()
    target_mask_float = np.float32(target_mask)

    # 设置目标层
    target_layers = [model.module.conv5]  # SwinUNet的最后一层

    # 计算Grad-CAM
    targets = [SemanticSegmentationTarget(target_category, target_mask_float)]
    with GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 保存结果
    img = Image.fromarray(cam_image)
    img.save(args.output)
    
    
    


        
        
        
        
        
        
        
        
        
        
        
        
        
