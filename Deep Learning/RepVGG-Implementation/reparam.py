import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RepBlock(nn.Module):
    def __init__(self, num_channels):
        super(RepBlock, self).__init__()
        self.fused_conv = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
                                            
    def forward(self, x):
        x = self.relu(self.fused_conv(x))
        return x


def identity3x3(num_channels):
    b = torch.zeros(num_channels, num_channels, 3, 3)
    for i in range(num_channels):
        b[i, i, 1, 1] = 1
    return b


def fuse(conv_weight, conv_bias, bn, num_channels):
    w_conv = conv_weight.clone().view(num_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
    fused_weight = nn.Parameter(torch.mm(w_bn, w_conv).view(num_channels,num_channels,3,3))
    
    bias = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps)) + conv_bias.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))                
    fused_bias = nn.Parameter(bias)

    return fused_weight, fused_bias


def reparameterization(conv3x3, conv1x1, bn3, bn1, bn):
    num_channels = conv1x1.weight.size(0)
    conv1x1to3x3 = F.pad(conv1x1.weight, (1, 1, 1, 1), 'constant', 0)
    
    fuse3_weight, fuse3_bias = fuse(conv3x3.weight, conv3x3.bias, bn3, num_channels)
    fuse1_weight, fuse1_bias = fuse(conv1x1to3x3, conv1x1.bias, bn1, num_channels)
    fuse0_weight, fuse0_bias = fuse(identity3x3(num_channels), torch.zeros(num_channels), bn, num_channels)
    
    reparam_conv = nn.Parameter(fuse3_weight + fuse1_weight + fuse0_weight)
    reparam_bias = nn.Parameter(fuse3_bias + fuse1_bias + fuse0_bias)

    return reparam_conv, reparam_bias


def inference(model, layer_list):
    num_hidden_layers = sum(layer_list)+len(layer_list)
    check_blocks = [0]*layer_list[0] + [1]
    for num_resblocks in layer_list[1:]:
        check_blocks += [0]*num_resblocks + [1]

    for i, j in zip(range(num_hidden_layers), check_blocks):
        if j == 0:
            reparam = RepBlock(model.hidden_layers[i].conv3x3.weight.size(0))
            reparam.fused_conv.weight, reparam.fused_conv.bias = (
                reparameterization(model.hidden_layers[i].conv3x3, model.hidden_layers[i].conv1x1,
                model.hidden_layers[i].bn3, model.hidden_layers[i].bn1, model.hidden_layers[i].bn
                )
            )
            model.hidden_layers[i] = reparam
    return model