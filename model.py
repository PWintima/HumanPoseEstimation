import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple, Optional
import math

class BasicBlock(nn.Module):
    """Basic residual block"""
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block"""
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class HighResolutionModule(nn.Module):
    """High-Resolution Module for HRNet"""
    
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        
        self.multi_scale_output = multi_scale_output
        
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)
    
    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            raise ValueError("NUM_BRANCHES should be equal to len(NUM_BLOCKS)")
        if num_branches != len(num_channels):
            raise ValueError("NUM_BRANCHES should be equal to len(NUM_CHANNELS)")
        if num_branches != len(num_inchannels):
            raise ValueError("NUM_BRANCHES should be equal to len(NUM_INCHANNELS)")
    
    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion)
            )
        
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        
        return nn.Sequential(*layers)
    
    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)
    
    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i])
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)
                            ))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        
        return nn.ModuleList(fuse_layers)
    
    def get_num_inchannels(self):
        return self.num_inchannels
    
    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        
        return x_fuse


class HRNet(nn.Module):
    """HRNet (High-Resolution Network) for pose estimation"""
    
    def __init__(self, num_joints=16, width=18):
        super(HRNet, self).__init__()
        
        self.num_joints = num_joints
        self.width = width
        
        # Stage 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 2
        self.layer1 = self._make_layer(Bottleneck, 64, 4)
        
        # Stage 3
        num_channels = [self.width * 2 ** i for i in range(4)]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(HighResolutionModule, 4, 1, num_channels, num_channels)
        
        # Stage 4
        self.transition2 = self._make_transition_layer(num_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(HighResolutionModule, 4, 4, num_channels, num_channels)
        
        # Stage 5
        self.transition3 = self._make_transition_layer(num_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(HighResolutionModule, 4, 3, num_channels, num_channels)
        
        # Final layers
        self.final_layer = nn.Conv2d(pre_stage_channels[0], num_joints, kernel_size=1, stride=1, padding=0)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or 64 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(64, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(64, planes, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(planes * block.expansion, planes))
        
        return nn.Sequential(*layers)
    
    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        
        return nn.ModuleList(transition_layers)
    
    def _make_stage(self, block_class, num_modules, num_branches, num_blocks, num_channels, fuse_method='SUM', multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            
            modules.append(block_class(num_branches, BasicBlock, num_blocks, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        
        return nn.Sequential(*modules), num_inchannels
    
    def forward(self, x):
        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Stage 2
        x = self.layer1(x)
        
        # Stage 3
        x_list = []
        for i in range(4):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        
        # Stage 4
        x_list = []
        for i in range(4):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        
        # Stage 5
        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        
        # Final layer
        y = self.final_layer(y_list[0])
        
        return y


class SimpleBaseline(nn.Module):
    """Simple Baseline model for pose estimation"""
    
    def __init__(self, num_joints=16, backbone='resnet50'):
        super(SimpleBaseline, self).__init__()
        
        self.num_joints = num_joints
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove avgpool and fc
            backbone_dim = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Deconvolution layers
        self.deconv_layers = self._make_deconv_layers(backbone_dim, 256)
        
        # Final prediction layer
        self.final_layer = nn.Conv2d(256, num_joints, kernel_size=1, stride=1, padding=0)
        
    def _make_deconv_layers(self, in_channels, out_channels):
        layers = []
        
        # First deconv layer
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Second deconv layer
        layers.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Third deconv layer
        layers.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply deconvolution layers
        x = self.deconv_layers(features)
        
        # Final prediction
        heatmaps = self.final_layer(x)
        
        return heatmaps


def create_model(model_type='hrnet', num_joints=16, **kwargs):
    """Create pose estimation model"""
    if model_type.lower() == 'hrnet':
        return HRNet(num_joints=num_joints, **kwargs)
    elif model_type.lower() == 'simplebaseline':
        return SimpleBaseline(num_joints=num_joints, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    print("Testing HRNet...")
    hrnet = HRNet(num_joints=16)
    x = torch.randn(1, 3, 256, 256)
    y = hrnet(x)
    print(f"HRNet input shape: {x.shape}, output shape: {y.shape}")
    
    print("\nTesting SimpleBaseline...")
    simple_baseline = SimpleBaseline(num_joints=16)
    y = simple_baseline(x)
    print(f"SimpleBaseline input shape: {x.shape}, output shape: {y.shape}")
    
    # Count parameters
    hrnet_params = sum(p.numel() for p in hrnet.parameters())
    simple_params = sum(p.numel() for p in simple_baseline.parameters())
    
    print(f"\nHRNet parameters: {hrnet_params:,}")
    print(f"SimpleBaseline parameters: {simple_params:,}")

