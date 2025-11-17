import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import timm
from torch.autograd import Variable

"""
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
"""
    


class FPN(nn.Module):
    #def __init__(self, block=Bottleneck, num_blocks=[2,4,23,3],n_channels=4,n_classes=1): # ResNet-101
    def __init__(self, n_channels=4, n_classes=1): # efficientnet_b4

        super(FPN, self).__init__()
        #self.in_planes = 64
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, in_chans=n_channels, features_only=True, out_indices=(1, 2, 3, 4)) # new efficientnet_b4
        #self.backbone = timm.create_model('swin_large_patch4_window7_224', pretrained=True, in_chans=n_channels, features_only=True, out_indices=(0, 1, 2, 3), img_size = (1008,1008)) # new swin transformer large

        #self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False) # old
        #self.bn1 = nn.BatchNorm2d(64) # old
        feature_channels = self.backbone.feature_info.channels()

        # we comment the following lines because we do not use ResNet blocks anymore
        """             
        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        """
        # Top layer (old)
        #self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # New
        self.toplayer = nn.Conv2d(feature_channels[3], 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers old 
        """
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        """
        # New
        self.latlayer1 = nn.Conv2d(feature_channels[2], 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(feature_channels[1], 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(feature_channels[0], 256, kernel_size=1, stride=1, padding=0)


        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 8, kernel_size=1, stride=1, padding=0)
        self.output_layer = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        self.gn1 = nn.GroupNorm(128, 128) 
        self.gn2 = nn.GroupNorm(256, 256)

        # self.output = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
    """
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    """
    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up (old)
        """
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        """
        # New for efficientnet_b4
        c2, c3, c4, c5 = self.backbone(x)

        # New for swin transformer large (uncomment if we want use the swin)
        # we neeed to reorder the dimensions from (B, C, H, W) to (B, H, W, C)
        #c2 = c2.permute(0, 3, 1, 2).contiguous()
        #c3 = c3.permute(0, 3, 1, 2).contiguous()
        #c4 = c4.permute(0, 3, 1, 2).contiguous()
        #c5 = c5.permute(0, 3, 1, 2).contiguous()
        


        # print(c1.shape,c2.shape,c3.shape,c4.shape,c5.shape)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # print(p2.shape,p3.shape,p4.shape,p5.shape)
        # Semantic
        _, _, h, w = p2.size()
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w)
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w)
        # 256->128
        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

        # 256->256
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h, w)
        # 256->128
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

        # 256->128
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

        s2 = F.relu(self.gn1(self.semantic_branch(p2)))

        s = self.conv3(s2 + s3 + s4 + s5)

        output = self.output_layer(s)

        _,_,h,w = x.shape
        output = self._upsample(output,h,w)

        return output, s
          
