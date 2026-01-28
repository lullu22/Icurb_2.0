
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
import timm 


# substitution of BatchNorm2d with GroupNorm
# --- NEW HELPER FUNCTION ---#######################################
def get_norm_layer(num_channels):
    # Use 32 groups or the number of channels, whichever is smaller (InstanceNorm)
    num_groups = min(32, num_channels) 
    return nn.GroupNorm(num_groups, num_channels)
######################################################################


# Refinement head to refine the heads of the FPN
# --- NEW CLASS DEFINITION ---#######################################
class RefinementHead(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super(RefinementHead, self).__init__()
        self.block = nn.Sequential(
            # first step: in_ch -> 64 (128-> 64 in our case)
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            get_norm_layer(64),
            nn.ReLU(inplace=True),
            
            # second step: 64 -> 16
            nn.Conv2d(64, 16, kernel_size=3, padding=1, bias=False),
            get_norm_layer(16),
            nn.ReLU(inplace=True),
            
            # Third step: 16 -> out_ch (1)
            nn.Conv2d(16, out_ch, kernel_size=1)
        )
        
    def forward(self, x):
        return self.block(x)
######################################################################




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = get_norm_layer(planes) ### New

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = get_norm_layer(planes) ### New

        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        #self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.bn3 = get_norm_layer(self.expansion*planes) ### New


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(self.expansion*planes)
                get_norm_layer(self.expansion*planes) ### New
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    


class FPN(nn.Module):
    def __init__(self, backbone_name = 'resnet101', resnet_num_blocks=[2,4,23,3],n_channels=4,n_classes=1,params=[3,1000,1000]):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.params = params
        self.backbone_name = backbone_name

        if backbone_name == 'resnet101': 
            print("using custom Resnet-101 backbone")

            self.in_planes = 64

            self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            #self.bn1 = nn.BatchNorm2d(64)
            self.bn1 = get_norm_layer(64) # New

            # Bottom-up layers
            self.layer1 = self._make_layer(Bottleneck,  64, resnet_num_blocks[0], stride=1)
            self.layer2 = self._make_layer(Bottleneck, 128, resnet_num_blocks[1], stride=2)
            self.layer3 = self._make_layer(Bottleneck, 256, resnet_num_blocks[2], stride=2)
            self.layer4 = self._make_layer(Bottleneck, 512, resnet_num_blocks[3], stride=2)
            
            feature_channels = [256,512,1024,2048]
                                
        elif backbone_name.startswith('efficientnet'):
            print(f"Using {backbone_name} backbone from TIMM")
            
            self.backbone = timm.create_model(backbone_name, pretrained=True, in_chans=n_channels, features_only=True, out_indices=(1, 2, 3, 4))
            feature_channels = self.backbone.feature_info.channels()
            
        else: 
            raise ValueError(f"Backbone '{backbone_name}' not supported. Options: 'resnet101', 'efficientnet_bX'")

        #### COMMON PART: DEFINITION OF FPN LAYERS ####

        # Top layer
        self.toplayer = nn.Conv2d(feature_channels[3], 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(feature_channels[2], 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(feature_channels[1], 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(feature_channels[0], 256, kernel_size=1, stride=1, padding=0)

        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.semantic_branch2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.feature_layer1 = nn.Conv2d(128, 8, kernel_size=1, stride=1, padding=0)

        # Original version (before modifying for init-vertex prediction)
        #self.output_layer1 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)  ## old version input 8 channels, we are no more interested in navigation ,so we increase input channels to 128
        #self.output_layer2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)

        # New version for init-vertex prediction 
        #self.output_layer1 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0) ## new version input 128 channels, same input channel as output_layer2 (boundary prediction)
        #self.output_layer2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)

        # Last version with Refinement Heads
        self.boundary_head = RefinementHead(128, 1)
        self.vertex_head = RefinementHead(128, 1)

        self.gn11 = nn.GroupNorm(128, 128) 

        #self.gn12 = nn.GroupNorm(256, 256)
        self.gn12 = nn.GroupNorm(32, 256) ### new 

        self.gn21 = nn.GroupNorm(128, 128) 

        #self.gn22 = nn.GroupNorm(256, 256)
        self.gn22 = nn.GroupNorm(32, 256) ### new


    def _make_layer(self, block, planes, num_blocks, stride):

        ''' example with stride 2 and num_blocks 3:
            we create a list with only one value equal to [stride] = [2], 
            then if we take the number of blocks minus one we have 2,
            so we create a list of two elements equal to 1: [1,1]
            finally we concatenate the two lists obtaining [2,1,1]
        '''

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

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
        if self.backbone_name == 'resnet101': 
            # Bottom-up
            # if we take x = input image with size 4X1000X1000
            c1 = F.relu(self.bn1(self.conv1(x))) # resolution is equal to 1/2 of input so c1 = 64X500X500
            c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1) # resolution is equal to 1/4 of input so c1 = 64X250X250
            c2 = self.layer1(c1) # resolution is equal to 1/4 of input so c2 = 256X250X250
            c3 = self.layer2(c2) # resolution is equal to 1/8 of input so c3 = 512X125X125
            c4 = self.layer3(c3) # resolution is equal to 1/16 of input so c4 = 1024X63X63
            c5 = self.layer4(c4) # resolution is equal to 1/32 of input so c5 = 2048X32X32
        else: 
            c2, c3, c4, c5 = self.backbone(x) 


        # print(c1.shape,c2.shape,c3.shape,c4.shape,c5.shape)

        # Top-down
        p5 = self.toplayer(c5) # the feature map reduced from 2048 to 256 channels with resolution equal to 32x32
        p4 = self._upsample_add(p5, self.latlayer1(c4)) # the feature map reduced from 1024 to 256 channels with resolution equal to 63x63
        p3 = self._upsample_add(p4, self.latlayer2(c3)) # the feature map reduced from 512 to 256 channels with resolution equal to 125x125
        p2 = self._upsample_add(p3, self.latlayer3(c2)) # the feature map reduced from 256 to 256 channels with resolution equal to 250x250

        # Smooth
        p4 = self.smooth1(p4) # smooth the feature map of p4 because it is noisy after the addition, so feature map remains 256 channels with resolution equal to 63x63
        p3 = self.smooth2(p3) # smooth the feature map of p3 because it is noisy after the addition, so feature map remains 256 channels with resolution equal to 125x125
        p2 = self.smooth3(p2) # smooth the feature map of p2 because it is noisy after the addition, so feature map remains 256 channels with resolution equal to 250x250

        # print(p2.shape,p3.shape,p4.shape,p5.shape)

        # Semantic segmentation branch (mask PREDICTION)
        _, _, h, w = p2.size() # we need the shape of p2 to upsample all the other feature maps to the same size (in this case 250x250)
        
        s5 = self._upsample(F.relu(self.gn12(self.conv2(p5))), h, w) # upsample p5 from 32x32 to 250x250
        s5 = self._upsample(F.relu(self.gn12(self.conv2(s5))), h, w) # upsample s5 from 250x250 to 250x250 (we do it twice to have a better representation of the features after first upsampling)
        s5 = self._upsample(F.relu(self.gn11(self.semantic_branch(s5))), h, w) # finally we reduce the number of channels from 256 to 128 and upsample to 250x250, always with upsampling from 250x250 to 250x250 (no change in size, only for security)

        
        s4 = self._upsample(F.relu(self.gn12(self.conv2(p4))), h, w) # upsample p4 from 63x63 to 250x250
        s4 = self._upsample(F.relu(self.gn11(self.semantic_branch(s4))), h, w) # reduce channels from 256 to 128 and upsample to 250x250, always with upsampling from 250x250 to 250x250 (no change in size, only for security)

        
        s3 = self._upsample(F.relu(self.gn11(self.semantic_branch(p3))), h, w) # upsample p3 from 125x125 to 250x250 and reduce channels from 256 to 128 

        s2 = F.relu(self.gn11(self.semantic_branch(p2))) # p2 is already at 250x250 and we just reduce channels from 256 to 128

        output1_feature = self.feature_layer1(s2 + s3 + s4 + s5) # we sum all the semantic feature maps and reduce channels from 128 to 8
        
        # original version
        #output1 = self._upsample(self.output_layer1(F.relu(output1_feature)), self.params[1], self.params[2]) # final output for mask prediction, from 8 to 1 channel and upsample to original image size (1000x1000)
        
        # new version for mask prediction
        #output1 = self._upsample(self.output_layer1(F.relu(s2+s3+s4+s5)), self.params[1], self.params[2]) # final output for mask prediction, from 128 to 1 channel and upsample to original image size (1000x1000)
        
        # last version with Refinement Head
        output1 = self._upsample(self.boundary_head(s2+s3+s4+s5), self.params[1], self.params[2]) # final output for mask prediction, from 128 to 1 channel and upsample to original image size (1000x1000)

        # Semantic segmentation branch (init-vertex PREDICTION)
        s5 = self._upsample(F.relu(self.gn22(self.conv3(p5))), h, w) # upsample p5 from 32x32 to 250x250
        s5 = self._upsample(F.relu(self.gn22(self.conv3(s5))), h, w) # upsample s5 from 250x250 to 250x250 (we do it twice to have a better representation of the features after first upsampling)
        s5 = self._upsample(F.relu(self.gn21(self.semantic_branch2(s5))), h, w) # finally we reduce the number of channels from 256 to 128 and upsample to 250x250, always with upsampling from 250x250 to 250x250 (no change in size, only for security)

        
        s4 = self._upsample(F.relu(self.gn22(self.conv3(p4))), h, w) # upsample p4 from 63x63 to 250x250
        s4 = self._upsample(F.relu(self.gn21(self.semantic_branch2(s4))), h, w) # reduce channels from 256 to 128 and upsample to 250x250, always with upsampling from 250x250 to 250x250 (no change in size, only for security)

        
        s3 = self._upsample(F.relu(self.gn21(self.semantic_branch2(p3))), h, w) # upsample p3 from 125x125 to 250x250 and reduce channels from 256 to 128

        s2 = F.relu(self.gn21(self.semantic_branch2(p2))) # p2 is already at 250x250 and we just reduce channels from 256 to 128
        #Original version 
        #output2 = self._upsample(self.output_layer2(s2 + s3 + s4 + s5), self.params[1], self.params[2]) # final output for init-vertex prediction, from 128 to 1 channel and upsample to original image size (1000x1000)

        #new version for init-vertex prediction
        #output2 = self._upsample(self.output_layer2(s2 + s3 + s4 + s5), self.params[1], self.params[2]) # final output for init-vertex prediction, from 128 to 1 channel and upsample to original image size (1000x1000)

        #last version with Refinement Head
        output2 = self._upsample(self.vertex_head(s2 + s3 + s4 + s5), self.params[1], self.params[2]) # final output for init-vertex prediction, from 128 to 1 channel and upsample to original image size (1000x1000)

        return output1, output2, output1_feature