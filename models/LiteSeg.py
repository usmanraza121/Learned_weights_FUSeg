import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# from .modules import conv1x1, ConvBNAct
# from .backbone import ResNet, Mobilenetv2
class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super(Activation, self).__init__()
        activation_hub = {'relu': nn.ReLU,             'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU,    'prelu': nn.PReLU,
                          'celu': nn.CELU,              'elu': nn.ELU, 
                          'hardswish': nn.Hardswish,    'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU,              'glu': nn.GLU, 
                          'selu': nn.SELU,              'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid,        'softmax': nn.Softmax, 
                          'tanh': nn.Tanh,              'none': nn.Identity,
                        }

        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')

        self.activation = activation_hub[act_type](**kwargs)

    def forward(self, x):
        return self.activation(x)

class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                    bias=False, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation

        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )
class DWConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                    dilation=1, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation

        super(DWConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                        dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )

def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, 
                    padding=0, bias=bias)

class ResNet(nn.Module):
    # Load ResNet pretrained on ImageNet from torchvision, see
    # https://pytorch.org/vision/stable/models/resnet.html
    def __init__(self, resnet_type, pretrained=False):
        super(ResNet, self).__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

        resnet_hub = {'resnet18':resnet18, 'resnet34':resnet34, 'resnet50':resnet50,
                        'resnet101':resnet101, 'resnet152':resnet152}
        if resnet_type not in resnet_hub:
            raise ValueError(f'Unsupported ResNet type: {resnet_type}.\n')

        resnet = resnet_hub[resnet_type](pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)       # 2x down
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 4x down
        x1 = self.layer1(x)
        x2 = self.layer2(x1)      # 8x down
        x3 = self.layer3(x2)      # 16x down
        x4 = self.layer4(x3)      # 32x down

        return x1, x2, x3, x4


class Mobilenetv2(nn.Module):
    def __init__(self, pretrained=False):
        super(Mobilenetv2, self).__init__()
        from torchvision.models import mobilenet_v2

        mobilenet = mobilenet_v2(pretrained=pretrained)

        self.layer1 = mobilenet.features[:4]
        self.layer2 = mobilenet.features[4:7]
        self.layer3 = mobilenet.features[7:14]
        self.layer4 = mobilenet.features[14:18]

    def forward(self, x):
        x1 = self.layer1(x)     # 4x down
        x2 = self.layer2(x1)    # 8x down
        x3 = self.layer3(x2)    # 16x down
        x4 = self.layer4(x3)    # 32x down

        return x1, x2, x3, x4
# _________________________________________________Model________________________________________________________________
class LiteSeg(nn.Module):
    def __init__(self, num_class=2, n_channel=3, backbone_type='mobilenet_v2', act_type='relu', name='LiteSeg'):
        super(LiteSeg, self).__init__()
        self.name = name
        if backbone_type == 'mobilenet_v2':
            self.backbone = Mobilenetv2()
            channels = [320, 32]
        elif 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type)
            channels = [512, 128] if backbone_type in ['resnet18', 'resnet34'] else [2048, 512]
        else:
            raise NotImplementedError()

        self.daspp = DASPPModule(channels[0], 512, act_type)
        self.seg_head = SegHead(512 + channels[1], num_class, act_type)
        self.con2d1 = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        size = x.size()[2:]
        x = self.con2d1(x)
        # print('x_:', x.shape)

        x = x.repeat(1, 3, 1, 1)
        # print('x__r:', x.shape)

        _, x1, _, x = self.backbone(x)
        size1 = x1.size()[2:]

        x = self.daspp(x)
        x = F.interpolate(x, size1, mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)

        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class DASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(DASPPModule, self).__init__()
        hid_channels = in_channels // 5
        last_channels = in_channels - hid_channels * 4
        self.stage1 = ConvBNAct(in_channels, hid_channels, 1, act_type=act_type)
        self.stage2 = ConvBNAct(in_channels, hid_channels, 3, dilation=3, act_type=act_type)
        self.stage3 = ConvBNAct(in_channels, hid_channels, 3, dilation=6, act_type=act_type)
        self.stage4 = ConvBNAct(in_channels, hid_channels, 3, dilation=9, act_type=act_type)
        self.stage5 = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            conv1x1(in_channels, last_channels)
                        )
        self.conv = ConvBNAct(2*in_channels, out_channels, 1, act_type=act_type)

    def forward(self, x):
        size = x.size()[2:]

        x1 = self.stage1(x)
        x2 = self.stage2(x)
        x3 = self.stage3(x)
        x4 = self.stage4(x)
        x5 = self.stage5(x)
        x5 = F.interpolate(x5, size, mode='bilinear', align_corners=True)

        x = self.conv(torch.cat([x, x1, x2, x3, x4, x5], dim=1))
        return x


class SegHead(nn.Sequential):
    def __init__(self, in_channels, num_class, act_type, hid_channels=256):
        super(SegHead, self).__init__(
            ConvBNAct(in_channels, hid_channels, 3, act_type=act_type),
            ConvBNAct(hid_channels, hid_channels//2, 3, act_type=act_type),
            conv1x1(hid_channels//2, num_class)
        )

if __name__ == "__main__":
    model = LiteSeg()

    # print(model)
    input_tensor = torch.randn(1, 3, 256, 256)  # Batch size 1, RGB image 256x256
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # Shape depends on timm architecture
    # s= summary(model, (1, 3, 256, 256))
    # print(s)
    # flops, params = profile(model, (dummy_input, ), verbose=False)
    # # -------------------------------------------------------------------------------#
    # #   flops * 2 because profile does not consider convolution as two operations.
    # # -------------------------------------------------------------------------------#
    # flops         = flops * 2
    # flops, params = clever_format([flops, params], "%.4f")
    # print(f'Total GFLOPs: {flops}')
    # print(f'Total Params: {params}')