# based on https://wandb.ai/ishandutta/semantic_segmentation_unet/reports/Semantic-Segmentation-with-UNets-in-PyTorch--VmlldzoyMzA3MTk1
import torch
import torch.nn as nn
from torchvision.models import vgg16_bn
import torchvision.models as models
import timm 

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    # def __init__(self, in_channels=3, num_classes=2, name='tinyunet'):
    def __init__(self, pretrained=True, out_channels=2, name='UNet'):
        super(UNet, self).__init__()
        self.name = name
        # self.name = 'UNet'


        self.encoder = vgg16_bn(pretrained=pretrained).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])


        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = conv(512, 1024)


        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = conv(64 + 128, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        # print('shape==', x.shape)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)


        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)


        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)


        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)


        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)


        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)


        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)


        x = self.conv11(x)


        return x

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=2, name='DeepLabV3'):
     
        super(DeepLabV3, self).__init__()
        self.name = name
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
   
        return self.model(x)['out']  
# ----------------------Model Zoo--------------------------

class SSModelZoo(nn.Module):
    def __init__(self, model_name='deeplabv3_resnet101', num_classes=2, pretrained=True):
        
        super(SSModelZoo, self).__init__()
        self.name = model_name
        self.model_name = model_name.lower()
        self.num_classes = num_classes

        # Supported PyTorch models
        if self.model_name == 'deeplabv3_resnet101':
            self.model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
        elif self.model_name == 'deeplabv3_resnet50':
            self.model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        elif self.model_name == 'fcn_resnet101':
            self.model = models.segmentation.fcn_resnet101(pretrained=pretrained)
        elif self.model_name == 'fcn_resnet50':
            self.model = models.segmentation.fcn_resnet50(pretrained=pretrained)
        elif self.model_name == 'lraspp_mobilenet_v3_large':
            self.model = models.segmentation.lraspp_mobilenet_v3_large(pretrained=pretrained)
        elif self.model_name == 'deeplabv3_mobilenet_v3_large':
            self.model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=pretrained)

        else:
            raise ValueError(f"Unsupported model: {model_name}. Supported models include: "
                             f"['deeplabv3_resnet101', 'deeplabv3_resnet50', 'fcn_resnet101', "
                             f"'fcn_resnet50', 'lraspp_mobilenet_v3_large', 'deeplabv3_mobilenet_v3_large', ")

        # Modify classifier to match the number of classes if applicable
        if hasattr(self.model, 'classifier'):  # For torchvision models with classifier
            in_channels = self.model.classifier[4].in_channels
            self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        if hasattr(self.model, 'aux_classifier'):  
            in_channels_aux = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=1)


    def forward(self, x):
     
        if hasattr(self.model, 'forward_features'):  # For timm models
            x = self.model.forward_features(x)
            return x
        else:  # For torchvision models
            return self.model(x)['out']

# =====================================


if __name__ == "__main__":
    # model = DeepLabV3Plus(num_classes=2)  
    model = SSModelZoo(model_name='resnet50d', num_classes=2, pretrained=False)
    print(model)
    input_tensor = torch.randn(1, 3, 256, 256)  
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  