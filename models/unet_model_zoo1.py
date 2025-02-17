# based on https://wandb.ai/ishandutta/semantic_segmentation_unet/reports/Semantic-Segmentation-with-UNets-in-PyTorch--VmlldzoyMzA3MTk1
import torch
import torch.nn as nn
from torchvision.models import vgg16_bn


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

def encoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# def encoder_block(input, num_filters):
#     conv = conv2D_block(input, num_filters)
#     pool = MaxPooling2D((2, 2))(conv)
#     return conv, pool


class UNet_VGG(nn.Module):
    def __init__(self, pretrained=True, out_channels=12, learn_gray=False):
        super().__init__()
        
        self.learn_gray = learn_gray
        if self.learn_gray:
            self.name = 'UNet_VGG_learn_gray'
        else:
            self.name = 'UNet_VGG'

        self.con2d_in = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        
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

        if self.learn_gray:
            x = self.con2d_in(x)  
            x = x.repeat(1, 3, 1, 1)
        
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


class UNet(nn.Module):
    def __init__(self, pretrained=True, out_channels=12, learn_gray=False):
        super().__init__()
        self.learn_gray = learn_gray
        if self.learn_gray:
            self.name = 'UNet_learn_gray'
        else:
            self.name = 'UNet'

        base = 16

        self.con2d_in = nn.Conv2d(3, 1, kernel_size=1, bias=False)

        self.encoder_block00 = conv(3, 1*base)
        self.maxpool00 = nn.MaxPool2d(2)
        
        self.encoder_block01 = conv(1*base, 2*base)
        self.maxpool01 = nn.MaxPool2d(2)
        
        self.encoder_block02 = conv(2*base,4*base)
        self.maxpool02 = nn.MaxPool2d(2)
        
        self.encoder_block03 = conv(4*base, 8*base)
        self.maxpool03 = nn.MaxPool2d(2)
        
        self.encoder_block04 = conv(8*base, 16*base)
        self.maxpool04 = nn.MaxPool2d(2)

        self.encoder_block05 = conv(16*base, 32*base)
        

        self.up_conv6 = up_conv(32*base, 16*base)
        self.conv6 = conv(16*base + 16*base, 16*base)
        self.up_conv7 = up_conv(16*base, 8*base)
        self.conv7 = conv(8*base + 8*base, 8*base)
        self.up_conv8 = up_conv(8*base, 4*base)
        self.conv8 = conv(4*base + 4*base, 4*base)
        self.up_conv9 = up_conv(4*base, 2*base)
        self.conv9 = conv(2*base + 2*base, 2*base)
        self.up_conv10 = up_conv(2*base, 1*base)
        self.conv10 = conv(1*base + 1*base, 1*base)
        self.conv11 = nn.Conv2d(1*base, out_channels, kernel_size=1)
        
    def forward(self, x):
      
        if self.learn_gray:
            x = self.con2d_in(x)  
            x = x.repeat(1, 3, 1, 1)
        
        encoder_block00 = self.encoder_block00(x)
        maxpool00 = self.maxpool00(encoder_block00)
        
        encoder_block01 = self.encoder_block01(maxpool00)
        maxpool01 = self.maxpool01(encoder_block01)

        encoder_block02 = self.encoder_block02(maxpool01)
        maxpool02 = self.maxpool02(encoder_block02)

        encoder_block03 = self.encoder_block03(maxpool02)
        maxpool03 = self.maxpool03(encoder_block03)

        encoder_block04 = self.encoder_block04(maxpool03)
        maxpool04 = self.maxpool04(encoder_block04)
        
        x = self.encoder_block05(maxpool04)
       
        x = self.up_conv6(x)
        x = torch.cat([x, encoder_block04], dim=1)
        x = self.conv6(x)


        x = self.up_conv7(x)
        x = torch.cat([x, encoder_block03], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, encoder_block02], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, encoder_block01], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, encoder_block00], dim=1)
        x = self.conv10(x)

        x = self.conv11(x)

        return x


