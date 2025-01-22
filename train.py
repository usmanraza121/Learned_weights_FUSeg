import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary

import matplotlib.pyplot as plt
import time
import os
import PIL
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True

from loaders.data_loader import *
from trainers.trainer import *
# from trainers.trainer import train_model

def UnetLoss(preds, targets):
    ce_loss = loss(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    return ce_loss, acc

batch_size = 16
num_workers = 1
img_dir = "data/Foot Ulcer Segmentation Challenge/train/images"
mask_dir = "data/Foot Ulcer Segmentation Challenge/train/labels"

img_dir_val = "data/Foot Ulcer Segmentation Challenge/validation/images"
mask_dir_val = "data/Foot Ulcer Segmentation Challenge/validation/labels"

data_transforms = data_preprocessing_unet()
# ----------------------------------------select image type for input------------------
image_loader_train = ImageLoader(img_dir, mask_dir=mask_dir,
                                    image_tform=data_transforms['train'], 
                                    mask_tform=data_transforms['train'], 
                                    imgloader=PIL.Image.open,
                                    image_type='PIL')  # Options: 'rgb', 'weights', 'PIL'
                                    
dataset_train_size = len(image_loader_train)
print('----Image_type ', image_loader_train.image_type, '------')
dataloader_train = torch.utils.data.DataLoader(image_loader_train,
                                                batch_size=batch_size, 
                                                shuffle=False, 
                                                drop_last=True, 
                                                num_workers=num_workers
                                                )


image_loader_valid = ImageLoader(img_dir_val, mask_dir=mask_dir_val,
                                    image_tform=data_transforms['valid'], 
                                    mask_tform=data_transforms['valid'], 
                                    imgloader=PIL.Image.open)
dataset_valid_size = len(image_loader_valid)
dataloader_valid = torch.utils.data.DataLoader(image_loader_valid, 
                                                batch_size=batch_size, 
                                                shuffle=False, 
                                                drop_last=True, 
                                                num_workers=num_workers
                                                )

dataloaders = {'train': dataloader_train, 'valid': dataloader_valid}
dataset_sizes = {'train': dataset_train_size, 'valid': dataset_valid_size}

from models.tiny_unet import *
from models.unet_model_zoo import UNet, DeepLabV3, SSModelZoo
from models.LiteSeg import LiteSeg
# ------------------------model--------------------------
# model = TinyUNet(in_channels=3, num_classes=2, name='tinyunet')
# model = UNet(out_channels=2 ,name='UNet')
model = LiteSeg(backbone_type='mobilenet_v2', name='LiteSeg')  # backbone_type='mobilenet_v2' or 'resnet' 
# model = DeepLabV3(num_classes=2, name='DeepLabV3')
# model = SSModelZoo(model_name='resnet50d', pretrained=False) 


model.name = f'{model.name}_{image_loader_train.image_type}'
# ----------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device is: {device}")

loss = nn.CrossEntropyLoss()
criterion = UnetLoss

# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

ckpt_folder = "./ckpt"

# model_res = train_model(model=model, 
#                         criterion=criterion, 
#                         optimizer=optimizer, 
#                         scheduler=exp_lr_scheduler, 
#                         device=device, 
#                         dataloaders=dataloaders, 
#                         dataset_sizes=dataset_sizes, 
#                         ckpt_folder=ckpt_folder, 
#                         num_epochs=100)

model_res = train_model(model=model, 
                        criterion=criterion, 
                        optimizer=optimizer, 
                        scheduler=exp_lr_scheduler, 
                        device=device, 
                        dataloaders=dataloaders, 
                        dataset_sizes=dataset_sizes, 
                        ckpt_folder=ckpt_folder, 
                        num_epochs=40)