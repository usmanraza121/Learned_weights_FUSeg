import torch
import numpy as np
import torchvision
import os
from torchvision import datasets, models, transforms
import PIL
import glob

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
import matplotlib.pyplot as plt

# Normalization should be added when necessary
def data_preprocessing_unet():
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize((256,256)),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ]),
        'valid': transforms.Compose([
            #transforms.Resize(256),
            transforms.Resize((256,256)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            #transforms.Resize(256),
            transforms.Resize((256,256)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }
    return data_transforms


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir=None, image_tform=None, 
                                                mask_tform=None, imgloader=PIL.Image.open, format='.png',
                                                image_type='rgb'):  # Options: 'rgb', 'weights', 'PIL'
        super(ImageLoader, self).__init__()
        self.image_type = image_type
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_filenames=sorted(glob.glob(img_dir + '/**/*' + format, recursive=True))
        # print(self.image_filenames)
        if self.mask_dir is not None:
            self.mask_filenames=sorted(glob.glob(mask_dir + '/**/*' + format, recursive=True))
            # print(self.mask_filenames)
        
        self.image_tform=image_tform
        self.mask_tform=mask_tform
        self.imgloader=imgloader
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, i):
        mask = None
        image = self.imgloader(self.image_filenames[i]).convert('RGB')
        #    ------------------------------------------Grayscale-PIL--------------
        if self.image_type == 'PIL':
            image = image.convert('L')
            # print('____PIL_____')
        
        if self.mask_dir is not None:
            mask = self.imgloader(self.mask_filenames[i]).convert('L')
            # if mask is smoothed, e.g., from JPEG use binarization
            mask = mask.point(lambda p: 255 if p >= 127 else 0)
    
        if self.image_tform:
            image = self.image_tform(image)
        else:
            image = torchvision.transforms.functional.pil_to_tensor(image)
        if self.mask_tform:
            mask = self.mask_tform(mask)
            # if mask is smoothed, e.g., from JPEG use binarization
            binary_mask = torch.where(mask < 0.5, torch.tensor(0.0), torch.tensor(1.0))
            
        else:
            binary_mask = torchvision.transforms.functional.pil_to_tensor(mask)

        # Convert the binary tensor to long
        binary_mask = binary_mask.long()

    #    ------------------------------------------Grayscale-weights----------------------------
        if self.image_type == 'weights':
            weights = torch.tensor([0.2989, 0.5870, 0.1140])  # Grayscale weights
            gray_tensor = torch.tensordot(image, weights, dims=([0], [0]))
            # print('print_gray==',gray_tensor.shape) 
            image = gray_tensor.unsqueeze(0).repeat(3, 1, 1)
            # print('____weights_____')
        
        if self.image_type == 'PIL':
            image = image.repeat(3, 1, 1)  # for PIL_L convert
        # print('----Image_type ', self.image_type, '------')

        return image, binary_mask

if __name__ == "__main__":
  
    img_dir = "data/Foot Ulcer Segmentation Challenge/train/images"
    mask_dir = "data/Foot Ulcer Segmentation Challenge/train/labels"

    image_transforms = Compose([Resize((256, 256)), ToTensor()])
    mask_transforms = Compose([Resize((256, 256)), ToTensor()])
    
    # Instantiate the ImageLoader
    dataset = ImageLoader(
        img_dir=img_dir,
        mask_dir=mask_dir,
        image_tform=image_transforms,
        mask_tform=mask_transforms,
        image_type='weights')  # Options: 'rgb', 'weights', 'PIL')
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for idx, (image, mask) in enumerate(dataloader):
        print(f"Image {idx} shape: {image.shape}")  
        print(f"Mask {idx} shape: {mask.shape}")   

        image_np = image[0].permute(1, 2, 0).numpy()  # Convert to HWC for visualization
        # image_np = image[0].numpy() 
        mask_np = mask[0].squeeze().numpy()          # Convert to HW for visualization
        
        # Display using matplotlib
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image_np)
        ax[0].set_title("Image")
        ax[1].imshow(mask_np, cmap='gray')
        ax[1].set_title("Mask")
        plt.show()
        
        # Break after the first batch for demonstration
        break