import torch
import numpy as np
import torchvision
import os
from torchvision import datasets, models, transforms
import PIL
import glob

def data_preprocessing_engine():
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize((224,224)),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            #transforms.Resize(256),
            transforms.Resize((224,224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            #transforms.Resize(256),
            transforms.Resize((224,224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


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

def get_train_valid_loader(data_path, data_transforms, batch_size=16, num_workers=4):

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x),
                                              data_transforms[x])
                      for x in ['train', 'valid']}
    dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers)
                  for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    return dataloader, dataset_sizes, class_names


def get_test_loader(data_path, data_transforms, batch_size=16, num_workers=4):
    test_dataset = {x: datasets.ImageFolder(os.path.join(data_path, x),
                                          data_transforms[x])
                  for x in ['test']}
    testloader = {x: torch.utils.data.DataLoader(test_dataset[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=4)
                  for x in ['test']}
    test_dataset_sizes = {x: len(testloader[x]) for x in ['test']}
    class_names = test_dataset['test'].classes
    
    return testloader, test_dataset_sizes, class_names


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir=None, image_tform=None, mask_tform=None, imgloader=PIL.Image.open):
        super(ImageLoader, self).__init__()
    
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_filenames=sorted(glob.glob(img_dir+'/**/*.png', recursive=True))
        # print(self.image_filenames)
        if self.mask_dir is not None:
            self.mask_filenames=sorted(glob.glob(mask_dir+'/**/*.png', recursive=True))
            # print(self.mask_filenames)
        
        self.image_tform=image_tform
        self.mask_tform=mask_tform
        self.imgloader=imgloader
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, i):
        mask = None
        image = self.imgloader(self.image_filenames[i]).convert('RGB')  # io.imread(self.filenames[i])
        
        if self.mask_dir is not None:
            mask = self.imgloader(self.mask_filenames[i]).convert('L')  # io.imread(self.filenames[i])
            mask = mask.point(lambda p: 255 if p >= 127 else 0)
            
    
        if self.image_tform:
            image = self.image_tform(image)
        else:
            image = torchvision.transforms.functional.pil_to_tensor(image)
        if self.mask_tform:
            mask = self.mask_tform(mask)
            binary_mask = torch.where(mask < 0.5, torch.tensor(0.0), torch.tensor(1.0))
            
        else:
            binary_mask = torchvision.transforms.functional.pil_to_tensor(mask)
        

        # Convert the binary tensor to long
        binary_mask = binary_mask.long()

        return image, binary_mask