import torch
import torchvision
import nibabel as nib
import glob
import os
import numpy as np


class BraTSDataSet(torch.utils.data.Dataset):
  def __init__(self, img_root, label_root, transform=None):
    self.img_root = img_root
    self.label_root = label_root
    self.transform = transform
    self.img_list = glob.glob(os.path.join(img_root, '*.nii.gz'))
    self.label_list = glob.glob(os.path.join(label_root, '*.nii.gz'))
    assert len(self.img_list)==len(self.label_list), "Some Data Samples are missing!"

  def __len__(self):
    return len(self.img_list)

  def __getitem__(self, idx):
    image = self.img_list[idx]
    label = self.label_list[idx]
    image = nib.load(image).get_fdata().astype(np.float32)
    label = nib.load(label).get_fdata()
    image = np.transpose(image)
    label = np.transpose(label)
    item_dict= {'image': image, 'label': label}
    if self.transform:
      item_dict = self.transform(item_dict)
    else:
      image = torchvision.transforms.ToTensor()(image)
      label = torchvision.transforms.ToTensor()(label)
      item_dict['image'] = image
      item_dict['label'] = label
    return item_dict['image'], item_dict['label']