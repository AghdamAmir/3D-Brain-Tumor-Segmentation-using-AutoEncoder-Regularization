from monai.transforms import (
    Compose, RandFlipd, Affined, Rand3DElastic, ToTensord, AddChanneld, DivisiblePadd, Resized
)
from config import input_shape

train_transforms = Compose([
                            AddChanneld(['label']),
                            # RandFlipd(['image', 'label'], prob=0.5, spatial_axis=0),
                            # RandFlipd(['image', 'label'], prob=0.5, spatial_axis=1),
                            Resized(['image', 'label'], spatial_size=input_shape[-3:]),
                            ToTensord(['image', 'label'])
])


val_transforms = Compose([
                            AddChanneld(['label']),
                            Resized(['image', 'label'], spatial_size=input_shape[-3:]),
                            ToTensord(['image', 'label'])
])