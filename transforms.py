from monai.transforms import (
    Compose, RandFlipd, Affined, Rand3DElastic, ToTensord, AddChanneld, DivisiblePadd
)

train_transforms = Compose([
                            AddChanneld(['label']),
                            RandFlipd(['image', 'label'], prob=0.5, spatial_axis=0),
                            RandFlipd(['image', 'label'], prob=0.5, spatial_axis=1),
                            DivisiblePadd(k=8, keys=['image', 'label']),
                            ToTensord(['image', 'label'])
])


val_transforms = Compose([
                            AddChanneld(['label']),
                            DivisiblePadd(k=8, keys=['image', 'label']),
                            ToTensord(['image', 'label'])
])