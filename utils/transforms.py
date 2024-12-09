# diagram_detector/utils/transforms.py

import torchvision.transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # Add data augmentation here if needed
        pass
    return T.Compose(transforms)
