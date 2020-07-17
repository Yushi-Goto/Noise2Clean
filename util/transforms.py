import torch
import numpy as np
import cv2


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        return {img: to_tensor(sample[img]) for img in sample}

def to_tensor(img):
    img = img.transpose((2, 0, 1))
    return torch.from_numpy(img).float()

class RotateFlip(object):
    def __call__(self, sample):
        p = torch.randint(0,8,(1,))
        return {img: func_rf(sample[img], p) for img in sample}

def func_rf(img, p):
    if p==0:
        img=img
    elif p==1:
        img=np.flip(img, axis=1)
    elif p==2:
        img=np.rot90(img, k=1, axes=(0,1))
    elif p==3:
        img=np.flip(img, axis=1)
        img=np.rot90(img, k=1, axes=(0,1))
    elif p==4:
        img=np.rot90(img, k=2, axes=(0,1))
    elif p==5:
        img=np.flip(img, axis=1)
        img=np.rot90(img, k=2, axes=(0,1))
    elif p==6:
        img=np.rot90(img, k=3, axes=(0,1))
    elif p==7:
        img=np.flip(img, axis=1)
        img=np.rot90(img, k=3, axes=(0,1))

    img=img.copy()

    return img
