import torch
import numpy
import cv2

class Resize(object):
    """Resize a face image to minimum size"""
    def __init__(self, output_size, inC):
        self.output_size = output_size
        if inC == 1:
            self.grayscale = True

    def __call__(self, img):
        if not (self.output_size == None):
            img = cv2.resize(img, (self.output_size, self.output_size))
        if self.grayscale:
            img = numpy.reshape(img,(img.shape[0], img.shape[1], 1))
        return img

class AddNoise(object):
    """Add noise to teach img for cleate target img"""
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, teach_img):
        target_img = teach_img + (self.sigma / 255) * numpy.random.randn(*teach_img.shape)
        return {'teach_img':teach_img, 'target_img':target_img}

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
        img=numpy.flip(img, axis=1)
    elif p==2:
        img=numpy.rot90(img, k=1, axes=(0,1))
    elif p==3:
        img=numpy.flip(img, axis=1)
        img=numpy.rot90(img, k=1, axes=(0,1))
    elif p==4:
        img=numpy.rot90(img, k=2, axes=(0,1))
    elif p==5:
        img=numpy.flip(img, axis=1)
        img=numpy.rot90(img, k=2, axes=(0,1))
    elif p==6:
        img=numpy.rot90(img, k=3, axes=(0,1))
    elif p==7:
        img=numpy.flip(img, axis=1)
        img=numpy.rot90(img, k=3, axes=(0,1))

    img=img.copy()

    return img
