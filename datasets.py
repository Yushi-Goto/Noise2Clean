import os
import numpy
import torch
import cv2
import shutil


class N2CDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, mode, inC, sigma, transform=None):
        self.mode = mode
        if inC == 1:
            self.grayscale = True
        self.sigma = sigma
        self.transform = transform

        if os.path.exists(dataset_path + '/train') and os.path.exists(dataset_path + '/test'):
            #Delete hidden files(.DS_Store) and Get file list
            files = os.listdir(dataset_path)
            self.files = []
            for f in files:
                if f.startswith('.'):
                    os.remove(f)
                else:
                    self.files.append(f)

            self.train_num = 900
            self.test_num = int(len(self.files) - 900)

            os.mkdir(dataset_path + '/train')
            os.mkdir(dataset_path + '/test')

            for i, f in enumerate(self.files):
                if i < self.train_num:
                    new_path = shutil.move(dataset_path + '/' + f, dataset_path + '/train/' + str(i+1) + '.jpg')
                else:
                    new_path = shutil.move(dataset_path + '/' + f, dataset_path + '/test/' + str(i+1) + '.jpg')

        if self.mode == 'train':
            self.dataset_path = dataset_path + '/train/'

            self.data = []
            for f in os.listdir(self.dataset_path):
                if f.startswith('.'):
                    os.remove(f)
                else:
                    self.data.append(f)

        else:
            self.dataset_path = dataset_path + '/test/'

            self.data = []
            for f in os.listdir(self.dataset_path):
                if f.startswith('.'):
                    os.remove(f)
                else:
                    self.data.append(f)

    def __len__(self):
        if self.mode == 'train':
            return self.train_num
        else:
            return self.test_num

    def __getitem__(self, idx):
        if self.grayscale:
            teach_img = cv2.imread(self.dataset_path + self.data[idx], cv2.IMREAD_GRAYSCALE)/255
            teach_img = numpy.reshape(teach_img,(teach_img.shape[0], teach_img.shape[1], 1))
            target_img = teach_img + (self.sigma/255) * numpy.random.randn(*teach_img.shape)
        else:
            teach_img = cv2.imread(self.dataset_path + self.data[idx])/255
            target_img = teach_img + (self.sigma/255) * numpy.random.randn(*teach_img.shape)

        sample = {'teach_img':teach_img, 'target_img':target_img}

        if self.transform:
            sample = self.transform(sample)

        return sample
