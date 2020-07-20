import os
import numpy
import torch
import cv2
import shutil
import random


class N2CDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, mode, transform, data_path=None):
        self.mode = mode
        self.transform = transform

        if not (os.path.exists(dataset_path + '/train') and os.path.exists(dataset_path + '/test')):
            #Delete hidden files(.DS_Store) and Get file list
            files = os.listdir(dataset_path)
            random.shuffle(files)
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

        elif self.mode == 'test':
            self.dataset_path = dataset_path + '/test/'

            self.data = []
            for f in os.listdir(self.dataset_path):
                if f.startswith('.'):
                    os.remove(f)
                else:
                    self.data.append(f)

        else:
            self.data_path = data_path

    def __len__(self):
        if self.mode == 'train':
            return len(os.listdir(self.dataset_path))
        elif self.mode == 'test':
            return len(os.listdir(self.dataset_path))
        else:
            return 1

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'test':
            self.data_path = self.dataset_path + self.data[idx]

        teach_img = cv2.imread(self.data_path, cv2.IMREAD_GRAYSCALE) / 255

        if self.transform:
            sample = self.transform(teach_img)

        return sample
