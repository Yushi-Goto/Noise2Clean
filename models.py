import os
import numpy
from matplotlib import pylab as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch import optim
import cv2
import datasets
import nets
import util.transforms


class N2CModel(object):
    def __init__(self, param):
        super(N2CModel, self).__init__()
        self.device = param['device']
        self.model_path = param['model_path']

        if param['mode'] == 'train':
            self.batch_size = param['batch_size']

            if param['lpfir'] or param['no_rf']:
                self.train_data_transform = transforms.Compose([
                    util.transforms.Resize(param['img_size'], param['inC']),
                    util.transforms.AddNoise(param['sigma']),
                    util.transforms.ToTensor()])
            else:
                self.train_data_transform = transforms.Compose([
                    util.transforms.Resize(param['img_size'], param['inC']),
                    util.transforms.AddNoise(param['sigma']),
                    util.transforms.RotateFlip(),
                    util.transforms.ToTensor()])

            self.train_dataset = datasets.N2CDataset(dataset_path=param['dataset_path'], mode=param['mode'], transform=self.train_data_transform)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

            self.net = nets.UNet(inC=param['inC'], midC=param['midC']).to(param['device'])

            self.learning_rate = param['lr']
            self.optimizer = optim.Adam(self.net.parameters(), param['lr'])
            self.criterion = nn.MSELoss()

        elif param['mode'] == 'test':
            self.batch_size = param['batch_size']

            self.test_data_transform = transforms.Compose([
                util.transforms.Resize(param['img_size'], param['inC']),
                util.transforms.AddNoise(param['sigma']),
                util.transforms.ToTensor()])

            self.test_dataset = datasets.N2CDataset(dataset_path=param['dataset_path'], mode=param['mode'], transform=self.test_data_transform)
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

            self.net = nets.UNet(inC=param['inC'], midC=param['midC']).to(param['device'])

            self.criterion = nn.MSELoss()

        elif param['mode'] == 'run':
            self.run_data_transform = transforms.Compose([
                util.transforms.Resize(None, param['inC']),
                util.transforms.AddNoise(param['sigma']),
                util.transforms.ToTensor()])

            self.run_dataset = datasets.N2CDataset(dataset_path=param['dataset_path'], mode=param['mode'],
                                                    transform=self.run_data_transform, data_path=param['data_path'])
            self.run_loader = torch.utils.data.DataLoader(self.run_dataset, batch_size=len(self.run_dataset))

            self.net = nets.UNet(inC=param['inC'], midC=param['midC']).to(param['device'])

    def train(self, epochs, lpfir):
        self.net.train()

        train_loss_list = []
        for e in range(epochs):
            train_loss = 0
            for batch_idx, sample in enumerate(self.train_loader):
                batch_target = sample['target_img'].to(self.device)
                batch_teach = sample['teach_img'].to(self.device)

                self.optimizer.zero_grad()
                if (e == 0) and (batch_idx == 0):
                    if lpfir:
                        self.select_param_conv()
                        self.over_write_forward()
                batch_output = self.net(batch_target)
                loss = self.criterion(batch_output, batch_teach)
                train_loss += loss.item()
                loss.backward()
                if lpfir:
                    self.over_write_backward()
                self.optimizer.step()
            train_loss_list.append(train_loss / len(self.train_dataset))

        self.save_model()
        self.plot_loss(epochs, train_loss_list)
        print('finished training')
        print('last trainning loss : {}'.format(train_loss_list[-1]))

    def test(self):
        self.load_model()
        self.net.eval()
        with torch.no_grad():
            test_loss = 0
            for batch_idx, sample in enumerate(self.test_loader):
                batch_target = sample['target_img'].to(self.device)
                batch_teach  =sample['teach_img'].to(self.device)

                batch_output = self.net(batch_target)
                loss = self.criterion(batch_output, batch_teach)
                test_loss += loss.item()
            test_loss = test_loss / len(self.test_dataset)

        print('finishied test')
        print('test loss : {}'.format(test_loss))

    def run(self):
        self.load_model()
        self.net.eval()
        with torch.no_grad():
            sample = next(self.run_loader.__iter__())
            target = sample['target_img'].to(self.device)
            teach = sample['teach_img'].to(self.device)

            output = self.net(target)

        self.save_img(target, teach, output)
        print('finishied running')

    def save_model(self):
        """Save model to self.model_path"""
        torch.save(self.net.state_dict(), self.model_path)

    def load_model(self):
        """Load model from self.model_path"""
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def plot_loss(self, epochs, loss_list):
        plt.figure()
        plt.plot(range(epochs), loss_list, color='blue', linestyle='-', label='train_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Train Loss')
        plt.savefig('TrainLossFig.png')

    def save_img(self, target, teach, output):
        target_img = numpy.squeeze(target.to(torch.device('cpu')).numpy())
        teach_img = numpy.squeeze(teach.to(torch.device('cpu')).numpy())
        output_img = numpy.squeeze(output.to(torch.device('cpu')).numpy())

        cv2.imwrite('target_img.jpg', target_img * 255)
        cv2.imwrite('teach_img.jpg', teach_img * 255)
        cv2.imwrite('output_img.jpg', output_img * 255)

    def over_write_forward(self):
        param_dic = self.net.state_dict()
        for name in self.param_conv:
            shape = param_dic[name].shape

            if not ((shape[2] == 1) or (shape[3] == 1)):
                param_dic[name][:, :, (shape[2]+1)//2:shape[2], :] = param_dic[name][:, :, torch.arange((shape[2]+1)//2-1-1, 0-1, -1), :]
                param_dic[name][:, :, :, (shape[3]+1)//2:shape[3]] = param_dic[name][:, :, :, torch.arange((shape[3]+1)//2-1-1, 0-1, -1)]

        param_dic.update(param_dic)
        self.net.load_state_dict(param_dic)

    def over_write_backward(self):
        for n, p in self.net.named_parameters():
            if n in self.param_conv:
                shape = p.shape
                radius = shape[2] // 2

                for y in range(radius):
                    for x in range(radius):
                        grad0 = p.grad[:, :, 0+y, 0+x]
                        grad1 = p.grad[:, :, 0+y, -1-x]
                        grad2 = p.grad[:, :, -1-y, 0+x]
                        grad3 = p.grad[:, :, -1-y, -1-x]
                        sum_grad = (grad0 + grad1 + grad2 + grad3) / 4
                        p.grad[:, :, 0+y, 0+x] = sum_grad
                        p.grad[:, :, 0+y, -1-x] = sum_grad
                        p.grad[:, :, -1-y, 0+x] = sum_grad
                        p.grad[:, :, -1-y, -1-x] = sum_grad

                for y in range(radius):
                    grad0 = p.grad[:, :, 0+y, radius]
                    grad1 = p.grad[:, :, -1-y, radius]
                    sum_grad = (grad0 + grad1) / 2
                    p.grad[:, :, 0+y, radius] = sum_grad
                    p.grad[:, :, -1-y, radius] = sum_grad

                for x in range(radius):
                    grad0 = p.grad[:, :, radius, 0+x]
                    grad1 = p.grad[:, :, radius, -1-x]
                    sum_grad = (grad0 + grad1) / 2
                    p.grad[:, :, radius, 0+x] = sum_grad
                    p.grad[:, :, radius, -1-x] = sum_grad

    def select_param_conv(self):
        param_dic = self.net.state_dict()
        self.param_conv = []
        for name in param_dic.keys():
            if not (('bias' in name) or ('bn' in name) or ('deconv' in name)):
                self.param_conv.append(name)
