import os
import numpy
import torch
import torch.nn as nn
import datasets
import torch import optim
import nets
import import utils.transforms


class N2CModel(object):
    def __init__(self, params):
        super(N2CModel, self).__init__()
        self.device = param['device']
        self.model_path = param['model_path']
        self.batch_size = param['batch_size']

        if param['mode'] == 'train':
            self.train_data_transform = transforms.Compose([
                util.transforms.RotateFlip(),
                util.transforms.ToTensor()])

            self.train_dataset = datasets.N2CDataset(dataset_path=param['dataset_path'], mode=param['train'],
                                                        inC=param['inC'], sigma=param['sigma'], transform=self.train_data_transform)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

            self.net = nets.UNet(inC=param['inC'], midC=param['midC']).to(param['device'])

            self.learning_rate = param['lr']
            self.optimizer = optim.Adam(self.net.parameters(), param['lr'])
            self.criterion = nn.MSELoss()

            self.save_train_parameters()

        elif param['mode'] == 'test':
            self.test_data_transform = transforms.Compose([
                util.transforms.ToTensor()])

            self.test_dataset = datasets.N2CDataset(dataset_path=param['dataset_path'], mode=param['test'],
                                                        inC=param['inC'], sigma=param['sigma'], transform=self.test_data_transform)
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

            self.net = nets.UNet(inC=param['inC'], midC=param['midC']).to(param['device'])

            self.criterion = nn.MSELoss()

    def train(self,epochs):
        self.net.train()

        train_loss_list = []
        for e in range(epochs):
            train_loss = 0
            for batch_idx, sample in enumerate(self.train_loader):
                # 入力データ・ラベル
                batch_target = sample['target_img'].to(self.device)
                batch_teach = sample['teach_img'].to(self.device)

                # optimizerの初期化 -> 順伝播 -> Lossの計算 -> 逆伝播 -> パラメータの更新
                self.optimizer.zero_grad()
                batch_output = self.net(batch_target)
                loss = self.criterion(batch_output, batch_teach)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            train_loss_list.append(train_loss / len(self.train_dataset))

        # モデルの保存
        self.save_model()
        print('finished training')
        print('last trainning loss : {}'.format(train_loss_list[-1])

    def test(self):
        # モデルの読み込み
        self.load_model(self.net)
        self.net.eval()
        with torch.no_grad():
            test_loss
            for batch_idx, sample in enumerate(self.test_loader):
                # 入力データ・ラベル
                batch_target = sample['target_img'].to(self.device)
                batch_teach  =sample['teach_img'].to(self.device)

                # 順伝搬
                batch_output = self.net(test_img.to(self.device))
                loss = self.criterion(batch_output, batch_teach)
                test_loss += loss.item()
            test_loss = test_loss / len(self.test_dataset)

        print('finishied test')
        print('test loss : {}'.format(test_loss))

    def save_model(self):
        """Save model to self.model_path"""
        torch.save(self.net.state_dict(), self.model_path)

    def load_model(self, model):
        """Load model from self.model_path"""
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def plot_loss(self):
        return
