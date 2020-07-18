import os
import torch
import argparse
import models


def parse_args():
    """Command-line argument parser"""
    parser = argparse.ArgumentParser(description='Fashion Recommendation System')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], help='select mode (train or test)')
    parser.add_argument('--dataset-path', default='./dataset', help='dataset root path (default: ./dataset)')
    parser.add_argument('--data-path', help='data path (mode : run) (default: ./dataset)')
    parser.add_argument('--inC', type=int, default=1, help='Number of input channels after reshape (default: 1)')
    parser.add_argument('--midC', type=int, default=96, help='output channel of hidden conv layer (default: 96)')
    parser.add_argument('--sigma', type=float, default=0.1, help='variance of noise (default: 0.1)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--model-path', default='./FRModel.pht', help='save model path (default: ./FRModel.pht)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--fir', action='store_true', help='FIR or NORMAL mode')

    return parser.parse_args()


def main():
    params = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    if args.mode == 'train':
        param = {'device':device, 'mode':args.mode, 'model_path':args.model_path, 'dataset_path':args.dataset_path,
            'inC':args.inC, 'midC':args.midC, 'sigma':args.sigma, 'batch_size':args.batch_size, 'lr':args.learning_rate}
        model = models.N2CModel(param)
        model.train(args.epochs, args.fir)

    elif args.mode == 'test':
        param = {'device':device, 'mode':args.mode, 'model_path':args.model_path, 'dataset_path':args.dataset_path,
            'inC':args.inC, 'midC':args.midC, 'sigma':args.sigma, 'batch_size':args.batch_size}
        model = models.N2VModel(param)
        model.test()

    elif args.mode == 'run':
        param = {'device':device, 'mode':args.mode, 'model_path':args.model_path, 'dataset_path':args.dataset_path,
            'inC':args.inC, 'midC':args.midC, 'sigma':args.sigma, 'data_path':args.data_path}
        model = models.N2CModel(param)
        model.run()

if __name__ == '__main__':
    main()
