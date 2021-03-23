import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader, random_split

from my_torch_project.datamodel import CustomDataset # local
from my_torch_project.trainer import trainer1
from my_torch_project.model import getModule


def get_args():
    parser = argparse.ArgumentParser(description='Train my_torch_project on images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    #parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
    #                    help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = getModule()

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    # ** define data **
    dataset = CustomDataset(
        "/home/sampsa/cnn/kaggle/data/my_torch_project/train_images",
        "/home/sampsa/cnn/kaggle/data/my_torch_project/train.csv",
        n_classes = 5,
        # n_max=500 # debugging
    )
    val_percent = 0.05
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    
    # do augmenting for certain classes..? say, with: 
    # https://stackoverflow.com/questions/51677788/data-augmentation-in-pytorch
    # then create the final augmented dataset with:
    # torch.utils.data.ConcatDataset([transformed_dataset,original])

    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train, 
        batch_size=args.batchsize, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True)
    
    val_loader = DataLoader(val, 
        batch_size=args.batchsize, 
        shuffle=False,
        num_workers=8, 
        pin_memory=True, 
        drop_last=True)

    trainer1(
        net=net,
        device=device,
        epochs=args.epochs,
        batch_size=args.batchsize,
        lr=args.lr,
        save_cp=True,
        cp_dir="checkpoints/",
        train_loader = train_loader,
        val_loader = val_loader
        )
