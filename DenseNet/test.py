# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 10:15:01 2021

@author: Dongil Ryu
"""
import argparse

import torch
from torchvision import datasets
from torchvision.transforms import transforms

from model import DenseNet

MEAN_LIST, STD_LIST = [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
IMG_SIZE = 32

def parse():
    parser = argparse.ArgumentParser(description='DenseNet Training')
    parser.add_argument('--data_path', '-dp', type=str, default='./data')
    parser.add_argument('--load_path', '-lp', type=str, default='')
    parser.add_argument('--L', '-l', type=int, default=40)
    parser.add_argument('--k', '-k', type=int, default=12)
    parser.add_argument('--is_BC', '-bc', type=int, default=0)
    parser.add_argument('--is_aug', '-aug', type=int, default=0)

    args = parser.parse_args()
    return vars(args)

def test(data_path='./data', load_path='', L=40, k=12, is_BC=0, is_aug=0):
    is_BC, is_aug = bool(is_BC), bool(is_aug)
    theta = 0.5 if is_BC else 1
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DenseNet(L=L, k=k, theta=theta, is_BC=is_BC, is_aug=is_aug, num_classes=100).to(device)

    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict)

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_LIST, std=STD_LIST)
    ])
    
    valid_dataset = datasets.CIFAR100(root=data_path, train=False, transform=transform_valid, download=True)    
    loss_function = torch.nn.CrossEntropyLoss()    
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    val_loss, correct, total = validate(loss_function, valid_loader, model, device)
    
    print ("Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
        val_loss / total, correct, total, 100 * correct / total))

def validate(loss_function, valid_loader, model, device):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            val_loss += loss_function(output, labels).item() * labels.size(0)
            
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            
            total += labels.size(0)
            
    return val_loss, correct, total

if __name__ == '__main__':
    args = parse()
    print(args)
    test(**args)
    