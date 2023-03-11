# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 18:49:54 2021

@author: Dongil Ryu
"""
import os
import argparse
import time
from os.path import join as opj
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
from torchvision import datasets
from torchvision.transforms import transforms

from model import DenseNet

MEAN_LIST, STD_LIST = [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
IMG_SIZE = 32

def parse():
    parser = argparse.ArgumentParser(description='DenseNet Training')
    parser.add_argument('--data_path', '-dp', type=str, default='./data')
    parser.add_argument('--save_path', '-sp', type=str, default='./model')
    parser.add_argument('--L', '-l', type=int, default=40)
    parser.add_argument('--k', '-k', type=int, default=12)
    parser.add_argument('--is_BC', '-bc', type=int, default=0)
    parser.add_argument('--is_aug', '-aug', type=int, default=0)

    args = parser.parse_args()
    return vars(args)

def train(data_path='./data', save_path='./model', L=40, k=12, is_BC=0, is_aug=0):
    is_BC, is_aug = bool(is_BC), bool(is_aug)
    theta = 0.5 if is_BC else 1
    save_path = opj(save_path, f'L{L}_k{k}_isBC{is_BC}_isAUG{is_aug}')
    os.makedirs(save_path, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DenseNet(L=L, k=k, theta=theta, is_BC=is_BC, is_aug=is_aug, num_classes=100).to(device)

    
    transform = transforms.Compose([
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_LIST, std=STD_LIST)
    ])
    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_LIST, std=STD_LIST)
    ])
    if not is_aug:
        transform = transform_valid
    
    train_dataset = datasets.CIFAR100(root=data_path, train=True, transform=transform, download=True)
    valid_dataset = datasets.CIFAR100(root=data_path, train=False, transform=transform_valid, download=True)
    
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    start = time.time()
    plot_list = [[], [], []]
    for epoch in range(300):
        model.train()
        print("{}th epoch starting.".format(epoch+1))
        
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}', unit='img', ncols=80) as pbar:
            for i, (images, labels) in enumerate(train_loader):
                if torch.cuda.is_available():
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                else:
                    images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                train_loss = loss_function(model(images), labels)
                train_loss.backward()
                print(train_loss.item())
                
                optimizer.step()
                
                pbar.update(images.shape[0])
        
        val_loss, correct, total = validate(loss_function, valid_loader, model, device)
    
        print ("Epoch [{}] Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            epoch+1, val_loss / total, correct, total, 100 * correct / total))
        torch.save(model.state_dict(), opj(save_path, f'epoch{str(epoch+1).zfill(4)}.pt'))
        
        plot_list[0].append(train_loss.item())
        plot_list[1].append(val_loss / total)
        plot_list[2].append(correct / total)
        
        if epoch in (150, 225):
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
    
    end = time.time()
    print("Time ellapsed in training is: {}".format(end - start))
    
    save_plot(plot_list, save_path)

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

def save_plot(plot_list, save_path):
    x = list(range(len(plot_list[0])))
    plt.plot(x, plot_list[0], x, plot_list[1], x, plot_list[2])
    plt.legend(('train', 'val', 'acc'))
    plt.savefig(f'{save_path.split("/")[-1]}.png', dpi=300)

if __name__ == '__main__':
    args = parse()
    print(args)
    train(**args)
