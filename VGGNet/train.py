import os
import argparse
import time
from os.path import join as opj
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
from torchvision import datasets
from torchvision.transforms import transforms

from advanced_transforms import RandomResize, AddColorShift
from model import VGGNet

MEAN_LIST = [0.507, 0.487, 0.441]
WEIGHT_LIST = ('encoder.0.0.0', 'encoder.1.0.0', 'encoder.2.0.0', 'encoder.2.1.0',
               'decoder.0', 'decoder.3', 'decoder.6')
IMG_SIZE, SMALL_SCALE, LARGE_SCALE, SCALE_RANGE = 32, 36, 56, [36, 72]

def parse():
    parser = argparse.ArgumentParser(description='VGGNet Training')
    parser.add_argument('--is_scale_fix', '-sf', type=int, default=1)
    parser.add_argument('--scale', '-s', type=int, default=SMALL_SCALE)
    parser.add_argument('--data_path', '-dp', type=str, default='./data')
    parser.add_argument('--save_path', '-sp', type=str, default='./model')
    parser.add_argument('--load_path', '-lp', type=str, default='./model/model.pt')
    parser.add_argument('--config', '-c', type=str, default='A')

    args = parser.parse_args()
    return vars(args)

def train(is_scale_fix=1, scale=SMALL_SCALE, data_path='./data', save_path='./model', load_path='./model/model.pt', config='A'):
    is_scale_fix = bool(is_scale_fix)
    if is_scale_fix:
        save_path = opj(save_path, f'scale{scale}_{config}')
    else:
        save_path = opj(save_path, f'scale_range_{config}')
    os.makedirs(save_path, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VGGNet(config=config, num_classes=100, img_size=IMG_SIZE).to(device, dtype=torch.float32)
    if is_scale_fix and scale == SMALL_SCALE:
        lr = 0.01
        if config != 'A':
            pre_A_dict = {k:v for k, v in torch.load(load_path) if k.startswith(WEIGHT_LIST)}
            model_dict = model.state_dict()
            model_dict.update(pre_A_dict)
            model.load_state_dict(model_dict)
    else:
        lr = 0.001
        model.load_state_dict(torch.load(load_path))
    
    transform_list = [
        transforms.ToTensor(),
     	transforms.Normalize(MEAN_LIST, [1, 1, 1])]
    if is_scale_fix:
        transform_list.append(transforms.Resize(scale))
    else:
        transform_list.append(RandomResize(SCALE_RANGE))
    transform_list.append(transforms.RandomCrop(IMG_SIZE))
    transform_list.append(AddColorShift())
    transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    transform = transforms.Compose(transform_list)
    
    train_dataset = datasets.CIFAR100(root=data_path, train=True, transform=transform, download=True)
    valid_dataset = datasets.CIFAR100(root=data_path, train=False, transform=transform, download=True)
    
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, eps=0.0)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=128, shuffle=False)
    
    start = time.time()
    plot_list = [[], [], []]
    for epoch in range(200):
        model.train()
        print("{}th epoch starting.".format(epoch+1))
        
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}', unit='img', ncols=80) as pbar:
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device, dtype=torch.float32), labels.to(device)
        
                optimizer.zero_grad()
                train_loss = loss_function(model(images), labels)
                train_loss.backward()
                
                optimizer.step()
                
                pbar.update(images.shape[0])
        
        val_loss, correct, total = validate(loss_function, valid_loader, model, device)
        scheduler.step(val_loss)
    
        print ("Epoch [{}] Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            epoch+1, val_loss / total, correct, total, 100 * correct / total))
        torch.save(model.state_dict(), opj(save_path, f'epoch{str(epoch+1).zfill(4)}.pt'))
        
        plot_list[0].append(train_loss.item())
        plot_list[1].append(val_loss / total)
        plot_list[2].append(correct / total)
        
        if optimizer.param_groups[0]['lr'] <= lr * 1e-4:
            break
    
    end = time.time()
    print("Time ellapsed in training is: {}".format(end - start))
    
    save_plot(plot_list, save_path)

def validate(loss_function, valid_loader, model, device):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device, dtype=torch.float32), labels.to(device)
            
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