import argparse
import time

import torch
from torchvision import datasets
from torchvision.transforms import transforms

from model import VGGNet
from test_model import VGGNetTest

MEAN_LIST = [0.507, 0.487, 0.441]
IMG_SIZE, SMALL_SCALE, LARGE_SCALE, SCALE_RANGE = 32, 36, 56, [36, 72]
SCALE_LIST_DICT = {SMALL_SCALE: [32, 36, 40], LARGE_SCALE: [48, 56, 64]}

def parse():
    parser = argparse.ArgumentParser(description='VGGNet Testing')
    parser.add_argument('--multi_scale', '-ms', type=int, default=0)
    parser.add_argument('--is_scale_fix', '-sf', type=int, default=1)
    parser.add_argument('--scale', '-s', type=int, default=SMALL_SCALE)
    parser.add_argument('--data_path', '-dp', type=str, default='./data')
    parser.add_argument('--load_path', '-lp', type=str, default='./model/model.pt')
    parser.add_argument('--config', '-c', type=str, default='A')

    args = parser.parse_args()
    return vars(args)

def replace(key):
    key = key.replace('3', '2')
    key = key.replace('6', '4')
    return key

def load_model_weights(test_model, train_model):
    test_model_dict = test_model.state_dict()
    
    encoder_dict = {k:v for k, v in train_model.state_dict().items() if k.startswith('encoder')}
    test_model_dict.update(encoder_dict)
    
    decoder_dict = {replace(k):v for k, v in train_model.state_dict().items() if k.startswith('decoder')}
    for k, v in decoder_dict.items():
        if k.endswith('weight'):
            decoder_dict[k] = v.unsqueeze(2).unsqueeze(3)
    test_model_dict.update(decoder_dict)
    
    test_model.load_state_dict(test_model_dict)
    return test_model

def test(multi_scale=0, is_scale_fix=1, scale=SMALL_SCALE, data_path='./data', load_path='./model/model.pt', config='A'):
    multi_scale, is_scale_fix = bool(multi_scale), bool(is_scale_fix)
    if not multi_scale:
        if not is_scale_fix:
            scale = int(sum(SCALE_RANGE) / len(SCALE_RANGE))
        scale_list = [scale]
    else:
        if is_scale_fix:
            scale_list = SCALE_LIST_DICT[scale]
        else:
            scale_list = [SCALE_RANGE[0], int(sum(SCALE_RANGE) / len(SCALE_RANGE)), SCALE_RANGE[1]]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_model = VGGNet(config=config, num_classes=100, img_size=IMG_SIZE).to(device, dtype=torch.float32)
    train_model.load_state_dict(torch.load(load_path))
    test_model = VGGNetTest(config=config, num_classes=100, img_size=IMG_SIZE).to(device, dtype=torch.float32)
    
    test_model = load_model_weights(test_model, train_model)

    transform_list = [
        transforms.ToTensor(),
     	transforms.Normalize(MEAN_LIST, [1, 1, 1])]
    transform = transforms.Compose(transform_list)
    
    test_dataset = datasets.CIFAR100(root=data_path, train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    loss_function = torch.nn.CrossEntropyLoss()
    
    start = time.time()
    test_loss, correct, total = validate(scale_list, loss_function, test_loader, test_model, device)
    print ("Test Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            test_loss / total, correct, total, 100 * correct / total))
    
    end = time.time()
    print("Time ellapsed in testing is: {:.4f}".format(end - start))

def validate(scale_list, loss_function, valid_loader, model, device):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    
    flip = transforms.RandomHorizontalFlip(p=1)
    resize = {scale:transforms.Resize(scale) for scale in scale_list}
    
    with torch.no_grad():
        for images, labels in valid_loader:
            output_list = list()
            for scale in scale_list:
                images, labels = images.to(device, dtype=torch.float32), labels.to(device)
                images = resize[scale](images)
            
                output = model(images).mean(dim=(-1, -2))            
                flip_output = model(flip(images)).mean(dim=(-1, -2))            
                output = (output + flip_output) / 2
                
                output_list.append(output)
                
            output = torch.stack(output_list, -1).mean(dim=-1)
            
            val_loss += loss_function(output, labels).item() * labels.size(0)
            
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            
            total += labels.size(0)
            
    return val_loss, correct, total

if __name__ == '__main__':
    args = parse()
    print(args)
    test(**args)