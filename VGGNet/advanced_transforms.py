from collections.abc import Sequence
from collections import defaultdict

import torch
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn.functional as F
import numpy as np

from sklearn.decomposition import PCA

MEAN_LIST = [0.485, 0.456, 0.406]


class RandomResize(torch.nn.Module):

    def __init__(self, size_range):
        super().__init__()
        if not isinstance(size_range, Sequence):
            raise TypeError("Size Range should be int or sequence. Got {}".format(type(size_range)))
        if isinstance(size_range, Sequence) and len(size_range) != 2:
            raise ValueError("Size Range should have 2 values")
        self.size_dict = defaultdict(int)
        self.size_range = size_range

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        img_key = img.numpy().tobytes()
        if self.size_dict[img_key] == 0:
            self.size_dict[img_key] = np.random.randint(*self.size_range)
        
        return F.interpolate(img, self.size_dict[img_key])

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size_range)


class AddColorShift(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        eigen_vectors, eigen_values = get_eigen_value_vector()
        self.eigen_vectors = torch.from_numpy(eigen_vectors)
        self.eigen_values = torch.from_numpy(eigen_values).view(3, 1)
    
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Color Shift Added Image.
        """
        r = torch.normal(mean=0, std=0.1, size=(3, 1))
        shift = torch.matmul(self.eigen_vectors, r * self.eigen_values)
        shift = shift.view(3).numpy()
        
        if len(img.size()) == 3:
            img = img.permute(1, 2, 0).contiguous()
            img = img + shift
            img = img.permute(2, 0, 1).contiguous()
        if len(img.size()) == 4:
            img = img.permute(0, 2, 3, 1).contiguous()
            img = img + shift
            img = img.permute(0, 3, 1, 2).contiguous()
        
        return img
        
    
    def __repr__(self):
        return self.__class__.__name__ + '(eigen_vectors={0}, eigen_values={1})'.format(self.eigen_vectors, self.eigen_values)


def get_eigen_value_vector(data_path='./data', transform=None):
    if not transform:
        transform = transforms.Compose([
            transforms.ToTensor(),
        	transforms.Normalize(MEAN_LIST, [1, 1, 1])])
    
    train_dataset = datasets.CIFAR100(root=data_path, train=True, transform=transform, download=True)
    
    s = torch.stack(tuple(x[0] for x in train_dataset))
    s = s.permute(0, 2, 3, 1).contiguous().view(-1, 3)

    pca = PCA(n_components=3, random_state=2071621)
    pca.fit(s)
    
    return pca.components_.T, pca.explained_variance_