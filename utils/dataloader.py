from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision.io import read_image

class MyDataset(data.Dataset):
    def __init__(self, annotation_lines, input_shape, transform=None):
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.transform = transform

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
        image = Image.open(annotation_path)

        if self.transform is not None:
            image = self.transform(image) 
        else:
            image = torch.from_numpy(np.transpose(np.array(image), [2, 0 ,1]))

        label = int(self.annotation_lines[index].split(';')[0])
        
        return image, label
