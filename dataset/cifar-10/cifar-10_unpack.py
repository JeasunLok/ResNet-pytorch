import torchvision
from torch.utils.data import DataLoader
import os
from imageio import imsave

train_data = torchvision.datasets.CIFAR10(root="dataset/cifar-10", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="dataset/cifar-10", train=False, transform=torchvision.transforms.ToTensor(), download=True)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 训练集
filename = './dataset/cifar-10/cifar-10-batches-py'  # 图片的路径
meta = unpickle(filename + '/batches.meta')
label_name = meta[b'label_names']
print(label_name) #打印标签

for i in range(len(label_name)):  #建立文件夹train
    file = label_name[i].decode()
    path = './dataset/cifar-10/train/' + file
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

for i in range(1, 6):
    content = unpickle(filename + '/data_batch_' + str(i))  # 解压后的每个data_batch_
    for j in range(10000):
        img = content[b'data'][j]
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)
        img_name = './dataset/cifar-10/train/' + label_name[content[b'labels'][j]].decode() + '/batch_' + str(i) + '_num_' + str(j) + '.jpg'
        imsave(img_name, img)

# 训练集改名
path = './dataset/cifar-10/train/'
filelist = os.listdir(path)
for item in filelist:
    pathnew=os.path.join(path,item)
    imagelist = os.listdir(pathnew)
    j = 1
    for i in imagelist:
        src = os.path.join(os.path.abspath(pathnew), i)
        dst = os.path.join(os.path.abspath(pathnew), '' + item + '.' + str(j) + '.jpg')
        j = j+1
        os.rename(src, dst)

# 测试集
meta1 = unpickle(filename + '/test_batch')  # 解压test_batch
label_name1 = meta[b'label_names']

for i in range(len(label_name1)):
    file = label_name1[i].decode()
    path = './dataset/cifar-10/test/' + file
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

for j in range(10000):
    img = meta1[b'data'][j]
    img = img.reshape(3, 32, 32)
    img = img.transpose(1, 2, 0)
    img_name = './dataset/cifar-10/test/' + label_name[
        meta1[b'labels'][j]].decode() + '/batch_' + str(j) + '_num_' + str(j) + '.jpg'
    imsave(img_name, img)

# 测试集改名
path = './dataset/cifar-10/test'
filelist = os.listdir(path)
for item in filelist:
    pathnew=os.path.join(path,item)
    imagelist = os.listdir(pathnew)
    j = 1
    for i in imagelist:
        src = os.path.join(os.path.abspath(pathnew), i)
        dst = os.path.join(os.path.abspath(pathnew), '' + item + '.' + str(j) + '.jpg')
        j = j+1
        os.rename(src, dst)


