import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.ResNet import *
from utils.dataloader import *
from utils.utils import *
from utils.accuracy import *
from train import *
from test import *

if __name__ == "__main__":
    classes_path = 'dataset/cls_classes.txt'
    cuda = False
    dp = False
    input_shape = [200, 200]
    pretrained = False
    model_path = ""
    epoch = 2
    lr = 0.001
    momentum = 0.9
    weight_decay = 5e-4
    batch_size = 32
    save_period = epoch // 2
    logs_dir = 'logs'
    checkpoints_dir = "checkpoints"
    train_annotation_path = "images/cls_train.txt"
    val_annotation_path = "images/cls_val.txt"
    test_annotation_path = "images/cls_test.txt"

    if cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    
    time_now = time.localtime()
    logs_folder = os.path.join(logs_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time_now))
    checkpoints_folder = os.path.join(checkpoints_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time_now))
    os.makedirs(logs_folder)
    os.makedirs(checkpoints_folder)

    class_names, num_classes = get_classes(classes_path)
    model = resnet34(num_classes=num_classes)

    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("load model path : " + model_path)
    else:
        # kaiming initialization
        # 适用于ReLU初始化二维卷积和线性层
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
    print("===============================================================================")

    if dp:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        print("DP model initialized!")
        print("===============================================================================")
    
    model = model.to(device)
        
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_annotation_path, encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding="utf-8") as f:
        val_lines = f.readlines()
    with open(test_annotation_path, encoding="utf-8") as f:
        test_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    num_test = len(test_lines)
    np.random.seed(3047)
    np.random.shuffle(train_lines)

    print("device:", device, "num_train:", num_train, "num_val:", num_val, "num_test:", num_test)
    print("===============================================================================")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch//2, gamma=0.9) 
    criterion = nn.CrossEntropyLoss().cuda()

    image_transform = get_transform(input_shape, IsResize=True, IsTotensor=True, IsRandomRotation=True) # set IsResize=True to make sure it works
    train_dataset = MyDataset(train_lines, input_shape=input_shape, transform=image_transform)
    val_dataset = MyDataset(val_lines, input_shape=input_shape, transform=image_transform)
    test_dataset = MyDataset(test_lines, input_shape=input_shape, transform=image_transform)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    #---------------------------------------#
    #   开始模型训练
    #---------------------------------------#
    print("start training")
    epoch_result = np.zeros([4, epoch])
    for e in range(epoch):
        model.train()
        train_acc1, train_acc3, train_loss = train_epoch(model, train_loader, criterion, optimizer, e, epoch, device)
        scheduler.step()
        print("Epoch: {:03d} | train_loss: {:.4f} | train_acc1: {:.2f}% | train_acc3: {:.2f}%".format(e+1, train_loss, train_acc1, train_acc3))
        epoch_result[0][e], epoch_result[1][e], epoch_result[2][e], epoch_result[3][e]= e+1, train_loss, train_acc1, train_acc3

        if ((e+1) % save_period == 0) | (e == epoch - 1):
            print("===============================================================================")
            print("start validating")
            model.eval()      
            val_acc1, val_acc3, val_loss, val_prediction, val_label = valid_epoch(model, val_loader, criterion, device)
            val_CM, val_weighted_recall, val_weighted_precision, val_weighted_f1 = output_metrics(val_label, val_prediction)
            if (e != epoch -1):
                print("Epoch: {:03d}  =>  Accuracy: {:.2f}% | W-Recall: {:.4f} | W-Precision: {:.4f} | W-F1: {:.4f}".format(e+1, val_acc1, val_weighted_recall, val_weighted_precision, val_weighted_f1))
            torch.save(model, checkpoints_folder + r"\\model_loss" + str(round(val_loss, 4)) + "_epoch" + str(e+1) + ".pth")
            torch.save(model.state_dict(), checkpoints_folder + r"\\model_state_dict_loss" + str(round(val_loss, 4)) + "_epoch" + str(e+1) + ".pth")
            print("===============================================================================")
    
    draw_result_visualization(logs_folder, epoch_result)
    print("save train logs successfully")
    print("===============================================================================")

    print("start testing")
    model.eval()
    test_acc1, test_acc3, test_prediction, test_label = test_epoch(model, val_loader, device)
    test_CM, test_weighted_recall, test_weighted_precision, test_weighted_f1 = output_metrics(test_label, test_prediction)
    print("Test Result  =>  Accuracy: {:.2f}%| W-Recall: {:.4f} | W-Precision: {:.4f} | W-F1: {:.4f}".format(test_acc1, test_weighted_recall, test_weighted_precision, test_weighted_f1))
    store_result(logs_folder, test_acc1, test_weighted_recall, test_weighted_precision, test_weighted_f1, test_CM, epoch, batch_size, lr, weight_decay)
    print("save test result successfully")
    print("===============================================================================")
