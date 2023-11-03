import numpy as np
from tqdm import tqdm
from utils.utils import AverageMeter
from utils.accuracy import accuracy
import torch

# test model
def test_epoch(model, test_loader, device):
    acc1 = AverageMeter()
    acc3 = AverageMeter()
    prediction = np.array([])
    label = np.array([])
    loop = tqdm(enumerate(test_loader), total = len(test_loader))
    for batch_idx, (batch_data, batch_label) in loop:
        batch_data = batch_data.to(device).float()
        batch_label = batch_label.to(device)
        
        batch_prediction = model(batch_data)

        # calculate the accuracy
        acc_batch = accuracy(batch_prediction, batch_label, topk=(1,3))
        n = batch_data.shape[0]
        batch_prediction = torch.argmax(batch_prediction, dim=1)

        # update the accuracy 
        acc1.update(acc_batch[0].data, n)
        acc3.update(acc_batch[1].data, n)

        prediction = np.append(prediction, batch_prediction.cpu().numpy())
        label = np.append(label, batch_label.data.cpu().numpy())

        loop.set_description(f'Test Epoch')
        loop.set_postfix({"test_accuracy1": str(round(acc1.average.item(), 3)) + "%", "test_accuracy3": str(round(acc3.average.item(), 3)) + "%"})
        
    return acc1.average.item(), acc3.average.item(), prediction, label
#-------------------------------------------------------------------------------