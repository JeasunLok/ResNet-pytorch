import torch
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = output.size(0)
	
	# 在输出结果中取前maxk个最大概率作为预测结果，并获取其下标，当topk=(1, 5)时取5就可以了。
    _, pred = torch.topk(output, k=maxk, dim=1, largest=True, sorted=True)
    
    # 将得到的k个预测结果的矩阵进行转置，方便后续和label作比较
    pred = pred.T
    # 将label先拓展成为和pred相同的形状，和pred进行对比，输出结果
    correct = torch.eq(pred, label.contiguous().view(1,-1).expand_as(pred))
	# 例：
	# 若label为：[1,2,3,4], topk = (1, 5)时
	# 则label.contiguous().view(1,-1).expand_as(pred)为：
	# [[1, 2, 3, 4],
	#  [1, 2, 3, 4],
	#  [1, 2, 3, 4],
	#  [1, 2, 3, 4],
	#  [1, 2, 3, 4]]
	
    res = []

    for k in topk:
    	# 取前k个预测正确的结果进行求和
        correct_k = correct[:k].contiguous().view(-1).float().sum(dim=0, keepdim=True)
        # 计算平均精度， 将结果加入res中
        res.append(correct_k*100/batch_size)
    return res

def output_metrics(prediction, label):
    CM = confusion_matrix(label, prediction)
    weighted_recall = recall_score(label, prediction, average="weighted")
    weighted_precision = precision_score(label, prediction, average="weighted")
    weighted_f1 = f1_score(label, prediction, average="weighted")
    return CM, weighted_recall, weighted_precision, weighted_f1
