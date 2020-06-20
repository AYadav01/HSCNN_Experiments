import pandas as pd
import numpy as np
import torch

def class_weight(train_path, num_tasks=3):
    df = pd.read_csv(train_path, header=None)
    label_list = []
    for i in range(1, num_tasks + 2):
        label_list.append(df.iloc[:, i].tolist())
    class_weight_dict = {}
    for i in range(len(label_list)):
        labels = label_list[i]
        num_classes = len(np.unique(labels))
        weight_list = []
        for j in range(num_classes):
            count = float(labels.count(int(j)))
            weight = 1 / (count / float(len(labels)))
            weight_list.append(weight)
        class_weight_dict[i] = torch.FloatTensor(weight_list).cuda()
    return class_weight_dict