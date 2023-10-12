
from torch.utils.data import Dataset

import torch

from Dataset.predate import readdata

def normalization(data):
    """
    归一化函数
    把所有数据归一化到[0，1]区间内，数据列表中的最大值和最小值分别映射到1和0，所以该方法一定会出现端点值0和1。
    此映射是线性映射，实质上是数据在数轴上等比缩放。

    :param data: 数据列表，数据取值范围：全体实数
    :return:
    """
    min_value = min(data)
    max_value = max(data)
    new_list = []
    for i in data:
        new_list.append((i - min_value) / (max_value - min_value))
    return new_list






class MyDataset(Dataset):

    def __init__(self, dataPath, args, position_encode, label_type):
        self.args = args
        self.label_type = label_type
        self.position_encode = position_encode
        self.data = readdata(dataPath, args, position_encode)


    def __getitem__(self, index):


   
        dic = self.data[index]

        return torch.tensor(dic['batch_x']), torch.tensor(dic['batch_y']), torch.tensor(dic['batch_y_lat']), torch.tensor(dic['batch_y_lng']),torch.tensor(dic['epicenter_dis']),dic['epicenter_dis_norm'], torch.tensor(dic['label_depth']), dic['label_depth_norm'], torch.tensor(dic['label_ml']), torch.tensor(dic['label_arrivetime']), dic['label_arrivetime_norm'], torch.tensor(dic['receive_location']), torch.tensor(dic['label_aiz']), dic['label_aiz_norm']



    def __len__(self):
        return len(self.data)

