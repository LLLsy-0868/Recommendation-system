from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


# 重构数据集
class Dataset(Dataset):
    def __init__(self, train_csv, user_emb_matrix):
        self.train_csv = pd.read_csv(train_csv, header=None)
        self.user = self.train_csv.loc[:, 0]
        self.item = self.train_csv.loc[:, 1]
        self.attr = self.train_csv.loc[:, 2]
        self.user_emb_matrix = pd.read_csv(user_emb_matrix, header=None)
        self.user_emb_values = np.array(self.user_emb_matrix[:])

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, idx):
        user = self.user[idx]
        item = self.item[idx]
        user_emb = self.user_emb_values[user]
        # user, item, attr, user_emb
        attr = self.attr[idx][1:-1].split()
        attr = torch.tensor(list([int(item) for item in attr]), dtype=torch.long)
        attr = np.array(attr)
        return user, item, attr, user_emb


"""
# 获取一个batch neg_data
def get_neg_batch(start_index, batch_size):
    neg = np.array(pd.read_csv('neg_data.csv',header =None))
    batch_data = neg[start_index: start_index+batch_size]
    user_batch = [x[0] for x in batch_data]
    item_batch = [x[1] for x in batch_data]
    attr_batch = [x[2][1:-1].split() for x in batch_data]
    user_emb_matrix = np.array(pd.read_csv(r'util/user_emb.csv',header=None))
    real_user_emb_batch = user_emb_matrix[user_batch]

    return user_batch,item_batch,attr_batch,real_user_emb_batch
"""


def load_test_data():
    test_item = pd.read_csv('data/test_item.csv', header=None).loc[:]
    test_item = np.array(test_item)
    test_attribute = pd.read_csv('data/test_attribute.csv', header=None).loc[:]
    test_attribute = np.array(test_attribute)
    return test_item, test_attribute