import time
import torch.utils.data
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import data
from evaluate import evaluate


learning_rate = 0.0001
epoch = 50
batch_size = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

attr_num = 18  # the number of attribute  item属性数
attr_present_dim = 5  # the dimention of attribute present 属性维度
hidden_dim = 100  # G hidden layer dimention G隐藏层维度
user_emb_dim = attr_num  # user维度与item一致
attr_dict_size = 2 * attr_num  # 原始数据0~35 因此词典大小:2*attr_num

g_loss_records = []
d_loss_records = []
g_path = []
d_path = []


# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.G_attr_matrix = nn.Embedding(attr_dict_size, attr_present_dim)
        # 正态分布
        for i in self.G_attr_matrix.modules():
            nn.init.xavier_normal_(i.weight)

        # 输入:attr_num * attr_present_dim
        # 输出:hidden_dim
        self.l1 = nn.Linear(attr_num * attr_present_dim, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim, bias=True)
        self.tanh= nn.Tanh()
        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_normal_(i.weight)
                nn.init.xavier_normal_(i.bias.unsqueeze(0))
            else:
                pass

    def forward(self, attribute_id):
        attribute_id = attribute_id.long()
        attr_present = self.G_attr_matrix(attribute_id)
        attr_feature = torch.reshape(attr_present, [-1, attr_num * attr_present_dim])

        # 输入:attr_feature [batch_size, attr_num * attr_present_dim]
        # 输出:user_generator [batch_size, user_emb_dim]
        output = self.tanh(self.l3(self.tanh(self.l2(self.tanh(self.l1(attr_feature))))))
        return output


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D_attr_matrix = nn.Embedding(attr_dict_size, attr_present_dim)
        for i in self.D_attr_matrix.modules():
            nn.init.xavier_normal_(i.weight)
        self.l1 = nn.Linear(attr_num * attr_present_dim + user_emb_dim, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim, bias=True)
        self.af = nn.Tanh()
        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_normal_(i.weight)
                nn.init.xavier_normal_(i.bias.unsqueeze(0))
            else:
                pass

    def forward(self, attribute_id, user_emb):
        attribute_id = attribute_id.long()
        attr_present = self.D_attr_matrix(attribute_id)
        attr_feature = torch.reshape(attr_present, [-1, attr_num * attr_present_dim])
        # 连接item向量和user向量 （用户-物品对）
        user_item_emb = torch.cat((attr_feature, user_emb.float()), 1)
        user_item_emb = user_item_emb.float()

        # 输入:user_item_emb[batch_size, attr_num*attr_present_dim + user_emb_dim]
        # 输出:[batch_size, user_emb_dim]
        output = self.l3(self.af(self.l2(self.af(self.l1(user_item_emb)))))
        y_prob = torch.sigmoid(output)  # 概率得分
        return y_prob

def train(train_data, neg_data, minLen):
    start_time = time.time()
    print("++++++++++++++++ Training on ", device, " +++++++++++++++")
    generator = Generator()
    discriminator = Discriminator()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=0)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=0)
    loss = nn.BCELoss(reduction='mean')  # 平均loss 交叉熵
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    max_n_10=0
    max_p_10=0
    max_n_20=0
    max_p_20=0
    for i in range(epoch):
        # D
        neg_iter = neg_data.__iter__()
        temp = 0
        for user, item, attr, user_emb in train_data:
            if batch_size * temp >= minLen:
                break
            # 正例
            attr = attr.to(device)
            user_emb = user_emb.to(device)
            # 负例
            neg_user, neg_item, neg_attr, neg_user_emb = neg_iter.next()
            neg_attr = neg_attr.to(device)
            neg_user_emb = neg_user_emb.to(device)
            # 生成
            gen_pos_user_emb = generator(attr)
            gen_neg_user_emb = generator(neg_attr)
            gen_pos_user_emb = gen_pos_user_emb.to(device)
            gen_neg_user_emb = gen_neg_user_emb.to(device)
            # D loss
            y_pos_d = discriminator(attr, user_emb)
            y_neg_d = discriminator(neg_attr, neg_user_emb)
            y_gen_pos_d = discriminator(attr, gen_pos_user_emb)
            y_gen_neg_d = discriminator(neg_attr, gen_neg_user_emb)
            d_optimizer.zero_grad()
            d_loss_pos = loss(y_pos_d, torch.ones_like(y_pos_d))  # 正例 1
            d_loss_neg = loss(y_neg_d, torch.zeros_like(y_neg_d))  # 负例 0
            d_loss_gen_pos = loss(y_gen_pos_d, torch.zeros_like(y_gen_pos_d))  # 生成 0
            d_loss_gen_neg = loss(y_gen_neg_d, torch.zeros_like(y_gen_neg_d))  # 生成 0
            d_loss = torch.mean(d_loss_pos + d_loss_neg + d_loss_gen_pos + d_loss_gen_neg)
            d_loss.backward()  # 反向传播，计算当前梯度
            d_optimizer.step()  # 根据梯度更新网络参数
            temp += 1

        # G
        for user, item, attr, user_emb in train_data:
            # G loss
            g_optimizer.zero_grad()
            attr = attr.long()
            attr = attr.to(device)
            gen_user_emb = generator(attr)
            gen_user_emb = gen_user_emb.to(device)
            y_gen = discriminator(attr, gen_user_emb)
            y_true = torch.ones_like(y_gen)
            g_loss = loss(y_gen, y_true)
            g_loss.backward()
            g_optimizer.step()

        g_loss_records.append(g_loss.item())
        d_loss_records.append(d_loss.item())

        total_time = time.time() - start_time

        print("epoch:", i + 1,"total_time:", total_time, "d_loss:", d_loss.item(), ",g_loss:", g_loss.item())

        item, attr = data.load_test_data()
        item = torch.tensor(item)
        attr = torch.tensor(attr, dtype=torch.long)
        item = item.to(device)
        attr = attr.to(device)
        gen_user = generator(attr)
        rec_user = recommendation(gen_user, 10)

        n_10, p_10, n_20, p_20 = evaluate(item, rec_user)
        if n_10>max_n_10:
            max_n_10 = n_10
        if n_20 > max_n_20:
            max_n_20=n_20
        if p_10>max_p_10:
            max_p_10=p_10
        if p_20>max_p_20:
            max_p_20=p_20

        print("\t\tndcg_10:", n_10, "ndcg_20:",n_20,",precision_10:", p_10,"precision_20:",p_20)
        print("\t\tmax_n_10:",max_n_10,"max_n_20:",max_n_20,"max_p_10:",max_p_10,"max_p_20:",max_p_20)


# 生成推荐用户

def recommendation(gen_user, k):
    # print(gen_user)
    gen_user = gen_user.to(device)
    user_attribute_matrix = pd.read_csv('data/user_attribute.csv', header=None)
    user_attribute_matrix = torch.tensor(np.array(user_attribute_matrix[:]), dtype=torch.float)
    user_embed_matrix = user_attribute_matrix.to(device)
    similar_matrix = torch.matmul(gen_user, user_embed_matrix.t())  #
    index = torch.argsort(-similar_matrix)
    torch.set_printoptions(profile="full")
    rec_users = index[:, 0:k]
    return rec_users

if __name__ == "__main__":

    # 使用MovieLens-1M数据集预处理成正例负例文件
    train_dataset = data.Dataset('data/train_data.csv', 'data/user_emb.csv')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    neg_dataset = data.Dataset('data/neg_data.csv', 'data/user_emb.csv')
    neg_loader = torch.utils.data.DataLoader(neg_dataset, batch_size=1024, shuffle=True, num_workers=0)

    # 训练
    model_path, _, g_loss, d_loss = train(train_loader, neg_loader, min(neg_dataset.__len__(), train_dataset.__len__()))
