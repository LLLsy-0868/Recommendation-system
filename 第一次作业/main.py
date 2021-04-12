from __future__ import print_function

from PMF import *

print('PMF Recommendation Model Example')

# choose dataset name and load dataset, 'ml-1m', 'ml-10m'
dataset = 'ml-100k'
processed_data_path = os.path.join(os.getcwd(), 'data', dataset)
user_id_index = pickle.load(open(os.path.join(processed_data_path, 'user_id_index.pkl'), 'rb'), encoding='bytes')
item_id_index = pickle.load(open(os.path.join(processed_data_path, 'item_id_index.pkl'), 'rb'), encoding='bytes')
data = np.loadtxt(os.path.join(processed_data_path, 'data.txt'), dtype=float)
print(data.shape[0], data.shape[1])
# set split ratio
ratio = 0.7
train_data = data[:int(ratio*data.shape[0])]
vali_data = data[int(ratio*data.shape[0]):int(0.9*data.shape[0])]
test_data = data[int(0.9*data.shape[0]):]

NUM_USERS = max(user_id_index.values()) + 1
NUM_ITEMS = max(item_id_index.values()) + 1
print(NUM_USERS, NUM_ITEMS)
print('dataset density:{:f}'.format(len(data)*1.0/(NUM_USERS*NUM_ITEMS)))


R = np.zeros([NUM_USERS, NUM_ITEMS])
for ele in train_data:
    R[int(ele[0]), int(ele[1])] = float(ele[2])

# construct model
print('training model.......')
lambda_alpha = 0.01
lambda_beta = 0.01
latent_size = 20
lr = 0.001#3e-5
iters = 200
PMF = PMF(R=R, lambda_alpha=lambda_alpha, lambda_beta=lambda_beta,
            latent_size=latent_size, momuntum=0.9, lr=lr, iters=iters, seed=1)
print('parameters are:ratio={:f}, reg_u={:f}, reg_v={:f}, latent_size={:d}, lr={:f}, iters={:d}'.format(ratio, lambda_alpha, lambda_beta, latent_size,lr, iters))
U, V, train_loss_list, vali_rmse_list = PMF.train(train_data=train_data, vali_data=vali_data)

print('testing model.......')
preds = PMF.predict(data=test_data)
test_rmse = PMF.RMSE(preds, test_data[:, 2])

print('test rmse:{:f}'.format(test_rmse))
