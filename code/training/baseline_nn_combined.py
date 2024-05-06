import datetime
# import talib
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import requests
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
import _pickle as cPickle
sys.path.insert(0, '../model_structure')
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = '../extraction/eb_target/bert_token_WMT.pkl'
csv_path = "DCNN_Final_more2.csv"
model_path = '../models/model.pt'
model_path_last = '_last.pt'.join(model_path.split('.pt'))

config = {
    'learning_rate': 0.001,
    'weight_decay': 1e-7,
    'batch_size': 32,
    'n_epochs': 100,
}


device = get_device()
print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# # Prepare Dataset
class EssayDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        if y is not None:
            self.y = y
        else:
            self.y = None

        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        # if self.y is not None:
        return self.X[index], self.y[index]

        # else:
        #     return self.a0[index], self.p[index], self.a1[index]

    def __len__(self):
        return self.n_samples

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# T
with open(csv_path, "w") as f:
    f.write(f'ticker,channels,train_acc,val_acc,test_acc,size\n')
def normalize(x):
    return ((x - x.min()) / (
            x.max() - x.min()))  # .values


def stationary(x):
    return (x / x.shift(1) - 1)



def d100(x):
    return (x[0] / 100, x[1] / 100)


with open(rf"../embeddings/transformer/y.pickle", "rb") as input_file:
    y = cPickle.load(input_file)
y_df = y
labels = {}
elements = {
    "leo":0,
    "aries":0,
    "sagittarius":0,
    "cancer":1,
    "scorpio":1,
    "pisces": 1,
    "gemini":2,
    "libra": 2,
    "aquarius":2,
    "taurus": 3,
    "virgo": 3,
    "capricorn": 3
}
for i in range(len(y.unique())):
    labels[y.unique()[i]] = i
def label_encode(x):
    # return math.sqrt(x)
    return labels[x]
def element_encode(x):
    # return math.sqrt(x)
    return elements[x]
y = np.array(list(map(label_encode, y_df)))
with open(rf"../embeddings/transformer/transformer_essay{1}.pickle", "rb") as input_file:
    e = cPickle.load(input_file)
for essay in range(2, 10):
    with open(rf"../embeddings/transformer/transformer_essay{essay}.pickle", "rb") as input_file:
        e2 = cPickle.load(input_file)
    e = np.concatenate((e,e2), axis = 1)
x_df = e


total = y.shape[0]
VAL_RATIO = 0.1
TEST_RATIO = 0.1
train_set_size = int(total * (1 - VAL_RATIO - TEST_RATIO))
valid_set_size = int(total * VAL_RATIO)
test_set_size = total - train_set_size - valid_set_size
tensor_S = torch.from_numpy(np.array(x_df)).to(device).float()
tensor_y = torch.from_numpy(np.array(y)).to(device).long()
dataset_test = EssayDataset(tensor_S[0:test_set_size], tensor_y[0:test_set_size])
dataset_val = EssayDataset(tensor_S[test_set_size:test_set_size + valid_set_size],
                           tensor_y[test_set_size:test_set_size + valid_set_size])
dataset = EssayDataset(tensor_S[test_set_size + valid_set_size::],
                       tensor_y[test_set_size + valid_set_size::])

BATCH_SIZE = config['batch_size']

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, num_workers=0)
test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=0)
train_samples = len(train_loader.dataset)
val_samples = len(val_loader.dataset)
test_samples = len(test_loader.dataset)
total = train_samples + val_samples + test_samples
total_samples = train_samples
print(f'total: {total}, train: {train_samples}, val: {val_samples}, test: {test_samples}')
# print_proportions(train_loader,val_loader, test_loader)
n_iterations = math.ceil(total_samples / BATCH_SIZE)


# fix random seed for reproducibility
same_seeds(2022)

# training parameters
num_epoch = config['n_epochs']

from baseline_nn_model import MLP, MLP_Combined

model = MLP_Combined(len(labels))
model = model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
list(model.parameters())

best_acc = 0.0
train_accs = []
val_accs = []
train_losses = []
val_losses = []

for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train()  # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs.float())
        batch_loss = criterion(outputs, labels)

        _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        batch_loss.backward()
        optimizer.step()

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()


    # validation
    if val_samples > 0:
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.float())
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)

                val_acc += (
                        val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / train_samples, train_loss / len(train_loader),
                val_acc / val_samples, val_loss / len(val_loader)
            ))

            train_accs.append(train_acc / train_samples)
            val_accs.append(val_acc / val_samples)
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc / val_samples))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / train_samples, train_loss / len(train_loader)
        ))

    if epoch % 10 == 0:
        plt.plot(train_losses)
        plt.plot(val_losses)

        # plt.show()
        #
        # plt.plot(train_accs)
        # plt.plot(val_accs)
        plt.savefig(f'../plots/baseline/combined/baseline_combined.png')
        plt.show()

# if not validating, save the last epoch
if valid_set_size == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')

pt = torch.load(model_path, map_location=device)
model.load_state_dict(pt)

test_acc = 0
test_loss = 0
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs.float())
        batch_loss = criterion(outputs, labels)
        _, test_pred = torch.max(outputs, 1)

        test_acc += (
                test_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
        test_loss += batch_loss.item()

    print(test_acc / test_samples)
# with open(csv_path, "a") as f:
#     f.write(
#         f'{t},{channels},{train_acc / train_samples},{val_acc / val_samples},{test_acc / test_samples},{total}\n')
#
# plt.plot(train_losses)
# plt.plot(val_losses)
# plt.show()
# plt.savefig(f'plots/baseline/combined/baseline_combined.png')