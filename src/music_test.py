#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This code is made by referring to https://github.com/locuslab/TCN
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import time
import argparse
import numpy as np

from mdata_load import Music_Sequence
from rnn_models import Basic_MLP, RNN_tanh, RNN_LSTM, RNN_GRU


# In[2]:


parser = argparse.ArgumentParser(description='RNN, LSTM, GRU - Polyphonic Music')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (default: 0)')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 500)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval (default: 20')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='RMSprop',
                    help='optimizer to use (default: RMSprop)')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer (default: 100)')
parser.add_argument('--data', type=str, default='nottingham',
                    help='the dataset to run (default: nottingham)')
parser.add_argument('--seed', type=int, default=190315,
                    help='random seed (default: 190315)')
parser.add_argument('--model-name', type=str, default='RNN',
                    help='the dataset to run (default: MLP)')

args = parser.parse_args()


# In[3]:


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so please run with --cuda")

print(args)


# In[4]:


def convert_nparr_to_tensor(X_data):
    for i, elem in enumerate(X_data):
        X_data[i] = torch.Tensor(elem.astype(np.float64))
    return X_data

mdata = Music_Sequence(args.data, forcing_max_length=True, stop=None)
X_train = convert_nparr_to_tensor(mdata.get_data(which_set='train'))
X_valid = convert_nparr_to_tensor(mdata.get_data(which_set='valid'))
X_test = convert_nparr_to_tensor(mdata.get_data(which_set='test'))


# In[5]:


# show data 
#temp = X_train[0]
#plt.matshow(temp.transpose(0, 1), cmap=ListedColormap(['w', 'k']))


# In[6]:


# settings
input_size = mdata.max_label
dropout = args.dropout
lr = args.lr


# In[7]:


model = None
if args.model_name == 'MLP':
    model = Basic_MLP(input_size, input_size, 66)
elif args.model_name == 'RNN':
    model = RNN_tanh(input_size, 100, cuda_on = args.cuda)
elif args.model_name == 'LSTM':
    model = RNN_LSTM(input_size, 36, cuda_on = args.cuda)
elif args.model_name == 'GRU':
    model = RNN_GRU(input_size, 46, cuda_on = args.cuda)
    
if args.cuda:
    model.cuda()
criterion = nn.BCELoss(reduction='sum')
optimizer = None
if args.optim == 'Adam' or args.optim == 'RMSprop':
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


# In[8]:


def evaluate(X_data, name='Eval'):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for idx in eval_idx_list:
            data_line = X_data[idx]
            x, y = Variable(data_line[:-1]), Variable(data_line[1:])
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            output = model(x.unsqueeze(0)).squeeze(0)
            loss = criterion(output, y)
            total_loss += loss.item()
            count += output.size(0)
        eval_loss = total_loss / count
        #print(name + " loss: {:.5f}".format(eval_loss))
        return eval_loss


# In[9]:


def train(ep):
    model.train()
    total_loss = 0
    count = 0
    train_idx_list = np.arange(len(X_train), dtype="int32")
    np.random.shuffle(train_idx_list)

    for idx in train_idx_list:
        data_line = X_train[idx]
        x, y = Variable(data_line[:-1]), Variable(data_line[1:])
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        output = model(x.unsqueeze(0)).squeeze(0)
        loss = criterion(output, y)
        # loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
        #                    torch.matmul((1 - y), torch.log(1 - output).float().t()))
        total_loss += loss.item()
        count += output.size(0)

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        loss.backward()
        optimizer.step()
    return total_loss / count


# In[10]:


def save_dict_to_file(dic, filename):
    f = open(filename, 'w')
    f.write(str(dic))
    f.close()


class EarlyStopping():
    # By https://forensics.tistory.com/29
    def __init__(self, patience, verbose=True):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print(f'Training process is stopped early....')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False

# In[11]:


if __name__ == "__main__":
    best_vloss = 1e8
    loss_out = {
        'train': [],
        'valid': [],
        'test': [],
        'step': [],
        'time': []
    }
    model_name = "./results/poly_music_{0}_{1}_{2}_{3}.pt".format(
        args.data, args.model_name, input_size, args.optim)
    file_name = "./results/poly_music_{0}_{1}_{2}_{3}.txt".format(
        args.data, args.model_name, input_size, args.optim)
    cumul_time = 0
    early_stopping = EarlyStopping(20)
    for ep in range(1, args.epochs+1):
        start_time = time.time()
        rloss = train(ep)
        time_duration = time.time()-start_time
        cumul_time += time_duration
        vloss = evaluate(X_valid, name='Validation')
        if vloss < best_vloss:
            with open(model_name, "wb") as f:
                torch.save(model, f)
                # print("Saved model!\n")
            best_vloss = vloss
        if ep % args.log_interval == 0:
            loss_out['step'].append(ep)
            loss_out['time'].append(cumul_time)
            loss_out['valid'].append(vloss)
            loss_out['train'].append(rloss)
            tloss = evaluate(X_test, name='Test')
            loss_out['test'].append(tloss)
            print("Epoch {:2d} \t| lr {:.5f} | rloss {:.5f} | vloss {:.5f} | tloss {:.5f} | time {:.3f}".
                  format(ep, lr, rloss, vloss, tloss, cumul_time))
        if ep > 100 and early_stopping.validate(vloss):
            break
    print('-' * 89)
    model = torch.load(open(model_name, "rb"))
    tloss = evaluate(X_test)
    print('Best performance: ', tloss)
    save_dict_to_file(loss_out, file_name)

#     for label in ['train', 'valid', 'test']:
#         plt.plot(loss_out['step'], loss_out[label], label=label)    
#     #plt.yscale('log')
#     plt.legend()
#     plt.show()

# In[ ]:





# In[ ]:




