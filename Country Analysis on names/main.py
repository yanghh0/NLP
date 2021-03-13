#!/usr/bin/env python3
# -*- coding:utf8 -*-

import os
import csv
import gzip
import torch
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence

BATCH_SIZE = 256
HIDDEN_SIZE = 100
N_LAYER = 2
N_EPOCHS = 15
N_CHARS = 128
USE_GPU = True

def String2List(name):
    arr = [ord(c) for c in name]
    return arr

def CreateTensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tensor = tensor.to(device)
    return tensor

def TransformData(names, countries):
    """
    一个名字和对应的国家就是一个样本，名字的每个字母就是每个时间步对网络的输入
    """
    name_sequences = [String2List(name) for name in names]
    seq_lengths = torch.LongTensor([len(x) for x in name_sequences])
    countries = countries.long()

    # 每个样本的时间步长弄成相同的，不足的补零
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return CreateTensor(seq_tensor), seq_lengths, CreateTensor(countries)

class NameDataset(data.Dataset):
    def __init__(self, is_train_set=True):
        file_name = 'names_train.csv.gz' if is_train_set else 'names_test.csv.gz'
        with gzip.open(os.path.join('dataset', file_name), 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.names = [row[0] for row in rows]
        self.countries = [row[1] for row in rows]
        self.len = len(self.names)

        self.country_list = list(sorted(set(self.countries)))
        self.country_dict = self.GetCountryDict()
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]

    def __len__(self):
        return self.len

    def GetCountryDict(self):
        """
        将国家名称变成数字类别
        """
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict

    def GetCountryNum(self):
        return self.country_num

    def Idx2CountryName(self, index):
        return self.country_list[index]

class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        self.gru = torch.nn.GRU(input_size=hidden_size, 
                                hidden_size=hidden_size, 
                                num_layers=n_layers, 
                                bidirectional=bidirectional)
        self.fc = torch.nn.Linear(in_features=hidden_size * self.n_directions, out_features=output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return CreateTensor(hidden)

    def forward(self, inputs, seq_lengths):
        # input: (bitch_size, seq_len)  output: (seq_len, bitch_size)
        inputs = inputs.t()

        # output: (n_layers * n_directions, batch_size, hidden_size)
        hidden = self._init_hidden(inputs.size(1))

        # input: (seq_len, bitch_size)  output: (seq_len, bitch_size, hidden_size)
        embedding = self.embedding(inputs)

        # pack them up
        gru_input = pack_padded_sequence(embedding, seq_lengths)

        # output: (seq_len, batch_size, n_directions * hidden_size)
        # hidden: (n_layers * n_directions, batch_size, hidden_size)
        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        
        return fc_output

def TrainModel():
    total_loss = 0
    batch_num = 0
    for batch_idx, data in enumerate(trainloader, 1):
        names, countries = data
        inputs, seq_lengths, target = TransformData(names, countries)
        optimizer.zero_grad()

        output = model(inputs, seq_lengths)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_num += 1
    print('loss: %.3f' % (total_loss / batch_num))

def TestModel():
    correct = 0
    total = len(testset)
    with torch.no_grad():
        for i, data in enumerate(testloader, 1):
            names, countries = data
            inputs, seq_lengths, target = TransformData(names, countries)
            output = model(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        print('accuracy on test set: %d %% ' % (100 * correct / total))
    return correct / total

if __name__ == '__main__':
    trainset = NameDataset(is_train_set=True)
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testset = NameDataset(is_train_set=False)
    testloader = data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    N_COUNTRY = trainset.GetCountryNum()
    model = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    if USE_GPU:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        # Train cycle
        TrainModel()
        acc = TestModel()
        acc_list.append(acc)

    epoch = np.arange(1, len(acc_list) + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()