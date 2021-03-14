#!/usr/bin/env python3
# -*- coding:utf8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
N_LAYER = 1

def ArgMax(vec):
    """
    vec 形如 [[2, 7, 9, 3]], 返回 vec 在 dim 为 1 上的最大值索引
    torch.max返回值: 
        values=tensor([9]),
        indices=tensor([2]))
    """
    _, idx = torch.max(vec, 1)
    return idx.item()

def LogSumExp(vec):
    """
    vec 形如 [[2, 7, 9, 3]]
    指数和累积到一定程度后，会超过计算机浮点值的最大值，变成inf，这样取log后也是inf。
    为了避免这种情况，用最大值 max 去提指数和的公因子，这样就不会使某项变得过大而无法计算
    SUM = log(exp(s1) + exp(s2) + ... + exp(s100))
        = log{exp(max) * [exp(s1-max) + exp(s2-max) + ... + exp(s100-max)]}
        = max + log[exp(s1-max) + exp(s2-max) + ... + exp(s100-max)]
    """
    max_score = vec[0, ArgMax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # 这里返回的是维数为0的tensor
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def PrepareSequence(seq, word_to_idx):
    idxs = [word_to_idx[w] for w in seq]
    return torch.LongTensor(idxs)

class BiLstmCrf(nn.Module):
    # vocab_size: 数据集单词字典大小
    def __init__(self, vocab_size, tag_to_idx, embedding_dim, hidden_dim, n_layers, bidirectional=True):
        super(BiLstmCrf, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.tagset_size = len(tag_to_idx)
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.word_embeds = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim // 2,
                            num_layers=n_layers,
                            bidirectional=bidirectional)

        # Maps the output of the LSTM into tag space.
        self.fc = nn.Linear(in_features=hidden_dim, out_features=self.tagset_size)

        # 转移矩阵的参数初始化，transitions[i,j]代表的是从第j个tag转移到第i个tag的转移分数.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # 初始化所有其他tag转移到START_TAG的分数非常小，即不可能由其他tag转移到START_TAG
        # 初始化STOP_TAG转移到所有其他tag的分数非常小，即不可能由STOP_TAG转移到其他tag
        self.transitions.data[tag_to_idx[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_idx[STOP_TAG]] = -10000

    def _init_hidden(self):
        # batch_size = 1
        # 初始化 h0, c0
        # c0-cn就是开关，决定每个神经元的隐藏状态值是否会影响的下一时刻的神经元的处理，形状应该和h0-hn一致。
        return (torch.randn(self.n_layers * self.n_directions, 1, self.hidden_dim // 2), 
                torch.randn(self.n_layers * self.n_directions, 1, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):
        self.hidden = self._init_hidden()

        # input: (seq_len)  embeds: (seq_len, batch_size, embedding_dim)
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)

        # input: (seq_len, batch_size, embedding_dim)  lstm_out: (seq_len, batch_size, num_directions * hidden_size)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        # 因为batch_size = 1，所以直接去掉，返回: (seq_len, num_directions * hidden_size)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)

        # lstm_feats: (seq_len, tagset_size)
        lstm_feats = self.fc(lstm_out)

        return lstm_feats

    def _forward_alg(self, feats):
        # 初始化alphas'，shape: (1, tagset_size)
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # 初始化 START 节点的score，START_TAG 位置取 0 其他位置取 -10000
        init_alphas[0][self.tag_to_idx[START_TAG]] = 0.
        previous = init_alphas

        # shape of feats: (seq_len, tagset_size)
        for feat in feats:
            # The forward tensors at this timestep
            alphas_t = []
            for next_tag in range(self.tagset_size):
                # 取出当前节点的发射分数，本来维数为0，拓展为：(1, tagset_size)
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # 取出转移到该节点的所有转移分数，(target_size,) 拓展为 (1, tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                # 之前的alphas' + 转移分数 + 发射分数
                next_tag_var = previous + trans_score + emit_score
                # shape of alphas_t: [[], [], []]
                alphas_t.append(LogSumExp(next_tag_var).view(1))
            previous = torch.cat(alphas_t).view(1, -1)
        # 会进行广播 
        terminal_var = previous + self.transitions[self.tag_to_idx[STOP_TAG]]
        # 这就是 Z'(x)
        scores = LogSumExp(terminal_var)
        return scores

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_idx[START_TAG]] = 0

        previous = init_vvars

        for feat in feats:
            # holds the backpointers for this step
            # 保存当前时间步的回溯指针
            bptrs_t = []

            # holds the viterbi variables for this step
            # 保存当前时间步的维特比变量
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = previous + self.transitions[next_tag]
                best_tag_id = ArgMax(next_tag_var)
                bptrs_t.append(best_tag_id)
                # [[],[],[],...]，最外面是个list，每个元素是形状为1的tensor
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            previous = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        # 考虑转移到STOP_TAG的转移分数
        terminal_var = previous + self.transitions[self.tag_to_idx[STOP_TAG]]
        best_tag_id = ArgMax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        # 通过回溯指针解码出最优路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.tag_to_idx[START_TAG]
        best_path.reverse()

        return path_score, best_path

    def _score_sentence(self, feats, tags):
        """
        计算给定targets标签序列的分数，即一条路径的分数
        """
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_idx[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_idx[STOP_TAG], tags[-1]]
        return score

    def _neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        # 损失函数值
        return forward_score - gold_score

    def forward(self, sentence):
        """
        神奇，这个函数居然没被调用
        """
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

def SimpleTest():
    """
    自制一个简单的数据集进行测试
    """
    # Make up some training data
    training_data = [("the wall street journal reported today that apple corporation made money".split(),
                      "B I I I O O O B I O O".split()),
                     ("georgia tech is a university in georgia".split(),
                      "B I O O O O B".split())]

    word_to_idx = {}
    tag_to_idx = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

    model = BiLstmCrf(vocab_size=len(word_to_idx), 
                      tag_to_idx=tag_to_idx, 
                      embedding_dim=EMBEDDING_DIM, 
                      hidden_dim=HIDDEN_DIM, 
                      n_layers=N_LAYER)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(10):
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.We need to clear them out before each instance.
            model.zero_grad()
            # Step 2. Get our inputs ready for the network, that is, turn them into word indices.
            # 把单词和标签转成数字
            inputs = PrepareSequence(sentence, word_to_idx)
            targets = torch.LongTensor([tag_to_idx[t] for t in tags])
    
            loss = model._neg_log_likelihood(inputs, targets)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        precheck_sent = PrepareSequence(training_data[0][0], word_to_idx)
        print(model(precheck_sent))

if __name__ == '__main__':
    SimpleTest()
