import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from collections import Counter
import numpy as np
import random
import pickle
import scipy
from sklearn.metrics.pairwise import cosine_similarity


class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word2idx, word_freqs, C, K):
        ''' text: a list of words, all text from the training dataset
            word2idx: the dictionary from word to index
            word_freqs: the frequency of each word
        '''
        super(WordEmbeddingDataset, self).__init__()  # 通过父类初始化模型，然后重写两个方法
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text]  # 把单词数字化表示。如果不在词典中，也表示为unk
        self.text_encoded = torch.LongTensor(self.text_encoded)  # nn.Embedding需要传入LongTensor类型
        self.word2idx = word2idx
        self.word_freqs = torch.Tensor(word_freqs)
        self.C = C
        self.K =K
        
        
    def __len__(self):
        return len(self.text_encoded) # 返回所有单词的总数，即item的总数
    
    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word
        '''
        center_words = self.text_encoded[idx] # 取得中心词
        pos_indices = list(range(idx - self.C, idx)) + list(range(idx + 1, idx + self.C + 1)) # 先取得中心左右各C个词的索引
        pos_indices = [i % len(self.text_encoded) for i in pos_indices] # 为了避免索引越界，所以进行取余处理
        pos_words = self.text_encoded[pos_indices] # tensor(list)
        
        neg_words = torch.multinomial(self.word_freqs, self.K * pos_words.shape[0], True)
        # torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标
        # 取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大
        # 每采样一个正确的单词(positive word)，就采样K个错误的单词(negative word)，pos_words.shape[0]是正确单词数量
        
        # while 循环是为了保证 neg_words中不能包含背景词
        while len(set(pos_indices) & set(neg_words)) > 0:
            neg_words = torch.multinomial(self.word_freqs, self.K * pos_words.shape[0], True)
        return center_words, pos_words, neg_words

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
         
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)  # 中心词权重矩阵
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)  # 周围词权重矩阵
        
    def forward(self, input_labels, pos_labels, neg_labels):
        ''' 
            input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]
            
            return: loss, [batch_size]
        '''
        input_embedding = self.in_embed(input_labels) # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)# [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels) # [batch_size, (window * 2 * K), embed_size]
        
        # squeeze是挤压的意思，所以squeeze方法是删除一个维度，反之，unsqueeze方法是增加一个维度
        
        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1]，在最后一维上增加一个维度
        # bmm方法是两个三维张量相乘，两个tensor的维度是，（b * m * n）, (b * n * k) 得到（b * m * k），相当于用矩阵乘法的形式代替了网络中神经元数量的变化
        # 矩阵的相乘相当于向量的点积，代表两个向量之间的相似度
        pos_dot = torch.bmm(pos_embedding, input_embedding)  # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2)  # [batch_size, (window * 2)]
        neg_dot = torch.bmm(neg_embedding, -input_embedding)  # [batch_size, (window * 2 * K), 1]，这里之所以用减法是因为下面loss = log_pos + log_neg，log_neg越小越好
        neg_dot = neg_dot.squeeze(2)  # batch_size, (window * 2 * K)]
        log_pos = F.logsigmoid(pos_dot).sum(1)  # .sum()结果只为一个数，.sum(1)结果是一维的张量
        log_neg = F.logsigmoid(neg_dot).sum(1)
        loss = log_pos + log_neg
        return -loss
    
    def input_embedding(self):
        return self.in_embed.weight.detach().numpy()


class node2vec(object):
    def __init__(self, Data, Train_Data, save_path, node):
        self.data = Data
        self.train_data = Train_Data
        self.save_path = save_path
        self.node = node

    def get_vec(self, C, K, epoch, MAX_VOCAB_SIZE, EMBEDDING_SIZE, batch_size, lr):
        '''
        C: background word
        K: noise of negative sampling
        epoch: Iterations
        MAX_VOCAB_SIZE: max number of words in vocabulary
        EMBEDDING_SIZE: shape of embedding vector
        batch_size
        lr : learning rate
        '''
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        POI_set = set(self.data[self.node])

        # start skip-gram
        text = list(self.train_data[self.node])
        #print("len(text) = ",len(text))
        vocab_dict = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
        vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values()))
        word2idx = {word:i for i, word in enumerate(vocab_dict.keys())}
        idx2word = {i:word for i, word in enumerate(vocab_dict.keys())}
        word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)
        word_freqs = word_counts / np.sum(word_counts)
        word_freqs = word_freqs ** (3./4.)

        dataset = WordEmbeddingDataset(text, word2idx, word_freqs, C, K)
        dataloader = tud.DataLoader(dataset, batch_size, shuffle=True)

        model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        print("- -"*10)
        print("-"*10+"start training"+"-"*10)
        for e in range(30):
            for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
                input_labels = input_labels.long()
                pos_labels = pos_labels.long()
                neg_labels = neg_labels.long()
                optimizer.zero_grad()
                loss = model(input_labels, pos_labels, neg_labels).mean()
                loss.backward()
                optimizer.step()
                if i % 300 == 0:
                    print('epoch', e, 'iteration', i, loss.item())
        embedding_weights = model.input_embedding()
        torch.save(model.state_dict(), self.save_path+self.node+"embedding-{}.th".format(EMBEDDING_SIZE))
        # get dict of word2idx and idx2word
        word2idx.popitem()
        emb_poi = set(word2idx.keys())
        dif_set = POI_set.difference(emb_poi)
        idx = len(word2idx)
        j = idx
        for i in dif_set:
            word2idx[i] = j
            j = j+1
        idx2word = {j:i for i,j in word2idx.items()}
        # 存储idx2word
        idx2word_file = open(self.save_path+self.node+'idx2node.pickle', 'wb')
        pickle.dump(idx2word, idx2word_file)
        idx2word_file.close()
        # 存储word2idx
        word2idx_file = open(self.save_path+self.node+'node2idx.pickle', 'wb')
        pickle.dump(word2idx, word2idx_file)
        word2idx_file.close()

        # get embedding for every POI
        other_emb = nn.Embedding(len(dif_set), EMBEDDING_SIZE)
        other_emb = other_emb.weight
        other_emb = other_emb.detach().numpy()
        embedding_weights = embedding_weights[:-1]
        #print("embedding_weights.shape = ",embedding_weights.shape)
        #print("other_emb.shape = ",other_emb.shape)
        poi_embedding = np.concatenate((embedding_weights, other_emb), axis = 0)
        np.save(self.save_path+self.node+'embedding.npy',poi_embedding)

        return word2idx, idx2word, poi_embedding

