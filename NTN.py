import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F
from models.bi_lstm import bilstm


class NeuralTensorNetwork(nn.Module):
    def __init__(self, embedding_size, tensor_dim, dropout, device="cpu"):
        super(NeuralTensorNetwork, self).__init__()
        self.device = device
        self.bilstm = bilstm(input_dim=300, num_lstm_units=128)

        embedding_matrix = pickle.load(open("D:\project\The People's Daily\dataset\processed_data/300_embedding_matrix.dat", 'rb'))
        embedding_matrix = torch.from_numpy(embedding_matrix)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        print('load', '{0}_embedding_matrix.dat'.format(str(embedding_size)))
        self.tensor_dim = tensor_dim

        ##Tensor Weight
        # |T1| = (embedding_size, embedding_size, tensor_dim)
        self.T1 = nn.Parameter(torch.Tensor(embedding_size * embedding_size * tensor_dim))
        self.T1.data.normal_(mean=0.0, std=0.02)

        # |T2| = (embedding_size, embedding_size, tensor_dim)
        self.T2 = nn.Parameter(torch.Tensor(embedding_size * embedding_size * tensor_dim))
        self.T2.data.normal_(mean=0.0, std=0.02)

        # |T3| = (tensor_dim, tensor_dim, tensor_dim)
        self.T3 = nn.Parameter(torch.Tensor(tensor_dim * tensor_dim * tensor_dim))
        self.T3.data.normal_(mean=0.0, std=0.02)

        # |W1| = (embedding_size * 2, tensor_dim)
        self.W1 = nn.Linear(embedding_size * 2, tensor_dim)

        # |W2| = (embedding_size * 2, tensor_dim)
        self.W2 = nn.Linear(embedding_size * 2, tensor_dim)

        # |W3| = (tensor_dim * 2, tensor_dim)
        self.W3 = nn.Linear(tensor_dim * 2, tensor_dim)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)

        self.linear = nn.Linear(2*self.tensor_dim, 10)
        self.title_linear = nn.Linear(256, 100)


    def forward(self, svo, titles):
        # # |svo| = (batch_size, max_length)
        # # |sov_length| = (batch_size, 3)
        #
        # svo = self.emb(svo)
        # # |svo| = (batch_size, max_lenght, embedding_size)
        #
        # ## To merge word embeddings, Get mean value
        # subj, verb, obj = [], [], []
        # for batch_index, svo_batch in enumerate(sov_length):
        #     sub_svo = svo[batch_index]
        #     len_s, len_v, len_o = svo_batch
        #     subj += [torch.mean(sub_svo[:len_s], dim=0, keepdim=True)]
        #     verb += [torch.mean(sub_svo[len_s:len_s+len_v], dim=0, keepdim=True)]
        #     obj += [torch.mean(sub_svo[len_s+len_v:len_s+len_v+len_o], dim=0, keepdim=True)]
        svo = self.embed(svo)    # [batch, max_events, 3, 2, embedding_size]
        svo = svo.view(-1, 3, 2, 300)  # [batch*max_events, 3, 2, embedding_size]
        svo = F.avg_pool2d(svo, (svo.shape[2], 1)).squeeze()  # [batch*max_events, 3, embedding_size]
        subj = svo[:, 0, :]
        verb = svo[:, 1, :]
        obj = svo[:, 2, :]
        # |subj|, |verb|, |obj| = (batch_size, embedding_size)

        R1 = self.tensor_Linear(subj, verb, self.T1, self.W1)
        R1 = self.tanh(R1)
        R1 = self.dropout(R1)
        # |R1| = (batch_size, tensor_dim)

        R2 = self.tensor_Linear(verb, obj, self.T2, self.W2)
        R2 = self.tanh(R2)
        R2 = self.dropout(R2)
        # |R2| = (batch_size, tensor_dim)

        U = self.tensor_Linear(R1, R2, self.T3, self.W3)
        U = self.tanh(U)

        # # 拼接序列事件的语境
        # titles = self.embed(titles)   # [batch, title_len, embedding_size]
        # output, text_hid = self.bilstm(titles)
        # text_hid = torch.cat((text_hid[0, :, :], text_hid[1, :, :]), 1)
        # titles = self.title_linear(text_hid).unsqueeze(1)
        # titles = titles.expand(-1, 70, -1)


        # view_predict = self.linear(U)    # (batch_size*max_events, 6)
        # view_predict = view_predict.view(-1, 70, 10)
        U = U.view(-1, 70, self.tensor_dim)    # (batch_size*max_events, tensor_dim)----> (batch_size, max_events,  tensor_dim)
        # U = torch.cat((U, titles), dim=2)
        # return U, view_predict
        return U


    def tensor_Linear(self, o1, o2, tensor_layer, linear_layer):
        # |o1| = (batch_size, unknown_dim)
        # |o2| = (batch_size, unknown_dim)
        # |tensor_layer| = (unknown_dim * unknown_dim * tensor_dim)
        # |linear_layer| = (unknown_dim * 2, tensor_dim)

        batch_size, unknown_dim = o1.size()

        # 1. Linear Production
        o1_o2 = torch.cat((o1, o2), dim=1)
        # |o1_o2| = (batch_size, unknown_dim * 2)
        linear_product = linear_layer(o1_o2)
        # |linear_product| = (batch_size, tensor_dim)

        # 2. Tensor Production
        tensor_product = o1.mm(tensor_layer.view(unknown_dim, -1))
        # |tensor_product| = (batch_size, unknown_dim * tensor_dim)
        tensor_product = tensor_product.view(batch_size, -1, unknown_dim).bmm(o2.unsqueeze(1).permute(0,2,1).contiguous()).squeeze()
        tensor_product = tensor_product.contiguous()
        # |tensor_product| = (batch_size, tensor_dim)

        # 3. Summation
        result = tensor_product + linear_product
        # |result| = (batch_size, tensor_dim)
        return result