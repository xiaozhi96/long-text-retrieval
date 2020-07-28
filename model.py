import sys
import torch
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pickle
import math
import numpy as np
from models.bi_lstm import BiLSTM, bilstm
from DeepCCA.DeepCCAModels import DeepCCA
from DeepCCA.linear_cca import linear_cca
from models.img_cnn import CNNModel
from models.C_NTN import contexual_NeuralTensorNetwork
# from models.NTN import NeuralTensorNetwork
from models.center_NTN import center_NeuralTensorNetwork
from models.ssl_NTN import ssl_NeuralTensorNetwork
from models.predicate_tensor_model import PTM
from models.cnn_enbeding import cnn_emb
from models.CRFT import NeuralTensorNetwork
from models.attention import Attention
from torch.autograd import Variable


class event_embedding_model(nn.Module):
    def __init__(self, args):
        super(event_embedding_model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=args.embedding_size, out_channels=args.feature_size, kernel_size=args.kernel_size)
        # self.conv2 = nn.Conv1d(in_channels=args.embedding_size, out_channels=args.feature_size, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=args.feature_size, out_channels=args.event_dimension, kernel_size=2)

    def forward(self, text, text_length):   # input(batch, event_num， 3， embed_dim）
        # [batch, max_event_num, 3, 2, embed_dim])-->[batch*max_event_num， 3，2 embed_dim]
        # F.avg_pool2d输入是四维
        text = text.view(-1, 3, 2, 300)    # 这个300是embedding_dim
        # [batch*max_event_num， 3，2 embed_dim]--->[batch*max_event_num， 3， embed_dim]
        text = F.avg_pool2d(text, (text.shape[2], 1)).squeeze()
        # text = text.reshape(128, 100, 3, 100)  Con1D的输入只能是3维
        text = text.permute(0, 2, 1)    # 一维卷积是在最后一个维度上卷积
        text = self.conv1(text)    # [batch*event_num, feature_size, 2]   A,P,O两两卷积
        text = self.conv3(text)    # [batch*event_num, event_dimension, 1]
        text = text.squeeze().reshape(-1, 100, 100)     # [batch, event_num, event_dimension]

        return text          # [batch, event_num, event_dimension]


class ImgNN(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=512, output_dim=200):
        super(ImgNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        # out = self.denseL1(x)
        return out


class TextNN(nn.Module):
    """Network to learn text representations"""

    def __init__(self, input_dim=256, output_dim=200):
        super(TextNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        # out = self.denseL1(x)
        return out


# 返回的是resnet提取的image_feature和BiLSTM提取的text_feature 的joint_embedding
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.bilstm = BiLSTM(args)
        self.resnet = CNNModel()
        self.event_embedding = event_embedding_model(args)

        self.NTN = contexual_NeuralTensorNetwork(embedding_size=300, tensor_dim=200, dropout=0.1)
        self.NTN.load_state_dict(torch.load("D:\project\The People's Daily/event_embedding/300_final_model_param.pt"))
        self.NTN.eval()

        # # center_NTN
        # self.center_NTN = center_NeuralTensorNetwork(embedding_size=300, tensor_dim=100, dropout=0.1)
        # self.center_NTN.load_state_dict(torch.load("D:\project\The People's Daily/event_embedding/300_super_NTN_param.pt"))
        # self.center_NTN.eval()

        # self.ssl_NTN = ssl_NeuralTensorNetwork(embedding_size=300, tensor_dim=100, dropout=0.1)
        # self.ssl_NTN.load_state_dict(torch.load("D:\project\The People's Daily/event_embedding/500_ssl_NTN_param.pt"))
        # # self.ssl_NTN.eval()

        # # CRFT
        # self.NTN = NeuralTensorNetwork(embedding_size=300, tensor_dim=100, dropout=0.1)
        # self.NTN.load_state_dict(torch.load("D:\project\The People's Daily/event_embedding/final_RFT_param.pt"))
        # self.NTN.eval()   # 要不要继续更新预训练的模型呢

        # self.PTM = PTM(embedding_size=300, tensor_dim=100, dropout=0.1)
        # self.PTM.load_state_dict(torch.load("D:\project\The People's Daily/event_embedding/final_PTM_param.pt"))
        # self.PTM.eval()  # 要不要继续更新预训练的模型呢

        V = args.vocab_size
        D = args.word_embed_dim

        # word embedding
        # if args.use_embedding:
        #     embedding_matrix_file_name = 'D:/project/multi-sarcasm/dataset/' \
        #                                  '{0}_embedding_matrix.dat'.format(str(args.word_embed_dim))
        #     embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
        #     self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        #     print('load', '{0}_{1}_embedding_matrix.dat'.format(str(args.word_embed_dim), args.dataset))

        if args.use_embedding:
            # 使用北师大数据的预训练词向量
            embedding_matrix = pickle.load(open("D:\project\The People's Daily\dataset\processed_data/300_embedding_matrix.dat", 'rb'))
            # embedding_matrix = pickle.load(open('D:\project\crossmodal_retrieval_in_news\dataset\processed_data/100_embedding_matrix.dat', 'rb'))
            embedding_matrix = torch.from_numpy(embedding_matrix)
            self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
            print('load', '{0}_{1}_embedding_matrix.dat'.format(str(args.embedding_size), args.dataset))
        else:
            self.embed = nn.Embedding(V, D, padding_idx=0)


        self.img_net = ImgNN(args.img_input_dim, args.img_output_dim)
        self.text_net = TextNN(args.text_input_dim, args.text_output_dim)
        self.linearLayer = nn.Linear(args.img_output_dim, args.minus_one_dim)
        self.linearLayer2 = nn.Linear(args.minus_one_dim, args.output_dim)

        self.event_predict = nn.Linear(200, 10)
        self.event_feature_linear = nn.Linear(300, 200)   # 平铺是21000

        self.conv1 = nn.Conv1d(in_channels=200, out_channels=256,
                               kernel_size=5)

    def forward(self, img, text, text_length, title):    # 输入的text是每个新闻的事件集合
        img_features = self.resnet(img)    # [batch_size, 512, 1, 1]
        # text = self.embed(text)    # [batch, max_length, 3, 2, embed_dim])
        # text_emb = self.event_embedding(text, text_length)    # [batch, max_event_num, event_dimension]将每个事件嵌入为事件向量
        text_emb, _ = self.NTN(text, title)   # # [batch, max_events, 2*tensor_dim]

        # # BILSTM
        # output, text_hid = self.bilstm(text_emb.cuda(), text_length.cuda())  # output:[seq_len, batch_size, 2*hid_dim]  text_hid:[2, 10, 50]
        # text_hid = torch.cat((text_hid[0, :, :], text_hid[1, :, :]),1)  # [batch_size, 2*hid_dim],把第一个维度取消也就是把前向后向隐状态拼接起来
        # output = output.permute(1, 0, 2)  # [batch_size, seq_len, 2*hid_dim]

        # # CNN
        # text_emb = text_emb.permute(0, 2, 1)   # [batch, event_dim, max-events]
        # text_hid = self.conv1(text_emb)   # [batch, feature_dim, 70-5+1]
        # text_hid = text_hid.permute(0, 2, 1)
        # text_hid = F.max_pool2d(text_hid, (text_hid.shape[1], 1)).squeeze()

        # 平铺事件feature
        # text_feature = text_emb.view(text_emb.shape[0], -1)   # [batch, max_events, tensor_dim]---->[batch, max_events*tensor_dim]
        text_feature = F.avg_pool2d(text_emb, (text_emb.shape[1], 1)).squeeze()
        view2_feature = self.event_feature_linear(text_feature)

        view1_feature = self.img_net(img_features.squeeze())
        # view2_feature = self.text_net(text_hid)

        # 公共空间，view1_feature和view2_feature是用来和标签监督生成语义区分的表示
        view1_feature = self.linearLayer(view1_feature)
        view2_feature = self.linearLayer(view2_feature)


        # label空间，view1_predict和view2_predict是用来和标签监督生成语义区分的表示
        view1_predict = self.linearLayer2(view1_feature)
        view2_predict = self.linearLayer2(view2_feature)

        return view1_feature, view2_feature, view1_predict, view2_predict

class DSCMR(nn.Module):
    def __init__(self, args):
        super(DSCMR, self).__init__()
        self.bilstm = BiLSTM(args)
        self.resnet = CNNModel()

        V = args.vocab_size
        D = args.word_embed_dim
        if args.use_embedding:
            # 使用北师大数据的预训练词向量
            embedding_matrix = pickle.load(open("D:\project\The People's Daily\dataset\processed_data/300_embedding_matrix.dat", 'rb'))
            # embedding_matrix = pickle.load(open('D:\project\crossmodal_retrieval_in_news\dataset\processed_data/100_embedding_matrix.dat', 'rb'))
            embedding_matrix = torch.from_numpy(embedding_matrix)
            self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
            print('load', '{0}_{1}_embedding_matrix.dat'.format(str(args.embedding_size), args.dataset))
        else:
            self.embed = nn.Embedding(V, D, padding_idx=0)


        self.img_net = ImgNN(args.img_input_dim, args.img_output_dim)
        self.text_net = TextNN(args.text_input_dim, args.text_output_dim)
        self.linearLayer = nn.Linear(args.img_output_dim, args.minus_one_dim)
        self.linearLayer2 = nn.Linear(args.minus_one_dim, args.output_dim)

    def forward(self, img, text, text_length):
        img_features = self.resnet(img)    # [batch_size, 512, 1, 1]
        text = self.embed(text)    # [batch, max_length, embed_dim])

        # # 使用事件的wordvector检索 [batch, max_event_num, 3, 2, embed_dim])-->[batch*max_event_num， 3，2 embed_dim]
        # # F.avg_pool2d输入是四维
        # # text = text.view(-1, 3, 2, 300)    # 这个300是embedding_dim
        # text = text[:, :, :, 0, :].squeeze()
        # # text = F.avg_pool2d(text, (text.shape[2], 1)).squeeze()   # [batch*max_event_num， 3，embed_dim]
        # # text = text.view(-1, 70, 3, 300)  # [batch, max_event_num， 3，embed_dim]
        # text = text.view(text.shape[0], -1, 300)

        output, text_hid = self.bilstm(text.cuda(),text_length.cuda())  # output:[seq_len, batch_size, 2*hid_dim]  text_hid:[2, 10, 50]
        text_hid = torch.cat((text_hid[0, :, :], text_hid[1, :, :]),1)  # [batch_size, 2*hid_dim],把第一个维度取消也就是把前向后向隐状态拼接起来
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, 2*hid_dim]

        view1_feature = self.img_net(img_features.squeeze())
        view2_feature = self.text_net(text_hid)

        # 公共空间，view1_feature和view2_feature是用来和标签监督生成语义区分的表示
        view1_feature = self.linearLayer(view1_feature)
        view2_feature = self.linearLayer(view2_feature)

        # label空间，view1_predict和view2_predict是用来和标签监督生成语义区分的表示
        view1_predict = self.linearLayer2(view1_feature)
        view2_predict = self.linearLayer2(view2_feature)

        return view1_feature, view2_feature, view1_predict, view2_predict


class DCAA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1,
                    input_size2, outdim_size, use_all_singular_values, device):
        super(DCAA, self).__init__()
        embedding_matrix = pickle.load(open("D:\project\The People's Daily\dataset\processed_data/300_embedding_matrix.dat", 'rb'))
        embedding_matrix = torch.from_numpy(embedding_matrix)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)

        self.bilstm = bilstm(input_dim=300, num_lstm_units=128)
        self.resnet = CNNModel()

        self.dcaa = DeepCCA(layer_sizes1, layer_sizes2, input_size1,
                    input_size2, outdim_size, use_all_singular_values, device).double()
        self.cca = linear_cca()

    def forward(self, image, text):
        img_feature = self.resnet(image).squeeze()
        text = self.embed(text)
        output, text_hid = self.bilstm(text)
        text_feature = torch.cat((text_hid[0, :, :], text_hid[1, :, :]), 1)

        out1, out2 = self.dcaa(img_feature.double(), text_feature.double())
        # out1, out2 = self.cca.fit(img_feature.cpu().detach().numpy(), text_feature.cpu().detach().numpy(), outdim_size=10)
        # out1 = torch.from_numpy(out1)
        # out2 = torch.from_numpy(out2)
        return out1, out2
