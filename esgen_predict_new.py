import torch
import pickle
import numpy as np
import torch.nn as nn


class Params:
    def __init__(self):
        self.filename = "data.txt"  # 源文件
        self.w2v_filename = "w2v.pkl"  # w2v文件
        self.device = "cuda" if torch.cuda.is_available () else "cpu"
        self.embedded_size = 256  # 每一个字从w2v算出来是[1 , embedded_size]向量
        self.word_size = 0  # w1算出来
        self.cut_size = 50  # 训练文段的长度
        self.batch_size = 32  # batch大小
        self.hidden_size = 128  # 中间层大小
        self.epochs = 1000  # 学习次数
        self.lr = 0.1  # 学习率


class GruModel ( nn.Module ):
    def __init__(self):
        super ( GruModel , self ).__init__ ()
        self.embedded_size = params.embedded_size
        self.hidden_size = params.hidden_size
        self.word_size = params.word_size
        self.gru = nn.GRU ( input_size=self.embedded_size , hidden_size=self.hidden_size , batch_first=True )
        self.flatten = nn.Flatten ( 0 , 1 )
        self.linear = nn.Linear ( self.hidden_size , self.word_size )
        self.cross_entropy = nn.CrossEntropyLoss ()

    def forward(self , x_embedded , h0):
        x_embedded = x_embedded.to ( params.device )
        h0 = h0.to ( params.device )
        # x_embedded的形状：[batch , cut_size , embedded_size]
        hidden , hn = self.gru ( x_embedded , h0 )
        # hidden的形状：[batch , cut_size , hidden_size]
        flatten = self.flatten ( hidden )
        # flatten的形状：[batch x cut_size , hidden_size]
        predict = self.linear ( flatten )
        # pre的形状：[batch x cut_size , word_size]
        return predict , hn

    def init_h0(self , batch_size):
        return torch.zeros ( (1 , batch_size , self.hidden_size) , device=params.device )


def generator(length , starts):
    h = model.init_h0 ( 1 )
    h = h.to ( params.device )
    result = ""
    result += starts
    if starts[ len ( starts ) - 1 ] in word_to_index.keys ():
        word_index = word_to_index[ starts[ len ( starts ) - 1 ] ]
    else:
        word_index = word_to_index[ "。" ]
    for i in range ( length ):
        word_embedded = w1[ word_index ].reshape ( 1 , 1 , -1 )
        word_embedded = torch.tensor ( word_embedded )
        prediction , h = model ( word_embedded , h )
        word_index = int ( torch.argmax ( prediction ) )
        word = index_to_word[ word_index ]
        result += word
    result = result.replace ( "X" , starts )
    print ( result )


if __name__ == '__main__':
    params = Params ()
    model = torch.load ( 'model.pth' , map_location=params.device )
    model.load_state_dict ( torch.load ( 'model_params.pth' , map_location=params.device ) )
    w1 , word_to_index , index_to_word = pickle.load ( open ( "w2v.pkl" , "rb" ) )
    while True:
        starts = input ( "请输入主题：" )
        length = input ( "请输入长度：" )
        try:
            length = int ( length )
        except:
            length = 200
        generator ( length , starts )
