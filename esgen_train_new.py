import numpy as np
from gensim.models.word2vec import Word2Vec
import pickle
import os
from torch.utils.data import Dataset , DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def w2v():
    """
    读取数据，进行w2v
    :return: w1，word_to_index，index_to_word
    w1:[word_size , embedded_size]
    一共有word_size个互不相同的字，拼在一起就是w1
    word_to_index:dict of word_size
    index_to_word:list of word_size
    """
    with open ( params.filename , "r" , encoding="utf-8" ) as f:
        raw_data = f.read ()
    raw_data_split = " ".join ( raw_data ).split ( "\n" )  # 按空格切开，便于w2v使用
    w2v_model = Word2Vec ( raw_data_split , vector_size=params.embedded_size , min_count=1 )
    w1 , word_to_index , index_to_word = w2v_model.syn1neg , w2v_model.wv.key_to_index , w2v_model.wv.index_to_key
    pickle.dump ( (w1 , word_to_index , index_to_word) , open ( params.w2v_filename , "wb" ) )
    params.word_size = w1.shape[ 0 ]
    return w1 , word_to_index , index_to_word


def fill_data():
    """
    读取并整理数据
    按照cut_size切开
    :return: filled_data:list of whole_length // cut_size - 1
    其中每一个元素为cut_size长度的str
    """
    with open ( params.filename , "r" , encoding="utf-8" ) as f:
        raw_data = f.read ()
    filled_data = [ ]
    raw_data = raw_data.replace ( "\n" , "" )
    raw_data = raw_data.replace ( " " , "" )
    whole_length = len ( raw_data )
    for i in range ( whole_length // params.cut_size - 1 ):
        filled_data.append ( raw_data[ i * params.cut_size: (i + 1) * params.cut_size ] )
    return filled_data


class DataProcess ( Dataset ):
    def __init__(self , data_sheet , w1 , word_to_index):
        """
        加载所有数据，存储初始化变量
        """
        self.data_sheet = data_sheet
        self.w1 = w1
        self.word_to_index = word_to_index

    def __getitem__(self , index):
        """
        获取一条数据进行处理
        index:拿出来第几段
        """
        a_paragraph_words = self.data_sheet[ index ]
        a_paragraph_index = [ self.word_to_index[ word ] for word in a_paragraph_words ]
        # 由于是从前一个字预测下一个字
        xs_index = a_paragraph_index[ :-1 ]  # 从第一个字到倒数第二个字
        ys_index = a_paragraph_index[ 1: ]  # 从第二个字到倒数第一个字
        xs_embedded = self.w1[ xs_index ]
        return xs_embedded , np.array ( ys_index ).astype ( np.int64 )

    def __len__(self):
        """
        获取数据总长度
        """
        return len ( self.data_sheet )


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
    w1 , word_to_index , index_to_word = w2v ()
    filled_data = fill_data ()
    dataset = DataProcess ( data_sheet=filled_data , w1=w1 , word_to_index=word_to_index )
    data_loader = DataLoader ( dataset , batch_size=params.batch_size , shuffle=True , drop_last=True )
    model = GruModel ()
    model = model.to ( params.device )
    optimizer = torch.optim.SGD ( model.parameters () , lr=params.lr , weight_decay=0 )

    for epoch in range ( params.epochs ):
        print ( f'epoch:{epoch + 1}' )
        h = None
        for x_embedded , y_index in data_loader:
            x_embedded = x_embedded.to ( params.device )
            y_index = y_index.to ( params.device )
            if h is None:
                h = model.init_h0 ( params.batch_size )
            else:
                h.detach_ ()
            pred , h = model ( x_embedded , h )
            loss = model.cross_entropy ( pred , y_index.reshape ( -1 ) )
            loss.backward ()
            optimizer.step ()
            optimizer.zero_grad ()
        print ( f'loss:{loss:.3f}' )
        generator ( 50 , "北京大学" )
        torch.save ( model , 'model.pth' )
        torch.save ( model.state_dict () , 'model_params.pth' )
