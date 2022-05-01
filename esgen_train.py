import numpy as np
from gensim.models.word2vec import Word2Vec
import pickle
import os
from torch.utils.data import Dataset , DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F


def split_data(filename="data.txt" , filename_split="split_data.txt"):
    """
    读取源文件按字用空格切开，存入新文件
    """
    raw_data = open ( filename , "r" , encoding="utf-8" ).read ()
    raw_data_split = " ".join ( raw_data )
    with open ( filename_split , "w" , encoding="utf-8" ) as f:
        f.write ( raw_data_split )


def train_w2vec(filename="data.txt" , filename_split="split_data.txt" , embedding_size=256):
    """
    Word2Vec的训练
    生成训练结果w1
    embedding_size=256:出来之后每一个字对应1x256的向量
    vocab_size（model算出来）:对于源文件有多少个互不相同的字（含标点），2022/4/30用的数据集：2560
    w1（model算出来）：一个[2560x256]的矩阵，代表每一个互不相同的字对应的vector，用index取出
    word_to_index（model算出来）：从字找到index
    index_to_word（model算出来）：从字找到index
    后三者存入w2v.pkl
    返回原始数据(每500个字切)，[w1，word_to_index，index_to_word]
    """
    split_data ()
    raw_data = open ( filename , "r" , encoding="utf-8" ).read ().split ( "\n" )
    raw_data = [ line for line in raw_data if line.strip () ]  # 去掉空行
    raw_data = "。".join ( raw_data )
    raw_data = raw_data.replace ( " " , "" )
    raw_data_length = len ( raw_data ) // 50 - 1
    filled_data = [ ]
    for i in range ( raw_data_length ):
        filled_data.append ( raw_data[ 50 * i: 50 * (i + 1) ] )
    raw_data_split = open ( filename_split , "r" , encoding="utf-8" ).read ().split ( "\n" )
    #    if os.path.exists ( "w2v.pkl" ):
    #        return filled_data , pickle.load ( open ( "w2v.pkl" , "rb" ) )
    model = Word2Vec ( raw_data_split , vector_size=embedding_size , min_count=1 )
    pickle.dump ( (model.syn1neg , model.wv.key_to_index , model.wv.index_to_key) , open ( "w2v.pkl" , "wb" ) )
    return filled_data , (model.syn1neg , model.wv.key_to_index , model.wv.index_to_key)


class DataProcess ( Dataset ):
    """
    封装数据
    """

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
        xs_embedding = self.w1[ xs_index ]
        return xs_embedding , np.array ( ys_index ).astype ( np.int64 )

    def __len__(self):
        """
        获取数据总长度
        """
        return len ( self.data_sheet )


class EsGenRnnModel ( nn.Module ):
    def __init__(self , embedding_size , hidden_size , word_size):
        super ( EsGenRnnModel , self ).__init__ ()
        self.embedding_size = embedding_size  # 每一个字向量的长度
        self.hidden_size = hidden_size  # lstm层的输出尺寸
        self.word_size = word_size  # 词表中不重复的字数，作为最终的输出，取最大值索引作为预测的y的index
        self.num_layers = 2
        self.lstm = nn.LSTM ( input_size=embedding_size , hidden_size=hidden_size , batch_first=True ,
                              num_layers=self.num_layers )
        self.dropout = nn.Dropout ( 0.3 )
        self.flatten = nn.Flatten ( 0 , 1 )
        self.linear = nn.Linear ( hidden_size , word_size )
        self.cross_entropy = nn.CrossEntropyLoss ()

    def forward(self , xs_embedded , h0=None , c0=None):
        device = "cuda" if torch.cuda.is_available () else "cpu"
        xs_embedded = xs_embedded.to ( device )
        if h0 is None or c0 is None:
            # h0和c0的形状：num_layers，batch_size，hidden_size
            h0 = torch.tensor (
                np.zeros ( (self.num_layers , xs_embedded.shape[ 0 ] , self.hidden_size) , np.float32 ) )
            c0 = torch.tensor (
                np.zeros ( (self.num_layers , xs_embedded.shape[ 0 ] , self.hidden_size) , np.float32 ) )
            h0 = h0.to ( device )
            c0 = c0.to ( device )
        # x_embedded 的形状：（batch，length，embedded_size，即8x49x256）
        hidden , (hn , cn) = self.lstm ( xs_embedded , (h0 , c0) )
        hidden = self.dropout ( hidden )
        # hidden 的形状：（batch，length，hidden_size，即8x49x64）
        flatten = self.flatten ( hidden )
        # flatten 的形状：（batch x length，hidden_size，即1568x64）
        predict = self.linear ( flatten )
        # predict 的形状：（1568 x word_size，即1568x2560）
        return predict , (hn , cn)


def generator(pred_length=50):
    result = ""
    word_index = np.random.randint ( 0 , word_size , 1 )[ 0 ]
    result += index_to_word[ word_index ]
    h0 = torch.tensor ( np.zeros ( (2 , 1 , model.hidden_size) , np.float32 ) )
    c0 = torch.tensor ( np.zeros ( (2 , 1 , model.hidden_size) , np.float32 ) )
    h0 = h0.to ( device )
    c0 = c0.to ( device )
    for i in range ( pred_length ):
        word_embedded = w1[ word_index ].reshape ( 1 , 1 , -1 )
        word_embedded = torch.tensor ( word_embedded )
        prediction , (h0 , c0) = model ( word_embedded , h0 , c0 )
        word_index = int ( torch.argmax ( prediction ) )
        word = index_to_word[ word_index ]
        result += word
    result = result.replace("X","北京大学")
    print ( result )


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available () else "cpu"
    filled_data , (w1 , word_to_index , index_to_word) = train_w2vec ()
    dataset = DataProcess ( data_sheet=filled_data , w1=w1 , word_to_index=word_to_index )
    batch_size = 8
    dataloader = DataLoader ( dataset , batch_size=batch_size , shuffle=False )  # 能帮我们按照batch数打包
    word_size , embedding_size = w1.shape

    model = EsGenRnnModel ( embedding_size=embedding_size , word_size=word_size , hidden_size=64 )
    model = model.to ( device )
    lr = 0.01
    epochs = 1000
    optimizer = torch.optim.Adam ( model.parameters () , lr )

    for epoch in range ( epochs ):
        print ( f'epoch:{epoch}' )
        for batch_index , (xs_embedding , ys_index) in enumerate ( dataloader ):
            xs_embedding = xs_embedding.to ( device )
            ys_index = ys_index.to ( device )
            predict , (h0 , c0) = model.forward ( xs_embedding )
            loss = model.cross_entropy ( predict , ys_index.reshape ( -1 ) )
            loss.backward ()
            optimizer.step ()
            optimizer.zero_grad ()
            if batch_index % 50 == 0:
                print ( f'loss:{loss:.3f}' )
                generator ()
    torch.save ( model , 'model.pth' )
    torch.save ( model.state_dict () , 'model_params.pth' )
