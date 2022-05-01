import torch
import pickle
import numpy as np
from esgen_train import *
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available () else "cpu"
    model = torch.load ( 'model.pth' ,map_location=device)
    model.load_state_dict ( torch.load ( 'model_params.pth' ,map_location=device) )
    w1 , word_to_index , index_to_word = pickle.load ( open ( "w2v.pkl" , "rb" ) )

    while True:
        result = ""
        start = input("开头：")
        length = int(input("长度："))
        result += start
        h0 = torch.tensor(np.zeros((2,1,model.hidden_size),np.float32))
        c0 = torch.tensor(np.zeros((2,1,model.hidden_size),np.float32))
        # 预热
        for i in range(len(start)):
            if start[i] in word_to_index.keys():
                word_index = word_to_index[start[i]]
                word_embedded = w1[ word_index ].reshape(1,1,-1)
                word_embedded = torch.tensor(word_embedded)
                prediction , (h0 , c0) = model ( word_embedded , h0 , c0 )
        # 生成
        if start[ len(start) - 1 ] in word_to_index.keys ():
            word_index = word_to_index[start[len(start) - 1]]
        for i in range(length):
            word_embedded = w1[ word_index ].reshape ( 1 , 1 , -1 )
            word_embedded = torch.tensor ( word_embedded )
            prediction , (h0 , c0) = model ( word_embedded , h0 , c0 )
            word_index = int ( torch.argmax ( prediction ) )
            word = index_to_word[ word_index ]
            result += word
        result = result.replace ( "X" , start )
        print ( result )
