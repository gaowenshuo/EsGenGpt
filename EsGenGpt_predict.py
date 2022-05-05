import torch
from mingpt.char_dataset import CharDataset
from mingpt.utils import sample

if __name__ == '__main__':
    # 配置dataset
    block_size = 128
    text = open ( 'data.txt' , 'r' , encoding="utf-8" ).read ()
    train_dataset = CharDataset ( text , block_size )
    # 读取模型
    model = torch.load ( "model.pth" , map_location="cpu" )
    context = "北京大学是世界一流大学"
    x = torch.tensor ( [ train_dataset.stoi[ s ] for s in context ] , dtype=torch.long )[ None , ... ]
    y = sample ( model , x , 2000 , temperature=1.0 , sample=True , top_k=10 )[ 0 ]
    completion = ''.join ( [ train_dataset.itos[ int ( i ) ] for i in y ] )
    print ( completion )
