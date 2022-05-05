import torch
from mingpt.char_dataset import CharDataset
from mingpt.utils import sample

if __name__ == '__main__':
    # 配置dataset
    block_size = 128
    text = open ( 'data.txt' , 'r' , encoding="utf-8" ).read ()
    train_dataset = CharDataset ( text , block_size )
    # 读取模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load ( "model.pth" , map_location=device )
    context = input("请输入第一句话：")
    length = input("请输入生成长度：")
    try:
        length = int(length)
    except:
        length = 200
    x = torch.tensor ( [ train_dataset.stoi[ s ] for s in context ] , dtype=torch.long )[ None , ... ].to(device)
    y = sample ( model , x , length , temperature=1.0 , sample=True , top_k=10 )[ 0 ]
    completion = ''.join ( [ train_dataset.itos[ int ( i ) ] for i in y ] )
    print ( completion )
