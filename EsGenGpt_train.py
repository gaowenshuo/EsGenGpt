import torch
from mingpt.char_dataset import CharDataset
from mingpt.model import GPT , GPTConfig
from mingpt.trainer import Trainer , TrainerConfig
import logging

if __name__ == '__main__':
    # 配置log
    logging.basicConfig (
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s" ,
        datefmt="%m/%d/%Y %H:%M:%S" ,
        level=logging.INFO ,
    )
    # 配置dataset
    block_size = 128
    text = open ( 'data.txt' , 'r' , encoding="utf-8" ).read ()
    train_dataset = CharDataset ( text , block_size )
    # 配置model
    mconf = GPTConfig ( train_dataset.vocab_size , train_dataset.block_size ,
                        n_layer=8 , n_head=8 , n_embd=512 )
    model = GPT ( mconf )
    # 配置trainer
    tconf = TrainerConfig ( max_epochs=2 , batch_size=512 , learning_rate=6e-4 ,
                            lr_decay=True , warmup_tokens=512 * 20 ,
                            final_tokens=2 * len ( train_dataset ) * block_size ,
                            num_workers=4 )
    trainer = Trainer ( model , train_dataset , None , tconf )
    trainer.train ()
    # 保存模型
    torch.save ( model , "model.pth" )
