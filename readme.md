# EsGenLstm:基于Word2Vec和GRU的申论文章生成器
# EsGenGpt:基于注意力机制和GPT2的申论文章生成器

## 2022.5.17更新
#### 新增文件
- `data1.txt` 从申论生成器爬取的1000篇文章,尚未训练
- `data(cleaned).txt` 清洗过的data.txt
- `begin.py` 新的程序入口,GUI的入口,已与模型预测文件对接
- `EsGenGpt_predict_withGUI.py` 与begin.py即图形界面对接而改动的predict文件
- `ui_test.py`qt for python 的ui文件
- `dataproducer.py` 用于爬取申论生成器的脚本
- `datacleaner.py` 用于清洗data所用的脚本

> **使用库:**
> `pyside6`: qt官方推出的python图形库  
> `playwright`: 用于producer爬取申论生成器文章  
> `beautifulsoup`: 用于解析爬取的html提取文章  

## 使用指南：
#### 需要文件：
~~- data.txt，~~
~~- EsGenGpt_predict.py，~~
> ~~按顺序运行两个py即可~~  
*2022.5.17更新:因加入GUI,原来的指南不再适用*
- data(cleaned).txt
- EsGenGpt_train.py
- begin.py
> 先训练EsGenGpt_train.py,然后运行begin.py即可

## 生成示例：
**开头**：解决农村人口贫困问题，必须精准扶贫


解决农村人口贫困问题，必须精准扶贫的核心之一。乡村振兴只有通过完善质量、补助，才能最终提高农村的产业水平。
解决农村人口贫困问题，必须精准扶贫是重要解决对策。要从产业、文化、人才端持续发力，才能让村民们的腰包越来越鼓，笑容越来越多，方能真正下好乡村振兴这篇大棋。
精准扶贫要注意因地制宜。各个地区的贫困状况也是不同的，因此各个地区的扶贫对策也是不同的，因此我们要从各地的本地情况出发，从根本上找到本地贫困落后的原因，这样我们才能够有针对性地制定扶贫的对策。因为没有做到从，本地出发因地制宜，从而导致扶贫失败的实例比比皆是。其他地区先进的扶贫经验，是否适用于本地，需要做到以本地的情况为出发点，明确本地的特有优势和现有水平，需要与本地的产业相对接，这样才能做精准扶贫。精准扶贫要注意多渠道、多元化。扶贫开发已经进入攻坚拔寨冲刺期，任务十分艰巨。扶贫开发贵在精准，重在精准，成败在于精准。单一的渠道往往不能够解决一地的贫困问题，各地情况也千差万别，因此探索多渠道、多元化的精准扶贫新路径是必要的选择。在这一过程中，要充分发挥社会组织能动作用，积极搭建社会参与平台，培育多元社会扶贫主体，为全面建成小康社会贡献力量。
