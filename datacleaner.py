import re
import string
from tqdm import tqdm
rubbish = [#'国考', '省考', '考公', '公务员', '来源网络', '侵删', '侵权', '笔试', '考生', '刷新网站', 
           # '资讯', '微信', '公众号', '复习', '学习交流', '背景', '参考',
           #'科目', '交流群', '推荐阅读', '详情', 'http', '注意事项', '申论', '大纲', '试卷', 
           # '全真', '模拟', '本题', '考试', '资料', '准考证', '点击查看', '查看', '收藏', '半月'
           #,'答案','寻找','文章','内容','1200','字数','文章','考','观点','要求','逻辑','论点']
           '为题','条理','结构流畅']
punctuation = '''！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏.'''
with open('crawl_demo_clean1.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    parag = []
    with open('crawl_demo_clean2.txt', 'w', encoding='utf-8') as f1:
        for line in lines:
            line = re.sub('\s+', '', line)
            line = re.sub('[【】[]]', '', line)
            '''
            line.replace('（','')
            line.replace('）','')
            line.replace('【','')
            line.replace('】','')
            line.replace('《','')
            line.replace('》','')
            '''
            line.strip()
            parag = re.split('[,，。!！?？;；:：]', line)
            parag = [p for p in parag if all(r not in p for r in rubbish)]
            f1.write(','.join(parag))
            f1.write('\n')
            parag.clear()
