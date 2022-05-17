from playwright.sync_api import sync_playwright
import requests
from bs4 import BeautifulSoup

order=0
theme=['年轻人买房','物价上涨','疫情','经济危机','农村贫困','全面建成小康社会']
num_theme=len(theme)



with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://sojo.im/slscq/")
    #s=page.wait_for_selector('id=numInputEl')
    #box = s.bounding_box()
    #page.mouse.move(box.x+100,box.y+8)
    #page.mouse.click()
    with open('data1.txt','w',encoding='utf-8') as f:
        for i in range(1000):
            page.fill('#themeInputEl',theme[order%num_theme])
            order+=1
            #page.pause()
            page.click('#生成')
            str=page.content()
            soup = BeautifulSoup(str)
            
            div_tags=soup.find_all('div',id='post')
            for tag in div_tags:
                p_tags=tag.find_all('p')
                for p_tag in p_tags:
                    f.write(p_tag.text)
                    f.write('\n')
                    
            #page.pause()

