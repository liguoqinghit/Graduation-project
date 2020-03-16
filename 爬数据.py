# coding = utf-8
import pandas as pd
import re
from urllib.request import urlopen  # 获取用以请求打开网页的库
from bs4 import BeautifulSoup  # 获取解析网页的库
from urllib import parse
import numpy_test


def get_data(url, entity):
    """
    根据搜索内容 返回一个 实体-属性-值 的 list
    """
    html = urlopen(url)
    bs = BeautifulSoup(html, 'html.parser')
    nameList = bs.findAll("dt", {"class": "basicInfo-item name"})
    valueList = bs.findAll("dd", {"class": "basicInfo-item value"})
    names = [i.get_text().replace('\n', '').replace('\xa0', '') for i in nameList]
    values = [i.get_text().replace('\n', '').replace('\xa0', '') for i in valueList]
    return [[entity, names[i], values[i]] for i in range(len(nameList))]


# 最好不是孤立的点
if __name__ == '__main__':
    '''
    思路：
        以现有实体作为基础，使用百度搜索
        想要的数据为 实体 + 标签中的属性 + 标签中的值
        
        原本想以标签中的值所带的超链接进行数据链接 但是没实现
        以标签中的值为索引(目前以长度小于5为索引) 继续搜索
    '''
    df = pd.read_csv('community_3(删除实体相同关系不同的).csv', encoding='utf-8')
    df.drop_duplicates(subset=[df.columns[0]], keep='first', inplace=True)
    entity_list = df[df.columns[0]].values.tolist()
    # print(entity_list)
    # print(len(entity_list))
    '''
    编码以及解码 (百度百科后面的东西，为了直接获得url)
    >>> parse.quote('三鲜豆皮')
    '%E4%B8%89%E9%B2%9C%E8%B1%86%E7%9A%AE'
    >>> parse.unquote('%E4%B8%89%E9%B2%9C%E8%B1%86%E7%9A%AE')
    '三鲜豆皮'
    '''
    # html = urlopen("https://baike.baidu.com/item/%E4%B8%89%E9%B2%9C%E8%B1%86%E7%9A%AE")  # 获取html结构与内容
    # bs0bj = BeautifulSoup(html, 'html.parser')
    # print(bs0bj.prettify())  # 整个源码
    #
    # nameList = bs0bj.findAll("dt", {"class": "basicInfo-item name"})
    # for i in nameList:
    #     print(i.get_text())
    # valueList = bs0bj.findAll("dd", {"class": "basicInfo-item value"})
    # for i in valueList:
    #     print(i.get_text())
    # print(bs0bj.a.contents)  # ['百度首页']
    # print(bs0bj.a.attrs)  # {'href': 'http://www.baidu.com/'}
    # print(bs0bj.a)  # <a href="http://www.baidu.com/">百度首页</a>
    # print(bs0bj.a.string)  # 百度首页
    # a = bs0bj.findAll("a")
    # for i in a:
    #     # print(re.findall(r'"([^"]*)"', str(i)))
    #     link = i.get('href')
    #     print(link)
    # print(bs0bj.select('class > a'))  # []
    # print(get_data('https://baike.baidu.com/item/%E4%B8%89%E9%B2%9C%E8%B1%86%E7%9A%AE'))
    ignore = []
    # data = []
    flag = 0
    for i in entity_list:
        data = []
        ignore.append(i)
        str = parse.quote(i)
        temp = get_data('https://baike.baidu.com/item/%s' % str, i)
        values = []
        for eav in temp:
            data.append(eav)
            if eav[0] != eav[2] and len(eav[2]) < 5 and eav[2] not in ignore:
                values.append(eav[2])
        print(values)
        for j in values:
            ignore.append(j)
            str = parse.quote(j)
            temp = get_data('https://baike.baidu.com/item/%s' % str, j)
            for eav in temp:
                data.append(eav)
        flag += 1
        numpy_test.two_dimensional_list_to_file('test.txt', data)
        # if flag == 5:
        #     break
    list = ['贡丸', '主要食材', '猪肉']