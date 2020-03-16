import pandas as pd
import networkx as nx
import random
import numpy_test

if __name__ == '__main__':
    """
    向文件中添加随即边 
    具体内容：
        向community_6.csv添加 人物 - 随机 - 食物 的新属性
        目前数据包含 人物 1 - 266 食物 267 - 399 
    """
    df = pd.read_csv('community_6_index.csv', encoding='utf-8')
    df.drop_duplicates(subset=[df.columns[0]], keep='first', inplace=True)
    entity = df[df.columns[0]].values.tolist()
    df = pd.read_csv('community_6_index.csv', encoding='utf-8')
    df.drop_duplicates(subset=[df.columns[1]], keep='first', inplace=True)
    attribute = df[df.columns[1]].values.tolist()
    G = nx.from_pandas_edgelist(df, '实体', '值', '属性')
    print(len(entity))
    print(len(attribute))
    len(entity)
    people = 0
    food = len(entity) - 1
    attribute_len = len(attribute)
    # 添加30个边  10%
    new = []
    for i in range(30):
        new.append([entity[people], attribute_len, entity[food]])
        attribute_len += 1
        people += 1
        if i % 2 == 0:
            food -= 2
    print(new)
    numpy_test.two_dimensional_list_to_file('community_6_2_index.csv', new)
