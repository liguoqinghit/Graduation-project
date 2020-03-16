import numpy_test
import pandas as pd

if __name__ == '__main__':
    '''
    删除community_5的数据
    如 A, b, A 或   (A, b, C and A, a, C)
    有缺陷 一般数据都是,分割 但是有的数据会包含,  如 A, B, A,B
    
    
    还没测试：
        去重方法 df.drop_duplicates(subset=['A', 'B'], keep='first')
        去掉A和B列中重复的行，并保留重复出现的行中第一次出现的行 
        当keep=False时，就是去掉所有的重复行 
        当keep=‘first’时，就是保留第一次出现的重复行 
        当keep=’last’时就是保留最后一次出现的重复行。    
    '''
    df = pd.read_csv(r'community_6.csv', encoding='utf-8')
    data = df.values
    print(len(data))
    filter_data1 = []
    filter_data2 = []
    data2 = []
    for a, b, c in data:
        if a == c:
            continue
        temp1 = [a, c]
        temp2 = [c, a]
        if temp1 not in filter_data1 and temp2 not in filter_data1:
            data2.append([a, b, c])
        filter_data1.append(temp1)
        filter_data1.append(temp2)
    print(len(data2))
    numpy_test.two_dimensional_list_to_file(r'community_6_1.csv', data2)

