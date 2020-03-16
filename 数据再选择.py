import json
import numpy_test

if __name__ == '__main__':
    '''
    从名义师兄的数据 获取相关信息
    以及统计从gephi导出社团数据，统计各个社团的数量 (社团序号, 社团内节点数量)
    '''
    data = []
    # set = set()
    # d = 0
    # with open(r'C:\Users\庆\Documents\WeChat Files\lgq0335\FileStorage\File\2020-03\clean_edge.json', 'r', encoding='utf-8') as f:
    #     for line in f:
    #         dic = json.loads(line)
    #         if dic['data']['r'] == 'co-occurrence':
    #             d += 1
    # print(d)
    # 一共 7129
    # hasObject 1586
    # hasActor 2191
    # hasRecipient 684
    # cooperate 134
    # BelongTo 108
    # co-occurrence 168

    # with open(r'C:\Users\庆\Documents\WeChat Files\lgq0335\FileStorage\File\2020-03\clean_edge.json', 'r', encoding='utf-8') as f:
    #     for line in f:
    #         dic = json.loads(line)
    #         temp = [dic['data']['source'], dic['data']['r'], dic['data']['target']]
    #         data.append(temp)
    # numpy_test.two_dimensional_list_to_file('新数据.csv', data)

    # with open(r'C:\Users\庆\Documents\WeChat Files\lgq0335\FileStorage\File\2020-03\clean_node.json', 'r', encoding='utf-8') as f:
    #     for line in f:
    #         dic = json.loads(line)
    #         if '0' <= dic['data']['id'][0] <= '9':
    #             data.append(dic)
    #     with open('导出数据.json', 'a', encoding='utf-8') as w:
    #     for dic in data:
    #         json.dump(dic, w)
    #         w.write('\n')
    # print(data)
    dic = {}
    with open('modularity(20w).csv', 'r', encoding='utf-8') as r:
        i = 0
        for line in r:
            i += 1
            if i == 1:
                continue
            line = line.strip().split(',')
            dic[int(line[3])] = dic.get(int(line[3]), 0) + 1
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    print(dic)

#     [(2, 675), (0, 660), (3, 199), (54, 175), (9, 168), (10, 167), (5, 161), (77, 155), (13, 132), (51, 117), (18, 109), (41, 74), (14, 68), (1, 51), (11, 46), (66, 40), (15, 26), (35, 14), (20, 10), (67, 9), (37, 8), (8, 7), (27, 6), (43, 6), (58, 6), (79, 6), (105, 6), (4, 5), (21, 5), (30, 5), (97, 5), (22, 4), (23, 4), (29, 4), (32, 4), (39, 4), (47, 4), (48, 4), (61, 4), (64, 4), (68, 4), (69, 4), (72, 4), (76, 4), (81, 4), (90, 4), (91, 4), (6, 3), (7, 3), (12, 3), (16, 3), (17, 3), (19, 3), (24, 3), (25, 3), (26, 3), (28, 3), (31, 3), (33, 3), (34, 3), (36, 3), (38, 3), (40, 3), (42, 3), (44, 3), (45, 3), (46, 3), (49, 3), (50, 3), (52, 3), (53, 3), (55, 3), (56, 3), (57, 3), (59, 3), (60, 3), (62, 3), (63, 3), (65, 3), (94, 3), (95, 3), (104, 3), (70, 2), (71, 2), (73, 2), (74, 2), (75, 2), (78, 2), (80, 2), (82, 2), (83, 2), (84, 2), (85, 2), (86, 2), (87, 2), (88, 2), (89, 2), (92, 2), (93, 2), (96, 2), (98, 2), (99, 2), (100, 2), (101, 2), (102, 2), (103, 2)]
'''
node

id  唯一标识
label   显示内容
tags    标签序列(只对Event)
timestamp   Event的时间戳
type    节点类型(Event, Stakeholder, Service, Domain)
attr    额外的属性(只对Stakeholder, Service, Domain)

{"data": 
{
"id": "9", 
"label": "动感单车健身品牌“HappyCycle”完成数千万Pre-A轮融资", 
"tags": [[0, 8, "Object"], [9, 19, "Actor"], [20, 22, "Action"], [22, 25, "Attribute"], [25, 33, "Object"], [34, 44, "Single"]], 
"timestamp": "2019-08-12", 
"type": "Event"
}, 
"classes": "msem-evnet"}


edge

id  唯一标识
source  源节点
target  目标节点
r   关系名
timestamp   s
generated_from  s

{"data": 
{
"id": "9#HappyCycle#2019-08-12#hasActor", 
"source": "9", 
"target": "HappyCycle", 
"timestamp": "2019-08-12", 
"type": "hasX", 
"r": "hasActor"
}, 
"classes": "msem-strutral msem-hasx"}
'''