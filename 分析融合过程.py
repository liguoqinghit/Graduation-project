# coding=utf-8
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import modularity

import experiment_community_detection6
import numpy_test

if __name__ == '__main__':
    # 分析社团连接过程
    """
    分析融合过程：
        newman:
        贡丸 科学 面包[面包]        五大夫松 历史 中国喀斯特 荔波大小七孔
        
        关系社团：
        荔波大小七孔
        贡丸 科学 面包[面包] 五大夫松 历史 中国喀斯特
        len 124 - 123
        
        1.发现问题 
        发现融合过程中一个社团(贡丸 科学 面包[面包] 五大夫松 历史)中加入了  中国喀斯特  这个实体
        应该不止这个社团(因为都是数字 并没有全部查看)
        2. 分析问题
        (1) 先找到原社团划分结果中这些实体中所在的社团
        0, 2, 102, 139, 141, 181, 150
        贡丸：漫画 玉兰片 油炸食品 酸辣粉[重庆小吃名称] 面包[面包] 椰果 小菜 炒鱿鱼[菜品名称] 鲜香 明太子[中日韩料理] 洪山菜苔 动物内脏 川菜系 休闲食品 猪肉 科学 拌三丝 糖醋排骨[中国名菜家常菜] 天妇罗 滇菜 马铃薯 红薯粉丝 菜品 过桥米线 悠久 可口 酸辣黄瓜 金华馒头 太白粉 拌菠菜松 云南菜 马铃薯淀粉 葡萄牙 贡丸 精盐 大众 日本料理 东坡肉 tempura 酸甜 红菜薹 贡丸汤 金瓜粉蒸排骨 食品 麻酱甜椒
        科学：和贡丸是一个社团
        面包[面包]：和贡丸是一个社团
        五大夫松：峨眉山[中国佛教名山，世界文化与自然双遗产] 五岳[中国五岳] 1960年 恒山 地中海气候 国家级文物保护单位 三潭印月[西湖十景之一] 太白山 亚热带季风性气候 灵峰 苍山洱海 草海湖 千佛山[山东济南千佛山] 剪刀峰 百脉泉 0854 博斯腾湖[中国最大的内陆淡水吞吐湖] 孔雀河 西南方言 历史 五大夫松 大明山[浙西大明山] 雅鲁藏布江[藏南大河] 死海[以色列，巴勒斯坦，约旦交界的内陆盐湖] 千亩田 中国喀斯特 地理 乐清市 千岛湖[新安江水库] 小瀛洲 12 内流湖 华山 嵩山 书籍 雁荡山 地点 泰山 地形地貌 贵阳市 旅游 1985年 村庄 普吉岛[泰国岛屿] 半天 衡山 约旦河 亚热带湿润季风气候 大龙湫 芙蓉峰 七星岩[广东肇庆七星岩] 景区 铜仁凤凰机场 黔灵公园 秦岭山脉 梵净山[贵州省铜仁市梵净山] 9米 温带季风气候 自然 0577 居庸关 0856 荔波大小七孔 国家AAAA级旅游景区 贝加尔湖[地理湖泊] 羊卓雍措 南亚
        历史：和五大夫松是一个社团
        中国喀斯特：和五大夫松是一个社团
        (2) 找到 中国喀斯特 加入该社团的原因
        贡丸,标签,科学
        面包[面包],标签,食品
        面包[面包],标签,菜品
        五大夫松,所属国家,中国
        五大夫松,标签,旅游
        五大夫松,标签,地理
        五大夫松,标签,地点
        五大夫松,标签,历史
        荔波大小七孔,电话区号,0854
        荔波大小七孔,地貌,中国喀斯特
        荔波大小七孔,标签,旅游
        荔波大小七孔,标签,地理
        荔波大小七孔,标签,地点
        
        可能加入该社团的原因：有连接 以及 有相同的边(标签) 
        中国喀斯特没有这两个特征，与它连接的只有荔波大小七孔，但是荔波大小七孔并没有加入到该社团
        按原社团划分的结果，该社团可以被分为两个社团：(1) 贡丸 科学 面包[面包] (2) 五大夫松 历史 中国喀斯特
        在原社团划分中，荔波大小七孔 加入了(2)所在的原社团
        但是在关系社团划分中，荔波大小七孔 179 没加入该社团 而 中国喀斯特 却加入了 why？
        
        320 256 193 227 257 98 49 179 19 52 317
        喀纳斯机场 华山[五岳之西岳华山] 吴中蕃 释永信 中国文学 提拉米苏[意大利甜点] 酸牛奶 荔波大小七孔 面窝 冷藏 喀纳斯湖
        355 - 124 = 231 
    """
    """
    # 找到一个社团
    df = pd.read_csv('gephi_关系社团划分2_5_点.csv', encoding='utf-8')
    fusion_order = [80, 256, 341, 58, 258, 14, 198, 139, 98, 138, 296, 217, 53, 175, 121, 253, 211, 306, 285, 119, 317,
                    230, 239, 6, 151, 99, 168, 88, 250, 190, 354, 302, 10, 334, 29, 0, 93, 12, 46, 162, 351, 312, 116,
                    331, 337, 82, 278, 266, 314, 262, 273, 16, 282, 63, 108, 22, 142, 157, 201, 304, 13, 86, 322, 286,
                    347, 326, 65, 244, 329, 268, 236, 111, 293, 248, 100, 60, 301, 205, 177, 31, 255, 195, 49, 133, 41,
                    129, 208, 270, 113, 114, 68, 104, 187, 123, 73, 246, 276, 271, 125, 11, 78, 20, 281, 229, 235, 292,
                    257, 202, 160, 176, 188, 283, 122, 127, 352, 112, 297, 254, 324, 275, 311, 251, 204, 319, 288, 75,
                    294, 267, 310, 303, 159, 74, 83, 227, 238, 280, 344, 131, 315, 259, 50, 185, 170, 199, 166, 330,
                    299, 77, 161, 169, 66, 234, 150, 300, 196, 321, 141, 336, 180, 305, 228, 224, 209, 323, 120, 221,
                    193, 118, 247, 284, 342, 320, 226, 298, 134, 36, 124, 345, 213, 109, 203, 72, 158, 325, 132, 350,
                    287, 232, 348, 7, 154, 26, 222, 240, 333, 51, 252, 308, 340, 178, 87, 274, 272, 289, 264, 206, 69,
                    167, 215, 96, 290, 70, 313, 27, 291, 343, 97, 32, 318, 149, 279, 214, 207, 295, 242, 327, 339, 309,
                    194, 243, 153, 181, 200, 225, 84, 249, 186, 40, 338, 173, 107, 220, 62, 95, 269, 103, 52, 265, 335,
                    328, 218, 212, 155, 147, 263, 61, 35, 353, 163, 130, 156, 102, 231, 148, 241, 105, 261, 28, 25, 79,
                    210, 219, 191, 197, 146, 233, 110, 171, 172, 179, 316, 143, 260, 346, 91, 182, 245, 106, 237, 47,
                    81, 174, 55, 71, 223, 332, 349, 39, 117, 307, 21, 101, 44, 164, 90, 56, 165, 37, 216, 92, 152, 67,
                    145, 115, 184, 89, 126, 18, 94, 48, 85, 45, 5, 8, 189, 33, 4, 76, 192, 128, 54, 64, 277, 144, 38, 2,
                    24, 34, 30, 183, 43, 42, 57, 19, 23, 135, 17, 15, 9]
    print(df.loc[0, 'Id'])  # 0
    print(df.iloc[80, 1])  # 语言
    # 贡丸 科学 面包[面包] 五大夫松 历史 中国喀斯特 城市
    community_1 = [0, 2, 102, 139, 141, 181, 150]
    # 记录是某个点是在什么时候被划分的，即社团长度为多少时被划分的
    fusion_time = []
    count = 355
    for i in fusion_order:
        count -= 1
        if i in community_1:
            fusion_time.append([i, count])
    print(fusion_time)
    for i in community_1:
        print([i, df.iloc[i, 1]], end=' ')
    print('\n')

    node_ignore = [198, 14, 138, 139, 217, 175, 331, 211, 306, 351, 312, 253, 116, 125, 142, 271, 270, 108, 41, 88, 65, 329, 201, 268, 255, 296, 12, 341, 354, 93, 6, 250, 302, 239, 119, 187, 151, 230, 317, 285, 168, 337, 314, 262, 304, 182, 162, 266, 278, 322, 16, 293, 49, 246, 347, 177, 326, 98, 80, 258, 256, 286, 0, 334, 248, 205, 301, 195, 133, 68, 111, 236, 63, 113, 86, 157, 60, 100, 208, 114, 104, 123, 26, 4, 229, 273, 73, 173, 10, 179, 11, 276, 235, 244, 190, 33, 324, 24, 31, 13, 29, 20, 129, 38, 53, 241, 309, 124, 94, 43, 260, 210, 263, 192, 223, 237, 109, 299, 214, 118, 219, 131, 252, 152, 164, 352, 224, 143, 158, 343, 176, 247, 212, 292, 222, 320, 180, 275, 209, 170, 333, 127, 251, 166, 315, 290, 311, 117, 319, 348, 338, 199, 308, 264, 163, 85, 284, 184, 243, 267, 307, 185, 37, 297, 225, 188, 303, 156, 226, 89, 145, 77, 353, 220, 238, 339, 313, 344, 215, 350, 289]
    # len从124变为123融合了 181 中国喀斯特
    # 需要字典来记录 当前社团状态，二维数组
    communities = dict()
    with open('community124.csv', 'r', encoding='utf-8') as read:
        index = 0
        for line in read:
            line = line.split(' ')
            community = [int(i) for i in line if i != '\n']
            communities[index] = frozenset(community)
            index += 1
        print(communities)
    # 53: frozenset({150, 139, 141, 102})
    # 85: frozenset({181})
    # 计算所有Q值
    pair = []
    for i in range(len(communities)):
        for j in range(i):
            if i == j:
                continue
            pair.append([i, j])

    # 套用ex……6里面的程序 计算所有的Q
    df = pd.read_csv('community_3_index.csv', encoding='utf-8')
    G = nx.from_pandas_edgelist(df, '实体', '值', '属性')
    all_nodes = list(G.nodes)
    all_edges = list(G.edges)
    df.drop_duplicates(subset=[df.columns[1]], keep='first', inplace=True)
    columns_list = df[df.columns[1]].values.tolist()
    communities = dict((i, frozenset([i])) for i in range(len(all_nodes)))
    all_degree = list(G.degree())

    # 巨人节点 度大于20 暂定
    node_ignore = []
    for key, value in all_degree:
        if value > 20:
            node_ignore.append(key)

    # 巨人边 从numpy_test.py读取出来的
    # edge_ignore = [1]
    edge_ignore = []

    # C 重新构建为 剩下的社团 但是可能影响结果。
    matrix_c = np.zeros((len(all_nodes), len(columns_list)), dtype=int)
    i = 9
    j = 9
    with open('%sC%s.txt' % (i, j), 'w', encoding='utf-8') as f:
        for i in matrix_c.tolist():
            f.writelines(str(i) + '\n')
    """
    # 以上代码(假期之前的)
    # df = pd.read_csv('gephi_关系社团划分2_5_点.csv', encoding='utf-8')
    # l = [320, 256, 193, 227, 257, 98, 49, 179, 19, 52, 317]
    # for i in l:
    #     print(df.iloc[i, 1], end=" ")
    # print("\n")
    # 计算最终社团的QR、QM和Q 以及去除喀斯特的时候
    # 读取当前社团状态
    # communities = []
    # with open('test_partition2_3.csv', 'r', encoding='utf-8') as f:
    #     for line in f:
    #         line = line.split()
    #         communities.append(line)
    # communities = [[int(i) for i in community] for community in communities]
    # 根据社团状态可以直接计算QM
    # QR由C、RC、D和RD组成
    # 数据初始化
    df = pd.read_csv('community_6_2_index.csv', encoding='utf-8')
    print(len(df.values))
    G = nx.from_pandas_edgelist(df, '实体', '值', '属性')
    all_nodes = list(G.nodes)
    all_edges = list(G.edges)
    df.drop_duplicates(subset=[df.columns[1]], keep='first', inplace=True)
    columns_list = df[df.columns[1]].values.tolist()
    print(len(all_edges))
    print(len(all_nodes))  # 327
    print(len(columns_list))  # 86
    print('----------')
    for i in df.values:
        temp1 = (i[0], i[2])
        temp2 = (i[2], i[0])
        if temp1 not in all_edges and temp2 not in all_edges:
            print(i)
    print('----------')
    '''
    [228  37 272]
    [236  47 268]
    [241  41 252]
    [254  52 298]
    [256  53 313]
    '''
    '''
    # C
    matrix_c = np.zeros((len(all_nodes), len(columns_list)), dtype=int)
    community_length = 0
    for community in communities:
        # 添加 a - b : a在社团内 b不在
        for node1 in community:
            for node2 in all_nodes:
                if node1 == node2 or node2 in community:
                    continue
                if G.has_edge(node1, node2):
                    matrix_c[community_length][int(G[node1][node2]['属性'])] += 1
        # 添加 a - b : a在社团内 b也在
        for i in range(len(community)):
            for j in range(i):
                if i == j:
                    continue
                node1 = community[i]
                node2 = community[j]
                if G.has_edge(node1, node2):
                    matrix_c[community_length][int(G[node1][node2]['属性'])] += 1
        community_length += 1
    # RC
    matrix_rc = np.zeros((len(all_nodes), len(columns_list)))
    for i in range(len(communities)):
        rc = np.sum(matrix_c[i]) / len(columns_list)
        matrix_rc[i] = np.linspace(rc, rc, len(columns_list))

    # D RD
    matrix_d = np.zeros((len(all_nodes), len(all_nodes)))
    matrix_rd = np.zeros((len(all_nodes), len(all_nodes)))
    for i in range(len(all_nodes)):
        for j in range(i):
            if i == j:
                continue
            matrix_d[i][j] = numpy_test.standardized_euclidean_distance(matrix_c[i], matrix_c[j], matrix_c)
            matrix_rd[i][j] = numpy_test.standardized_euclidean_distance(matrix_rc[i], matrix_rc[j], matrix_rc)
    print(np.sum(matrix_d))
    # 正规化D 和 RD
    d_max = np.max(matrix_d)
    d_min = np.min(matrix_d)
    nd = [[(matrix_d[i][j] - d_min) / (d_max - d_min) for j in range(len(matrix_d[i]))] for i in range(len(matrix_d))]
    rd_max = np.max(matrix_rd)
    rd_min = np.min(matrix_rd)
    nrd = [[(matrix_rd[i][j] - rd_min) / (rd_max - rd_min) for j in range(len(matrix_rd[i]))] for i in range(len(matrix_rd))]
    # QR
    qr2 = 2 * (np.sum(nd) - np.sum(nrd)) / len(matrix_d) / (len(matrix_d) + 1)
    # QM
    qm2 = modularity(G, communities)
    # Q
    p1 = 0.2
    p2 = 0.8
    q = p2 * qm2 - p1 * qr2
    print(qr2)
    print(qm2)
    print(q)
    '''
'''
长度为124
移除181 喀斯特
QR : 0.030970139064856836
QM : 0.19072873864618933
Q  : 0.1463889631039801

长度为123
Q = 0.8 * QM - 0.2 * QR
QR : 0.030692487738576982
QM : 0.19070170577421702
Q  : 0.14642286707165825

Q变大
D[][] 
社团数量的影响 太大了？

C 0
RC 

D
2 2 2
0 0 0

9

'''