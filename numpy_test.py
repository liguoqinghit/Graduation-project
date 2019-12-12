import numpy as np
from scipy import spatial
import math
import pandas as pd
import networkx as nx
import source_code


# def vector_cos(vertora, vertorb):
#     """计算两向量的余弦值
#
#     :param vertora: 向量a, list
#     :param vertorb: 向量b, list
#     :return: 余弦值 cos(a, b)
#     """
    # vertora = np.asarray(vertora)
    # vertorb = np.asarray(vertorb)
    # if (vertora == 0).all() or (vertorb == 0).all():
    #     return 0
    # a_norm = np.linalg.norm(vertora)
    # b_norm = np.linalg.norm(vertorb)
    # cos_a_b = vertora.dot(vertorb) / (a_norm * b_norm)
    # cos_a_b = cos_a_b if cos_a_b <= 1 else 1.0
    # cos_a_b = cos_a_b if cos_a_b >= -1 else -1.0
    # return cos_a_b


def euclidean_distance(vertora, vertorb):
    # 标准化欧氏距离 有点问题
    vertora = np.asarray(vertora)
    vertorb = np.asarray(vertorb)
    # 判断向量是否为0
    # if not (vertora == 0).all() or not (vertorb == 0).all():
    #     return 0
    dis = 0
    for i in range(len(vertora)):
        if not np.var([vertora[i], vertorb[i]]):
            continue
        dis += np.sqrt(np.sum(np.square(vertora[i] - vertorb[i]) / np.var([vertora[i], vertorb[i]])))
    return dis


def standardized_euclidean_distance(vertora, vertorb, v):
    # 标准化欧氏距离
    ver = np.array([np.array(vertora), np.array(vertorb)])
    a = spatial.distance.pdist(ver, 'seuclidean', v)
    if math.isnan(a):
        return 0
    else:
        return a


def two_dimensional_list_to_file(filename, communities):
    with open(filename, 'w', encoding='utf-8') as f:
        for community in communities:
            for node in community:
                f.write(str(node) + ' ')
            f.write('\n')


# def test(com):
#     com[0] = frozenset(com[0] | com[1])
#     del com[1]
#     print('函数里面', com)


# communities = dict((i, frozenset([i])) for i in range(5))
# test(communities)
# print('函数外面', communities)
# b = np.array([1, 1, 1])
# test(b)
# print('函数外面', b)
# G = nx.Graph()
# G.add_edges_from(([0, 1], [0, 2], [2, 3]))
# C = source_code.greedy_modularity_communities(G)
# print(C)
# a = np.array([1, 1])
# b = np.array([2, 2])
# print(euclidean_distance(a, b)/2)
# df = pd.read_csv('community_2.csv', encoding='utf-8')
# IG = nx.Graph()
# G = nx.from_pandas_edgelist(df, '实体', '值', '属性')
# c = source_code.greedy_modularity_communities(G)
# two_dimensional_list_to_file('原社团划分.txt', c)
# N = G.number_of_nodes()
# print(G.nodes())
# label_for_node = dict((i, v) for i, v in enumerate(G.nodes()))
# print(label_for_node)
# communities = dict((i, frozenset([i])) for i in range(N))
# print(communities)
# partition = [[label_for_node[x] for x in c] for c in communities.values()]
# q_cnm = modularity(G, partition)
# print(q_cnm)
# c = source_code.greedy_modularity_communities(G)
# print(c)
# a = np.zeros((3, 2), dtype=np.int)
# a[1][0] = 1
# a[1][1] = 0
# a[2][0] = 5
# a[2][0] = 1
# print(vector_cos(a[1], a[2]))
# x = [1, 1, 0]
# y = [2, 1, 0]
# x = np.linspace(0.0028, 0.0028, 100)
# y = np.linspace(0, 0, 100)
# print(standardized_euclidean_distance(x, y, [x, y])**2)
"""
# 向量模长
a1_norm = np.linalg.norm(a[1])
a2_norm = np.linalg.norm(a[2])
# 点积
a1_dot_a2 = a[1].dot(a[2])
# 余弦 得到的是弧度
cos_theta = np.arccos(a1_dot_a2 / (a1_norm * a2_norm))
# 弧度转度数
print(np.rad2deg(cos_theta))
print(cos_theta/3.14)
"""
