import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.community import modularity


def vector_cos(vertora, vertorb):
    """计算两向量的余弦值

    :param vertora: 向量a, list
    :param vertorb: 向量b, list
    :return: 余弦值 cos(a, b)
    """
    vertora = np.asarray(vertora)
    vertorb = np.asarray(vertorb)
    if (vertora == 0).all() or (vertorb == 0).all():
        return 0
    a_norm = np.linalg.norm(vertora)
    b_norm = np.linalg.norm(vertorb)
    cos_a_b = vertora.dot(vertorb) / (a_norm * b_norm)
    cos_a_b = cos_a_b if cos_a_b <= 1 else 1.0
    cos_a_b = cos_a_b if cos_a_b >= -1 else -1.0
    return cos_a_b


def euclidean_distance(vertora, vertorb):
    # 标准化
    vertora = np.asarray(vertora)
    vertorb = np.asarray(vertorb)
    return np.sqrt(np.sum(np.square(vertora-vertorb)))


df = pd.read_csv('community_2.csv', encoding='utf-8')
IG = nx.Graph()
G = nx.from_pandas_edgelist(df, '实体', '值', '属性')
N = G.number_of_nodes()
print(G.nodes())
label_for_node = dict((i, v) for i, v in enumerate(G.nodes()))
print(label_for_node)
communities = dict((i, frozenset([i])) for i in range(N))
print(communities)
partition = [[label_for_node[x] for x in c] for c in communities.values()]
q_cnm = modularity(G, partition)
print(q_cnm)
# a = np.zeros((3, 2), dtype=np.int)
# a[1][0] = 1
# a[1][1] = 0
# a[2][0] = 5
# a[2][0] = 1
# print(vector_cos(a[1], a[2]))
# x = [1, 1, 1]
# y = [2, 2, 2]
# print(vector_cos(x, y))
# print(euclidean_distance(x, y))
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
