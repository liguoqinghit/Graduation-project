import networkx as nx
import pandas as pd
import numpy as np
import copy
import time
import numpy_test
from networkx.algorithms.community import modularity
import source_code


def merge_community(v, w, com, C, RC, D, RD, Q_pre):
    # 计算QR
    # ij结合成一个社团，判断ij是否存在边，若存在，更新C、RC、R、RD
    # 存储ij社团结合后 边属性
    edge_attr = []
    for x in com[v]:
        for y in com[w]:
            if G.has_edge(label_for_node[x], label_for_node[y]):
                edge_attr.append(G[label_for_node[x]][label_for_node[y]]['属性'])
    if len(edge_attr) > 0:
        time1 = time.process_time()
        # C
        # TODO O(1)
        #
        for attr in edge_attr:
            C[w][columns_list.index(attr)] += 1
        # RC
        rc = np.sum(C[w]) / len(all_nodes)
        # TODO O(1)
        # RC[w] = np.linspace( rc )
        for i in range(len(columns_list)):
            RC[w][i] = rc
        # D and RD
        # 所有社团
        for k in range(len(all_nodes)):
            if k == w or k == v:
                continue
            D[w][k] = numpy_test.standardized_euclidean_distance(C[w], C[k], C)
            # D[k][w] = D[w][k]
            # rc1 = np.linspace(RC[w], RC[w], len(columns_list))
            # rc2 = np.linspace(RC[k], RC[k], len(columns_list))
            # RC_all = [[np.linspace(i, i, len(columns_list)) for i in RC]]
            RD[w][k] = numpy_test.standardized_euclidean_distance(RC[w], RC[k], RC)
            # RD[k][w] = RD
        # QR
        qr2 = np.sum(D) - np.sum(RD)
        time2 = time.process_time()

        # QM
        com[w] = frozenset(com[v] | com[w])
        del com[v]
        part = [[label_for_node[i] for i in community] for community in com.values()]
        qm2 = modularity(G, part)
        time3 = time.process_time()
        # 计算Q
        Q = p2 * qm2 - p1 * qr2
        print('循环内部的时间')
        print(time2 - time1)
        print(time3 - time2)
        return Q
        # return qm2
        # print(np.sum(D), np.sum(RD), qr2, qm2, Q)
    return Q_pre


t1 = time.process_time()
# 构建图
df = pd.read_csv('community_2.csv', encoding='utf-8')
G = nx.from_pandas_edgelist(df, '实体', '值', '属性')

# 原社团划分
"""
c = source_code.greedy_modularity_communities(G)
numpy_test.two_dimensional_list_to_file('原社团划分(实体).txt', c)
t2 = time.process_time()
print('结束:', t2-t1)  # 0.078125
"""


# 建立初始状态
# 点表、边表（含属性）和关系类别
all_nodes = list(copy.deepcopy(G.nodes))
all_edges = list(copy.deepcopy(G.edges.data()))
df.drop_duplicates(subset=[df.columns[2]], keep='first', inplace=True)
columns_list = df[df.columns[2]].values.tolist()

# 社团，点的两个字典（可以利用点表来表示，为了方便而建立的，后面为了减少内存的使用，可以修改）
communities = dict((i, frozenset([i])) for i in range(len(all_nodes)))
origin = dict((i, frozenset([i])) for i in range(len(all_nodes)))
label_for_node = dict((i, v) for i, v in enumerate(all_nodes))
node_for_label = dict((label_for_node[i], i) for i in range(len(all_nodes)))

# 初始化状态
# C、RC、D、RD以及社团划分状态
matrix_c = np.zeros((len(all_nodes), len(columns_list)), dtype=int)
matrix_rc = np.zeros((len(all_nodes), len(columns_list)), dtype=int)
matrix_d = np.zeros((len(all_nodes), len(all_nodes)))
matrix_rd = np.zeros((len(all_nodes), len(all_nodes)))
partition = [[label_for_node[i] for i in community] for community in communities.values()]
# 构建点对,类似[1, 0]
pair = []
for i in range(len(all_nodes)):
    for j in range(i):
        if i == j:
            continue
        pair.append([i, j])

# 要进行的循环次数（最外层），QR、QM和Q
c_length = len(matrix_c)
qr = (np.sum(matrix_d) - np.sum(matrix_rd)) / 2
qm = modularity(G, partition)
p1 = 0.2
p2 = 0.8
# q = p2 * qm - p1 * qr
t2 = time.process_time()
q_max = -float('inf')
ignore = []
print('构建图和初始化花费时间：', t2 - t1)
# 社团划分
while c_length - 1:
    c_length -= 1
    # 发生变化的临时变量
    q = -float('inf')
    temp = ()
    # 循环遍历所有点对（不重复且不相等）
    for i, j in pair:
        if i in ignore or j in ignore:
            continue
        matrix_c2 = copy.deepcopy(matrix_c)
        matrix_rc2 = copy.deepcopy(matrix_rc)
        matrix_d2 = copy.deepcopy(matrix_d)
        matrix_rd2 = copy.deepcopy(matrix_rd)
        communities2 = copy.deepcopy(communities)
        # q2 = -float('inf')

        # 函数可能不会发生改变
        # np.array 会发生改变
        # communities 也会发生改变
        q2 = merge_community(i, j, communities2, matrix_c2, matrix_rc2, matrix_d2, matrix_rd2, q)

        if q2 > q:
            q = q2
            temp = (i, j)
        # break
    # 输出并修改
    q_max = max(q_max, q)
    if temp:
        a, b = temp
        merge_community(a, b, communities, matrix_c, matrix_rc, matrix_d, matrix_rd, q)
        ignore.append(a)

    # 查看变化和时间
    print('第' + str(355-c_length) + '次')
    print(ignore)
    t4 = time.process_time()
    print('时间：', t4 - t2)
    if c_length == 354:
        break

# 思路还需要重新写一遍 方便更好的理解过程

# 通过集合的运算来查看社团划分前后的变化
# [[i,b[i]- a[i]&b[i]] for i in range(5) if i in a.keys() and i in b.keys()]
# change = [[i, origin[i]] for i in range(len(all_nodes))
#           if i not in communities.keys() and i in origin.keys()]
# print(change)



