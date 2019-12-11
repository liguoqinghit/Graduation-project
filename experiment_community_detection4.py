import networkx as nx
import pandas as pd
import numpy as np
import copy
import time
import numpy_test
from networkx.algorithms.community import modularity


def merge_community2(v, w, com, C, RC, D, RD, Q_pre):
    edge_attr = np.zeros(len(columns_list), dtype=int)
    time0 = time.process_time()
    for x in com[v]:
        for y in com[w]:
            if G.has_edge(x, y):
                edge_attr[int(G[x][y]['属性'])] += 1
    if np.sum(edge_attr) > 0:
        time1 = time.process_time()
        # C
        C[w] += edge_attr
        # RC
        rc = np.sum(C[w]) / len(columns_list)
        RC[w] = np.linspace(rc, rc, len(columns_list))
        # D RD
        for k in range(len(com)):
            if k == w or k == v:
                continue
            D[w][k] = numpy_test.standardized_euclidean_distance(C[w], C[k], C)
            RD[w][k] = numpy_test.standardized_euclidean_distance(RC[w], RC[k], RC)
        # QR
        qr2 = np.sum(D) - np.sum(RD)
        time2 = time.process_time()
        # QM
        com[w] = frozenset(com[v] | com[w])
        del com[v]
        part = [[co for co in community] for community in com.values()]
        qm2 = modularity(G, part)
        time3 = time.process_time()
        # 计算Q
        Q = p2 * qm2 - p1 * qr2
        print('循环内的时间花费')
        print(time0-time1)
        print('QR:', time2 - time1)
        print('QM:', time3 - time2)
        return Q
    return Q_pre


t1 = time.process_time()
# 构建图
df = pd.read_csv('community_2_index.csv', encoding='utf-8')
G = nx.from_pandas_edgelist(df, '实体', '值', '属性')
all_nodes = G.nodes
df.drop_duplicates(subset=[df.columns[2]], keep='first', inplace=True)
columns_list = df[df.columns[2]].values.tolist()
communities = dict((i, frozenset([i])) for i in range(len(all_nodes)))

# 初始化状态
# C、RC、D、RD以及社团划分状态
# 循环次数、QR和QM的参数、全局Q的最大值及其对应的社团状态、融合的社团(即需要忽略的)
matrix_c = np.zeros((len(all_nodes), len(columns_list)), dtype=int)
matrix_rc = np.zeros((len(all_nodes), len(columns_list)), dtype=int)
matrix_d = np.zeros((len(all_nodes), len(all_nodes)))
matrix_rd = np.zeros((len(all_nodes), len(all_nodes)))
partition = [[i for i in community] for community in communities.values()]
cycle_times = len(all_nodes) - 1
p1 = 0.2
p2 = 0.8
q_max = -float('inf')
q_max_communities = copy.deepcopy(communities)
ignore = []

# 构建点对,类似[1, 0]
pair = []
for i in range(len(all_nodes)):
    for j in range(i):
        if i == j:
            continue
        pair.append([i, j])

t2 = time.process_time()
print('构建图和初始化花费时间：', t2 - t1)

while len(communities) > 1:
    q = -float('inf')
    temp = ()
    # 循环遍历所有点对（不重复且不相等）
    for i, j in pair:
        if i in ignore or j in ignore:
            continue
        # 初始循环内的变量
        matrix_c2 = copy.deepcopy(matrix_c)
        matrix_rc2 = copy.deepcopy(matrix_rc)
        matrix_d2 = copy.deepcopy(matrix_d)
        matrix_rd2 = copy.deepcopy(matrix_rd)
        communities2 = copy.deepcopy(communities)
        # 社团融合i j
        q2 = merge_community2(i, j, communities2, matrix_c2, matrix_rc2, matrix_d2, matrix_rd2, q)
        # 选取循环内q的最大值

        if q2 > q:
            q = q2
            temp = (i, j)
        if i == 100:
            break

    # 根据循环内最大值，更新社团状态以及全局状态
    if temp:
        a, b = temp
        q = merge_community2(a, b, communities, matrix_c, matrix_rc, matrix_d, matrix_rd, q)
        ignore.append(a)
        # 更新全局q及其对应社团状态
        if q > q_max:
            q_max = q
            q_max_communities = copy.deepcopy(communities)

    break
t3 = time.process_time()
print('合并一次社团花费时间:', t3 - t2)
print(ignore)

