import random
import networkx as nx
import pandas as pd
import numpy as np
import copy
import time
import numpy_test
from concurrent import futures
from networkx.algorithms.community import modularity


def merge_community2(v, w, com, C, RC, D, RD, columns_list, G, p1=0.2, p2=0.8, edge_ignore=[]):
    if len(com) < 50:
        p1 = 0.8
        p2 = 0.2
    edge_attr = np.zeros(len(columns_list), dtype=int)
    for x in com[v]:
        for y in com[w]:
            if G.has_edge(x, y) and G[x][y]['属性'] not in edge_ignore:
                edge_attr[int(G[x][y]['属性'])] += 1
    time1 = time.process_time()
    # C
    C[w] -= edge_attr
    C[w] += C[v]
    C[v] = np.zeros(len(columns_list), dtype=int)
    # RC
    rc = np.sum(C[w]) / len(columns_list)
    RC[w] = np.linspace(rc, rc, len(columns_list))
    RC[v] = np.zeros(len(columns_list))

    # D RD
    # 忽略v的大小
    # v合并到w
    D[v] = np.zeros(len(all_nodes))
    RD[v] = np.zeros(len(all_nodes))
    for k in range(len(com)):
        if k == w or k == v:
            # D[w][k] = 0
            # D[k][w] = 0
            # RD[w][k] = 0
            # RD[k][w] = 0
            continue
        else:
            D[w][k] = numpy_test.standardized_euclidean_distance(C[w], C[k], C)
            RD[w][k] = numpy_test.standardized_euclidean_distance(RC[w], RC[k], RC)

    # 正规化D 和 RD
    # (1) 使用max和min (x - min) / (max - min)  x / max
    D_max = np.max(D)
    D_min = np.min(D)
    ND = [[(D[i][j] - D_min) / (D_max - D_min) for j in range(len(D[i]))] for i in range(len(D))]
    RD_max = np.max(RD)
    RD_min = np.min(RD)
    NRD = [[(RD[i][j] - RD_min) / (RD_max - RD_min) for j in range(len(RD[i]))] for i in range(len(RD))]
    # QR
    qr2 = 2 * (np.sum(ND) - np.sum(NRD)) / len(D) / (len(D) + 1)
    print('qr2:', qr2)
    time2 = time.process_time()
    # QM
    com[w] = frozenset(com[v] | com[w])
    del com[v]
    part = [[co for co in community] for community in com.values()]
    qm2 = modularity(G, part)
    print('qm2:', qm2)
    print('\n\n')
    time3 = time.process_time()
    # 计算Q
    Q = p2 * qm2 - p1 * qr2
    return Q, qm2, qr2


def merge_community_job(vwpair, com, C, RC, D, RD, Q_pre, columns_list, G, ijpair=(), edge_ignore=[]):
    for i, j in vwpair:
        # 初始循环内的变量
        matrix_c2 = copy.deepcopy(C)
        matrix_rc2 = copy.deepcopy(RC)
        matrix_d2 = copy.deepcopy(D)
        matrix_rd2 = copy.deepcopy(RD)
        communities2 = copy.deepcopy(com)
        # 社团融合i j
        q2, qm, qr = merge_community2(i, j, communities2, matrix_c2, matrix_rc2, matrix_d2, matrix_rd2, columns_list, G, 0.2, 0.8, edge_ignore)
        # 选取循环内q的最大值
        if q2 > Q_pre:
            Q_pre = q2
            ijpair = (i, j)
        else:
            ijpair = vwpair[random.randint(0, len(vwpair) - 1)]
    return Q_pre, ijpair


if __name__ == '__main__':
    # 构建图
    t1 = time.time()
    df = pd.read_csv('community_6_2_index.csv', encoding='utf-8')
    g = nx.Graph()
    G = nx.from_pandas_edgelist(df, '实体', '值', '属性')
    all_nodes = list(G.nodes)
    all_edges = list(G.edges)
    df.drop_duplicates(subset=[df.columns[1]], keep='first', inplace=True)
    columns_list = df[df.columns[1]].values.tolist()
    communities = dict((i, frozenset([i])) for i in range(len(all_nodes)))
    all_degree = list(G.degree())

    # 巨人节点 度大于20 暂定
    node_ignore = []
    # for key, value in all_degree:
    #     if value > 80:
    #         node_ignore.append(key)

    # 巨人边 从numpy_test.py读取出来的
    # edge_ignore = [1]
    edge_ignore = []

    # 初始化状态
    # C、RC、D、RD以及社团划分状态
    # 循环次数、QR和QM的参数、全局Q的最大值及其对应的社团状态、融合的社团(即需要忽略的)
    matrix_c = np.zeros((len(all_nodes), len(columns_list)), dtype=int)
    for node1, node2, relation in G.edges.data():
        if relation['属性'] in edge_ignore:
            continue
        matrix_c[node1][relation['属性']] += 1
        matrix_c[node2][relation['属性']] += 1

    matrix_rc = np.zeros((len(all_nodes), len(columns_list)))
    for i in range(len(matrix_c)):
        rc = np.sum(matrix_c[i]) / len(columns_list)
        matrix_rc[i] = np.linspace(rc, rc, len(columns_list))
    print(matrix_rc)

    matrix_d = np.zeros((len(all_nodes), len(all_nodes)))
    matrix_rd = np.zeros((len(all_nodes), len(all_nodes)))
    # 可以写在建立点对pair后面
    nodes_list = []
    for node1 in all_nodes:
        nodes_list.append(node1)
        for node2 in nodes_list:
            if node1 == node2:
                continue
            matrix_d[node1][node2] = numpy_test.standardized_euclidean_distance(matrix_c[node1], matrix_c[node2], matrix_c)
            matrix_rd[node1][node2] = numpy_test.standardized_euclidean_distance(matrix_rc[node1], matrix_rc[node2], matrix_rc)

    print(sum(matrix_d))
    print(sum(matrix_rd))
    partition = [[i for i in community] for community in communities.values()]
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

    t2 = time.time()
    print('构建图和初始化花费时间：', t2 - t1)

    t3 = time.time()
    while len(communities) > 1:
        q = -float('inf')
        ijpair = ()

        # 循环遍历所有点对（不重复且不相等）
        workers = 32
        with futures.ProcessPoolExecutor(max_workers=workers) as executor:
            worker_to_pair = [[] for i in range(workers)]
            future_list = []
            # 输出QR QM写入文件
            # 画个图
            if len(communities) > 300:
            # if False:
                for idx in range(len(all_edges)):
                    i, j = all_edges[idx]
                    if i in ignore or j in ignore or i in node_ignore or j in node_ignore:
                        continue
                    # 如果巨人节点都被加入到ignore即ignore包含node_ignore的所有点
                    # judge = [False for node in node_ignore if node not in ignore]
                    # if i in node_ignore or j in node_ignore or not judge:
                    worker_to_pair[idx % workers].append(all_edges[idx])
            else:
                for idx in range(len(pair)):
                    i, j = pair[idx]
                    if i in ignore or j in ignore or i in node_ignore or j in node_ignore:
                        continue
                    worker_to_pair[idx % workers].append(pair[idx])
            for idx in range(workers):
                future_list.append(
                    executor.submit(merge_community_job, worker_to_pair[idx], communities, matrix_c, matrix_rc,
                                    matrix_d, matrix_rd, q, columns_list, G, ijpair, edge_ignore))

            for future in futures.as_completed(future_list):
                q2, ijpair2 = future.result()
                if q2 > q:
                    q = q2
                    ijpair = ijpair2
            # 记录融合结果，还没进行过测试
            #     if len(communities) == 123:
            #         with open('%sC%s.txt' % (ijpair2), 'w', encoding='utf-8') as f:
            #             for i in matrix_c.tolist():
            #                 f.writelines(str(i) + '\n')
            #             f.writelines('Q:', q2)
        if ijpair or ijpair2 == ijpair:
            a, b = ijpair
            if ijpair2 == ijpair:
                a, b = ijpair2
            print('融合 %s - %s' % (a, b))
            q, qm, qr = merge_community2(a, b, communities, matrix_c, matrix_rc, matrix_d, matrix_rd, columns_list, G, 0.2, 0.8, edge_ignore)
            ignore.append(a)
            filename = str(len(communities))
            partition = [[i for i in community] for community in communities.values()]
            # numpy_test.two_dimensional_list_to_file('community%s.csv' % filename, partition)
            with open('node2.txt', 'a+', encoding='utf-8') as w:
                line = [q, qm, qr]
                for i in line:
                    w.write(str(i))
                    w.write(' ')
                w.write('\n')
            # 更新全局q及其对应社团状态
            if q > q_max:
                q_max = q
                q_max_communities = copy.deepcopy(communities)
        print('communities len: ', len(communities), time.time() - t3)
        t3 = time.time()
    print('合并社团花费时间:', time.time() - t2)
    print(ignore)
    partition = [[i for i in community] for community in q_max_communities.values()]
    numpy_test.two_dimensional_list_to_file('test_partition2_6_2.csv', partition)
