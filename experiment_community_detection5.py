import random
import networkx as nx
import pandas as pd
import numpy as np
import copy
import time
import numpy_test
from concurrent import futures
from networkx.algorithms.community import modularity
from experiment_community_detection4 import merge_community2
from experiment_community_detection4 import merge_community_job


if __name__ == '__main__':
    # 构建图
    t1 = time.time()
    df = pd.read_csv('community_3_index.csv', encoding='utf-8')
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
    # node_ignore = ['旅游', '地理', '食品', '菜品']
    # node_ignore = [136, 137]
    # n_ignore = [6, 16]

    for key, value in all_degree:
        if value > 20:
            node_ignore.append(key)

    # 初始化状态
    # C、RC、D、RD以及社团划分状态
    # 循环次数、QR和QM的参数、全局Q的最大值及其对应的社团状态、融合的社团(即需要忽略的)
    matrix_c = np.zeros((len(all_nodes), len(columns_list)), dtype=int)
    matrix_rc = np.zeros((len(all_nodes), len(columns_list)), dtype=int)
    matrix_d = np.zeros((len(all_nodes), len(all_nodes)))
    matrix_rd = np.zeros((len(all_nodes), len(all_nodes)))
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
    while len(communities)-len(node_ignore) > 1:
        q = -float('inf')
        ijpair = ()

        # 循环遍历所有点对（不重复且不相等）
        workers = 32
        with futures.ProcessPoolExecutor(max_workers=workers) as executor:
            worker_to_pair = [[] for i in range(workers)]
            future_list = []
            if len(communities) > 250:
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
                                    matrix_d, matrix_rd, q, columns_list, G, ijpair))
            for future in futures.as_completed(future_list):
                q2, ijpair2 = future.result()
                if q2 > q:
                    q = q2
                    ijpair = ijpair2
        if ijpair:
            a, b = ijpair
            q = merge_community2(a, b, communities, matrix_c, matrix_rc, matrix_d, matrix_rd, columns_list, G)
            ignore.append(a)
            # 更新全局q及其对应社团状态
            if q > q_max:
                q_max = q
                q_max_communities = copy.deepcopy(communities)
        # if round >= 1:
        #     break
        print('communities len: ', len(communities), time.time() - t3)
        t3 = time.time()
    print('合并社团花费时间:', time.time() - t2)
    print(ignore)
    partition = [[i for i in community] for community in q_max_communities.values()]
    numpy_test.two_dimensional_list_to_file('test_partition2_3.csv', partition)




