import networkx as nx
import pandas as pd
import numpy as np
import copy
import time
import numpy_test

start = time.process_time()
df = pd.read_csv('community_2.csv', encoding='utf-8')
G2 = nx.from_pandas_edgelist(df, '实体', '值', '属性')
print(G2.edges.data())
all_nodes = list(copy.deepcopy(G2.nodes))
df2 = df
df2.drop_duplicates(subset=[df.columns[2]], keep='first', inplace=True)
# columns_list = ['社团']
columns_list = df2[df.columns[2]].values.tolist()
print(columns_list)

matrix_c = np.zeros([len(all_nodes), len(columns_list)], dtype=int)
node_community = np.zeros(len(all_nodes))
for i in range(len(all_nodes)):
    node_community[i] = int(i)
for node1, node2, relation in G2.edges.data():
    for kv in relation:
        matrix_c[all_nodes.index(node1)][columns_list.index(kv)] += 1
        matrix_c[all_nodes.index(node2)][columns_list.index(kv)] += 1
# print(node_community)
# print(np.sum(matrix_c[0]))

matrix_rc = np.zeros(G2.number_of_nodes())
matrix_d = np.zeros([G2.number_of_nodes(), G2.number_of_nodes()], dtype=int)
matrix_rd = np.zeros([G2.number_of_nodes(), G2.number_of_nodes()], dtype=int)

nodes_list = []
for node1 in all_nodes:
    nodes_list.append(node1)
    for node2 in nodes_list:
        if node1 == node2:
            continue
        node1_id = all_nodes.index(node1)
        c_node1 = matrix_c[node1_id]
        matrix_rc[node1_id] = np.sum(c_node1)
        node2_id = all_nodes.index(node2)
        c_node2 = matrix_c[node2_id]
        matrix_rc[node2_id] = np.sum(c_node2)
        # 计算D
        temp1 = numpy_test.euclidean_distance(c_node1, c_node2)
        matrix_d[node1_id, node2_id] = temp1
        matrix_d[node2_id, node1_id] = temp1
        # 计算RD
        np_sum1 = np.linspace(matrix_rc[node1_id] / 96, matrix_rc[node1_id] / 96, 96)
        np_sum2 = np.linspace(matrix_rc[node2_id] / 96, matrix_rc[node2_id] / 96, 96)
        temp2 = numpy_test.euclidean_distance(np_sum1, np_sum2)
        matrix_rd[node1_id, node2_id] = temp2
        matrix_rd[node2_id, node1_id] = temp2
QR = (np.sum(matrix_d) - np.sum(matrix_rd)) / 2

print(np.sum(matrix_c))
print(np.sum(matrix_rc))
print(np.sum(matrix_d))
print(np.sum(matrix_rd))
print(QR)

end = time.process_time()
print('time:' + str(end-start))























