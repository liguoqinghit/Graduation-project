import networkx as nx
import pandas as pd
import numpy as np
import copy
import time
import numpy_test


def np_sum(a):
    b = copy.deepcopy(a)
    for x in range(a):
        for y in range(a[0]):
            if b[x][y] == float('inf'):
                b[x][y] = 0
    return np.sum(b)


df = pd.read_csv('community_2.csv', encoding='utf-8')
G2 = nx.from_pandas_edgelist(df, '实体', '值', '属性')
print(G2.edges.data())
all_nodes = list(copy.deepcopy(G2.nodes))
df2 = df
df2.drop_duplicates(subset=[df.columns[2]], keep='first', inplace=True)
# columns_list = ['社团']
columns_list = df2[df.columns[2]].values.tolist()
print(columns_list)
matrix_c = np.zeros((len(all_nodes), len(columns_list)), dtype=int)
node_community = np.zeros(len(all_nodes))
# for i in range(len(all_nodes)):
#     node_community[i] = int(i)
# for node1, node2, relation in G2.edges.data():
#     for kv in relation:
#         matrix_c[all_nodes.index(node1)][columns_list.index(kv)] += 1
#         matrix_c[all_nodes.index(node2)][columns_list.index(kv)] += 1
# print(node_community)
# print(np.sum(matrix_c[0]))

matrix_rc = np.zeros(len(all_nodes))
# D和RD的初始值不应该为0 需要修改，正无穷=float('inf') 计算的时候手动按0计算
# 初值全为0
matrix_d = np.linspace(float('inf'), float('inf'), len(all_nodes))
matrix_rd = np.linspace(float('inf'), float('inf'), len(all_nodes))

start = time.process_time()
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
        matrix_d[node1_id][node2_id] = temp1
        matrix_d[node2_id][node1_id] = temp1
        # 计算RD
        np_sum1 = np.linspace(matrix_rc[node1_id] / 96, matrix_rc[node1_id] / 96, 96)
        np_sum2 = np.linspace(matrix_rc[node2_id] / 96, matrix_rc[node2_id] / 96, 96)
        temp2 = numpy_test.euclidean_distance(np_sum1, np_sum2)
        matrix_rd[node1_id][node2_id] = temp2
        matrix_rd[node2_id][node1_id] = temp2
QR = (np.sum(matrix_d) - np.sum(matrix_rd)) / 2
print(np.sum(matrix_c))
print(np.sum(matrix_rc))
print(np.sum(matrix_d))
print(np.sum(matrix_rd))
print(QR)

end1 = time.process_time()
print('time:' + str(end1 - start))

# 临时存放状态的
temp_cur_nodes_list = []
temp_matrix_c = np.zeros((len(all_nodes), len(columns_list)), dtype=int)
temp_matrix_rc = np.zeros(len(all_nodes))
temp_matrix_d = np.zeros((len(all_nodes), len(all_nodes)))
temp_matrix_rd = np.zeros((len(all_nodes), len(all_nodes)))

cur_nodes_list = all_nodes
flag = False
while True:
    if flag:
        print('重新赋值')
        cur_nodes_list = copy.deepcopy(temp_cur_nodes_list)
        matrix_c = copy.deepcopy(temp_matrix_c)
        matrix_rc = copy.deepcopy(temp_matrix_rc)
        matrix_d = copy.deepcopy(temp_matrix_d)
        matrix_rd = copy.deepcopy(temp_matrix_rd)
    flag = False
    i = 0
    nodes_list2 = []
    for node1 in cur_nodes_list:
        nodes_list2.append(node1)
        for node2 in nodes_list2:
            if node1 == node2:
                continue
            # 更新cur_
            temp_nodes_list = copy.deepcopy(cur_nodes_list)
            temp_nodes_list.remove(node2)
            # 更新C node2加入到node1，删除node2的C
            matrix_c2 = copy.deepcopy(matrix_c)
            node1_id = cur_nodes_list.index(node1)
            node2_id = cur_nodes_list.index(node2)
            matrix_c2[node1_id] = np.sum(matrix_c2[[node1_id, node2_id]], axis=0)
            matrix_c2 = np.delete(matrix_c2, node2_id, axis=0)
            # 更新RC node2加入到node1，删除node2的C
            matrix_rc2 = copy.deepcopy(matrix_rc)
            matrix_rc2[node1_id] += matrix_rc2[node2_id]
            matrix_rc2 = np.delete(matrix_rc2, node2_id)
            # 更新D 更新RD
            matrix_d2 = copy.deepcopy(matrix_d)
            matrix_rd2 = copy.deepcopy(matrix_rd)
            for temp_node in temp_nodes_list:
                if node1 == temp_node:
                    continue
                temp_node_id = cur_nodes_list.index(temp_node)
                c_temp_node = matrix_c[temp_node_id]
                # 计算D
                temp1 = numpy_test.euclidean_distance(matrix_c2[node1_id], c_temp_node)
                matrix_d2[node1_id][temp_node_id] = temp1
                matrix_d2[temp_node_id][node1_id] = temp1
                # 计算RD
                np_sum1 = np.linspace(matrix_rc[node1_id] / 96, matrix_rc[node1_id] / 96, 96)
                np_sum2 = np.linspace(matrix_rc[temp_node_id] / 96, matrix_rc[temp_node_id] / 96, 96)
                temp2 = numpy_test.euclidean_distance(np_sum1, np_sum2)
                matrix_rd2[node1_id][temp_node_id] = temp2
                matrix_rd2[temp_node_id][node1_id] = temp2
            # 删除node2 对应的D和RD
            matrix_d2 = np.delete(matrix_d2, node2_id, axis=0)
            matrix_d2 = np.delete(matrix_d2, node2_id, axis=1)
            matrix_rd2 = np.delete(matrix_rd2, node2_id, axis=0)
            matrix_rd2 = np.delete(matrix_rd2, node2_id, axis=1)
            temp_QR = (np.sum(matrix_d2) - np.sum(matrix_rd2)) / 2
            if temp_QR < QR:
                QR = temp_QR
                flag = True
                temp_cur_nodes_list = copy.deepcopy(temp_nodes_list)
                temp_matrix_c = copy.deepcopy(matrix_c2)
                temp_matrix_rc = copy.deepcopy(matrix_rc2)
                temp_matrix_d = copy.deepcopy(matrix_d2)
                temp_matrix_rd = copy.deepcopy(matrix_rd2)
        i += 1
        if i == 100:
            flag = False
            break
    if not flag:
        break
print(np.sum(temp_matrix_c))
print(np.sum(temp_matrix_rc))
print(np.sum(temp_matrix_d))
print(np.sum(temp_matrix_rd))
print(QR)
end2 = time.process_time()
print('time:' + str(end2 - end1))

"""
import numpy
narray=numpy.array(nlist)
sum1=narray.sum()
narray2=narray*narray
sum2=narray2.sum()
mean=sum1/N
var=sum2/N-mean**2
"""
