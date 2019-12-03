import networkx as nx
import pandas as pd
import numpy as np
import copy
import numpy_test

"""
数据结构：
    点的总数记作nodes，边的总数记作edges，边的种类记作types
    1.需要一个Q_ij(初始大小为nodes^2) 记录将某两个社团i和j融合到一起后（记作j）Q的值 
    2.需要一个nodes * types大小的邻接矩阵matrix记录每个社团包含的边的种类的数量
    3.………………
    
思路:
    一、构建图 创建边表或者直接用pandas与networkx的转化
    1.读取csv文件中的数据，dataframe转化到list或narray
    2.以[(实体 实体 {属性(或关系) : 次数}),(……)……]的列表作为边表，构建图
    二、建立初始值
    以每个点（一共355个点）作为一个社团，建立一个dataframe，以社团+96个社团属性类别作为列元素
    matrix_C （96+1）* 355
    matrix_RC 355
    matrix_D 
    matrix_RD
    matrix_QR
    node_community 以node为索引
    三、根据初始值和已知式子开始进行社团划分
    计算QRij，即合并任意两个社团i和j，更新二中各个矩阵的值
    需要进行的操作：
    1.将C中社团i和社团j的值加和放到社团j中，删除C中的社团i
    2.将RC中社团i删除，重新计算RC中的社团j
    3.将D中社团i的行列删除，重新计算社团j的行列
    4.RD与D的变化相同
    5.计算QRij
    
"""


def calculation_qr(df_matrix_c, df_matrix_rc, df_matrix_d, df_matrix_rd, nodes_list):
    nodes_list2 = []
    for node1 in nodes_list:
        nodes_list2.append(node1)
        for node2 in nodes_list2:
            if node1 == node2:
                break
            # 计算Dvw
            c_node1 = df_matrix_c.loc[node1].values
            c_node1 = np.delete(c_node1, 0, axis=0)
            df_matrix_rc.loc[:, node1] = np.sum(c_node1)
            # print(c_node1)
            c_node2 = df_matrix_c.loc[node2].values
            c_node2 = np.delete(c_node2, 0, axis=0)
            df_matrix_rc.loc[:, node2] = np.sum(c_node2)
            # print(c_node2)
            temp1 = numpy_test.euclidean_distance(c_node1, c_node2)
            df_matrix_d.loc[node1, node2] = temp1
            df_matrix_d.loc[node2, node1] = temp1

            # 计算RDvw
            sum1 = df_matrix_rc.loc[0, node1]
            sum2 = df_matrix_rc.loc[0, node2]
            np_sum1 = np.linspace(sum1 / 96, sum1 / 96, 96)
            np_sum2 = np.linspace(sum2 / 96, sum2 / 96, 96)
            temp2 = numpy_test.euclidean_distance(np_sum1, np_sum2)
            df_matrix_rd.loc[node1, node2] = temp2
            df_matrix_rd.loc[node2, node1] = temp2
    # 计算QR
    QR = (np.sum(df_matrix_d.values) - np.sum(df_matrix_rd.values)) / 2
    return QR


df = pd.read_csv('community_2.csv', encoding='utf-8')
# 遍历创建边表
# df2 = df.values
# edges_list = list()
# for row in df2:
#     edges_list.append((row[1], row[3], {'属性': row[2]}))
# print(edges_list)
#
# G = nx.Graph()
# G.add_edges_from(edges_list)
# print(G.number_of_nodes())
# print(G.number_of_edges())
# print(G.edges.data())

# 根据pandas的建立图 和上面的相同
# G2 = nx.Graph()
G2 = nx.from_pandas_edgelist(df, '实体', '值', '属性')
# print(G2.number_of_nodes())
# print(G2.number_of_edges())
print(G2.edges.data())
df2 = df
df2.drop_duplicates(subset=[df.columns[2]], keep='first', inplace=True)
df2 = df2[df.columns[2]]
columns_list = ['社团']
columns_list += df2.values.tolist()
print(columns_list)
# print(len(columns_list)) 97个 社团 + 96个社团的属性分类
# df_matrix_c记录C值，即每个向量
matrix_c = np.zeros([G2.number_of_nodes(), 97], dtype=int)
df_matrix_c: pd.DataFrame = pd.DataFrame(matrix_c, index=G2.nodes, columns=columns_list)
# print(df_matrix_c)
# print(df_matrix_c.loc['贡丸']['主要食材'])
# 把每个点赋予一个社团初值
i = 0
for row in G2.nodes:
    df_matrix_c.loc[row, '社团'] = i
    i += 1
# 统计每个点的向量值
for node1, node2, relation in G2.edges.data():
    for kv in relation:
        df_matrix_c.loc[node1, relation[kv]] += 1
        df_matrix_c.loc[node2, relation[kv]] += 1
# print(df_matrix_c)

# df_matrix_rc记录理论Cv值，即向量的平均值
matrix_rc = np.zeros([1, G2.number_of_nodes()])
df_matrix_rc = pd.DataFrame(matrix_rc, columns=G2.nodes)

# df_matrix_d记录Dvw，即vw的关系向量距离
matrix_d = np.zeros([G2.number_of_nodes(), G2.number_of_nodes()], dtype=int)
df_matrix_d = pd.DataFrame(matrix_d, index=G2.nodes, columns=G2.nodes)
# df_matrix_rd记录RDvw，即vw的平均关系向量距离
matrix_rd = np.zeros([G2.number_of_nodes(), G2.number_of_nodes()], dtype=int)
df_matrix_rd = pd.DataFrame(matrix_rd, index=G2.nodes, columns=G2.nodes)

# QR = calculation_qr(df_matrix_rc, df_matrix_d, df_matrix_rd, G2)
all_nodes = G2.nodes
nodes_list = []
for node1 in all_nodes:
    nodes_list.append(node1)
    for node2 in nodes_list:
        if node1 == node2:
            continue
        # 计算Dvw
        c_node1 = df_matrix_c.loc[node1].values
        c_node1 = np.delete(c_node1, 0, axis=0)
        df_matrix_rc.loc[:, node1] = np.sum(c_node1)
        # print(c_node1)
        c_node2 = df_matrix_c.loc[node2].values
        c_node2 = np.delete(c_node2, 0, axis=0)
        df_matrix_rc.loc[:, node2] = np.sum(c_node2)
        # print(c_node2)
        temp1 = numpy_test.euclidean_distance(c_node1, c_node2)
        df_matrix_d.loc[node1, node2] = temp1
        df_matrix_d.loc[node2, node1] = temp1

        # 计算RDvw
        sum1 = df_matrix_rc.loc[0, node1]
        sum2 = df_matrix_rc.loc[0, node2]
        np_sum1 = np.linspace(sum1 / 96, sum1 / 96, 96)
        np_sum2 = np.linspace(sum2 / 96, sum2 / 96, 96)
        temp2 = numpy_test.euclidean_distance(np_sum1, np_sum2)
        df_matrix_rd.loc[node1, node2] = temp2
        df_matrix_rd.loc[node2, node1] = temp2
# 计算QR
QR = (np.sum(df_matrix_d.values) - np.sum(df_matrix_rd.values)) / 2
# 输出
# C的求和没有去除 社团编号的求和 理论上 C = RC
print(np.sum(df_matrix_c.values))
print(np.sum(df_matrix_rc.values))
print(np.sum(df_matrix_d.values))
print(np.sum(df_matrix_rd.values))
print(QR)
# dmax = np.max(df_matrix_d.values)
# print(dmax)
# print(node_list)
# print(distance_min)

# print(numpy_test.euclidean_distance(np_sum1, np_sum2))
# print(df_matrix_c)
# print(df_matrix_c.loc['猪肉']['主要食材'])


# G2边表是三维 G2[一个节点][另一个节点][边属性] = 边属性的值
# dic = dict()
# dic = G2['猪肉']['贡丸']
# something = 'ds'
# something = '属性'
# if something not in dic.keys():
#     print('mude')
# else:
#     print(G2['猪肉']['贡丸'][something])
# print(G2['贡丸'])
# print(G2['生活'])
"""

# 合并v和w
nodes_list = []
i = 0
for node1 in G2.nodes:
    nodes_list.append(node1)
    for node2 in nodes_list:
        if node1 == node2:
            break
        # drop后的c2改变了不会改变c
        # 把node2的加到node1
        df_matrix_c2 = copy.deepcopy(df_matrix_c)
        for column in columns_list:
            if column == '社团':
                continue
            df_matrix_c2.loc[node1, column] += df_matrix_c2.loc[node2, column]
        df_matrix_c2 = df_matrix_c2.drop(node2)
        df_matrix_rc2 = df_matrix_rc.drop(node2, axis=1)
        df_matrix_d2 = df_matrix_d.drop(node2)
        df_matrix_d2 = df_matrix_d.drop(node2, axis=1)
        df_matrix_rd2 = df_matrix_rd.drop(node2)
        df_matrix_rd2 = df_matrix_rd.drop(node2, axis=1)
        temp_nodes_list = list(copy.deepcopy(G2.nodes))
        temp_nodes_list.remove(node2)
        QR2 = calculation_qr(df_matrix_c2, df_matrix_rc2, df_matrix_d2, df_matrix_rd2, temp_nodes_list)
        if QR2 < QR:
            QR = QR2
            # print(QR)
            df_matrix_c3 = copy.deepcopy(df_matrix_c2)
            df_matrix_rc3 = copy.deepcopy(df_matrix_rc2)
            df_matrix_d3 = copy.deepcopy(df_matrix_d2)
            df_matrix_rd3 = copy.deepcopy(df_matrix_rd2)
    # i += 1
    # if i == 3:
    #     break
print(df_matrix_c2)

df_matrix_c3.to_csv('c.csv', encoding='utf-8')
df_matrix_rc3.to_csv('rc.csv', encoding='utf-8')
df_matrix_d3.to_csv('d.csv', encoding='utf-8')
df_matrix_rd3.to_csv('rd.csv', encoding='utf-8')
"""








