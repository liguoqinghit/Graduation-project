import pandas as pd
import copy
import networkx as nx
import numpy as np

if __name__ == '__main__':
    '''
    将 实体-属性-值 的数据转换为 1-1-1 类型的数据
    '''
    df = pd.read_csv('community_6.csv', encoding='utf-8')
    G = nx.from_pandas_edgelist(df, '实体', '值', '属性')
    df.drop_duplicates(subset=[df.columns[1]], keep='first', inplace=True)
    # 列元素
    columns_list = df[df.columns[1]].values.tolist()
    print(len(columns_list))
    # 点和边
    all_nodes = list(copy.deepcopy(G.nodes))
    all_edges = list(copy.deepcopy(G.edges.data()))

    # 建立矩阵 点的id 边属性的id 点的id  大小为 边的数量 * 3
    a = []
    for i, j, r in all_edges:
        a.append([all_nodes.index(i), columns_list.index(r['属性']), all_nodes.index(j)])

    df_a = pd.DataFrame(a, columns=['实体', '属性', '值'])
    df_a.to_csv('community_6_index.csv', encoding='utf-8', index=False)
    print(df_a)
