import pandas as pd
import networkx as nx

if __name__ == '__main__':

    df = pd.read_csv('community_2.csv', encoding='utf-8')
    # df = pd.read_csv('community_2_index.csv', encoding='utf-8')
    G = nx.from_pandas_edgelist(df, '实体', '值', '属性')
    all_nodes = G.nodes
    all_edges = list(G.edges.data())

    label_for_node = dict((i, v) for i, v in enumerate(all_nodes))
    node_for_label = dict((label_for_node[i], i) for i in range(len(all_nodes)))

    # 建立点表
    # Id   Label   Modularity Class

    communities = []
    # 原社团划分的点表
    # with open('原社团划分(index).txt', 'r', encoding='utf-8') as f:
    # 关系社团划分的点表
    with open('test_partition.csv', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split()
            communities.append(line)
    communities = [[int(i) for i in community] for community in communities]
    modularity = []
    id = 0
    for community in communities:
        for node in community:
            modularity.append([node, label_for_node[node], id])
        id += 1
    print(sorted(modularity))
    df_nodes = pd.DataFrame(sorted(modularity), columns=['Id', 'Label', 'Modularity Class'])
    # df_nodes.to_csv('gephi_原社团划分_点.csv', encoding='utf-8', index=False)
    df_nodes.to_csv('gephi_关系社团划分_点.csv', encoding='utf-8', index=False)
    # df_nodes = pd.read_csv('gephi_原社团划分_点.csv', encoding='utf-8')
    # nodes = df_nodes.values.tolist()
    # print(nodes)

    # 建立边表
    # Source   Target   Type   Id   Label   Weight
    #                   undirected             1
    # id = 0
    # edges = []
    # for i, j, r in all_edges:
    #     edges.append([node_for_label[i], node_for_label[j], 'undirected', id, r['属性'], 1])
    #     id += 1
    # print(edges)
    # df_edges = pd.DataFrame(edges, columns=['Source', 'Target', 'Type', 'Id', 'Label', 'Weight'])
    # df_edges.to_csv('gephi_边.csv', encoding='utf-8', index=False)

