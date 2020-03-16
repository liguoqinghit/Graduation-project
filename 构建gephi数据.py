import pandas as pd
import networkx as nx

if __name__ == '__main__':
    '''
    构建gephi的点表和边表
    (1) 根据 社团状态信息(社团划分的结果, 即每个社团包含的节点) 创建点表和边表
    (2) 根据 实体-属性-值 构建边表
    (3) 根据 实体-属性-值 创建pair文件(主要是一些算法不考虑边的属性) 实体-值
    '''
    '''
    df = pd.read_csv(r'C:\work\graph_of_knowledge\group-of-service\entity4.csv', encoding='utf-8')
    # df = pd.read_csv('community_3_index.csv', encoding='utf-8')
    G = nx.from_pandas_edgelist(df, '实体', '值', '属性')
    all_nodes = G.nodes
    all_edges = list(G.edges.data())

    label_for_node = dict((i, v) for i, v in enumerate(all_nodes))
    node_for_label = dict((label_for_node[i], i) for i in range(len(all_nodes)))

    # node_giant = ['旅游', '地理', '食品', '菜品']
    # node_giant_id = [node_for_label[i] for i in node_giant]
    # print(node_giant_id)
    # [136, 137, 6, 16]

    # 建立点表
    # Id   Label   Modularity Class

    communities = []
    # 原社团划分的点表
    # with open('原社团划分(index).txt', 'r', encoding='utf-8') as f:
    # 关系社团划分的点表
    with open('test_partition2_4.csv', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split(',')
            communities.append(line)
    communities = [[int(i) for i in community] for community in communities]
    modularity = []
    id = 0
    for community in communities:
        for node in community:
            modularity.append([node, label_for_node[node], id])
        id += 1
    # print(sorted(modularity))
    df_nodes = pd.DataFrame(sorted(modularity), columns=['Id', 'Label', 'Modularity Class'])
    # df_nodes.to_csv('gephi_原社团划分_点.csv', encoding='utf-8', index=False)
    df_nodes.to_csv('gephi_关系社团划分_4_2_点.csv', encoding='utf-8', index=False)
    # df_nodes = pd.read_csv('gephi_原社团划分_点.csv', encoding='utf-8')
    # nodes = df_nodes.values.tolist()
    # print(nodes)

    # 建立边表
    # Source   Target   Type   Id   Label   Weight
    #                   undirected             1
    id = 0
    edges = []
    for i, j, r in all_edges:
        edges.append([node_for_label[i], node_for_label[j], 'undirected', id, r['属性'], 1])
        id += 1
    print(edges)
    df_edges = pd.DataFrame(edges, columns=['Source', 'Target', 'Type', 'Id', 'Label', 'Weight'])
    df_edges.to_csv('gephi_关系社团划分_4_2_边.csv', encoding='utf-8', index=False)
    '''
    # 根据 实体-属性-值 构建边表
    id = 0
    edges = []
    with open(r'community_6.csv', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(',')
            edges.append([line[0], line[2], 'undirected', id, line[1], 1])
            id += 1
    df_edges = pd.DataFrame(edges, columns=['Source', 'Target', 'Type', 'Id', 'Label', 'Weight'])
    df_edges.to_csv('gephi_关系社团划分_6_01_边.csv', encoding='utf-8', index=False)

    # 根据 实体 属性 值 创建pair文件 实体 - 值
    # with open('test.pairs', 'a', encoding='utf-8') as w:
    #     with open('community_6.csv', 'r', encoding='utf-8') as f:
    #         for line in f:
    #             line = line.strip().split(',')
    #             w.writelines([line[0] + ' ', line[2] + '\n'])
