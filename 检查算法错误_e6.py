import pandas as pd
import networkx as nx

if __name__ == '__main__':
    '''
    结论： 巨人结点的数据太多导致无法进行社团划分
    '''
    df = pd.read_csv('community_4.csv', encoding='utf-8')
    G = nx.from_pandas_edgelist(df, '实体', '值', '属性')
    all_nodes = list(G.nodes)
    all_edges = list(G.edges)
    all_degree = list(G.degree())
    node_ignore = []
    for key, value in all_degree:
        if value > 20:
            node_ignore.append(key)
    print(node_ignore)
    # print(len(all_edges))
    # l = [[i, j] for i, j in all_edges if i in node_ignore or j in node_ignore]
    # print(len(l))
