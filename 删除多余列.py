import pandas as pd


def delete_column(filename, drop_list):
    """
    删除filename的第几列，列数为drop_list中的数字，从0开始

    :param filename:
    :param drop_list:
    :return:
    """
    df = pd.read_csv(filename, encoding="utf-8")
    drop_list2 = df.columns[drop_list]
    df.drop(drop_list2, axis=1, inplace=True, index=True)
    df.to_csv(filename, encoding="utf-8")
    print("删除完毕")


filename = "community_2.csv"
drop_list = [0]
delete_column(filename, drop_list)

