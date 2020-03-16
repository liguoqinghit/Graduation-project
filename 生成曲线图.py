# -*- coding: utf-8 -*-
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

if __name__ == '__main__':
    '''
    根据生成的 Q QM QR 数据生成折线图或散点图
    主要是为了观察各个值得变化
    '''
    # draw_pic_test()
    x = []
    q = []
    qm = []
    qr = []
    index = 1
    with open('node.txt', 'r') as f:
        for i in f:
            x.append(index)
            q.append(float(i.split(' ')[0]) * 1000)
            qm.append(float(i.split(' ')[1]) * 1000)
            qr.append(float(i.split(' ')[2]) * 1000)
            index += 1
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
    # 折线
    plt.plot(x, q, "x-", color='blue', label="q")
    plt.plot(x, qm, "--", color='red', label="qm")
    plt.plot(x, qr, "+-", color='yellow', label="qr")
    # 散点
    # plt.scatter(x, q, marker="x", color='blue', label="q", s=30)
    # plt.scatter(x, qm, marker="x", color='red', label="qm", s=30)
    # plt.scatter(x, qr, marker="x", color='yellow', label="qr", s=30)
    plt.title('Q值的变化')
    plt.xlabel('社团大小的变化')
    plt.ylabel('各个Q的值')
    plt.legend()
    plt.savefig('test_zhexian.png')
    # plt.savefig('test_sandian.png')
    plt.close()
