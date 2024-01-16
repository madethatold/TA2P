import os
import random
import time
from vggTrimmedModel_10 import TrimmedModel
from Cifar10_DataLoader import CifarDataManager
import numpy as np
from tqdm import tqdm
import numpy as np
import json
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def manhattan_distance(list1, list2):
    distance = 0
    for i in range(len(list1)):
        distance += abs(list1[i] - list2[i])
    return distance


def cosine_similarity(list1, list2):
    dot_product = 0
    norm_list1 = 0
    norm_list2 = 0
    for i in range(len(list1)):
        dot_product += list1[i] * list2[i]
        norm_list1 += list1[i] ** 2
        norm_list2 += list2[i] ** 2
    return dot_product / (math.sqrt(norm_list1) * math.sqrt(norm_list2))


def merge_para(indexes):
    gate_dict = []
    for classid in indexes:
        jsonpath = "./ClassEncoding_10/class" + str(classid) + ".json"
        with open(jsonpath, 'r') as f:
            gate_tmp = json.load(f)
            gate_dict.append(gate_tmp)

    ce_layer = []
    for i in range(len(gate_dict[0])):
        # 对每一层进行判断
        l_0 = gate_dict[0][i]['shape']
        l_1 = gate_dict[1][i]['shape']
        l_2 = gate_dict[2][i]['shape']
        ce_tmp = (cosine_similarity(l_0, l_1) + cosine_similarity(l_0, l_2) + cosine_similarity(l_1, l_2))
        ce_layer.append(ce_tmp)

    return np.sum(ce_layer)


# lo =[[1,5,8],[0,5,9],[4,5,7],[3,4,5],[6,7,9],[2,3,6]]
# simi = []
# for l in lo:
#     simi.append(merge_para(l))
# print(simi)
i = list(range(6))
x = [[1, 5, 8], [0, 5, 9], [4, 5, 7], [3, 4, 5], [6, 7, 9], [2, 3, 6]]
simi = [44.06490862800995, 44.364546402772724, 44.329285640168344, 44.41962086168222, 44.189053356940406,
        44.47534933012404]
delta = [0.9333333333332984, 1.0000000000000009, 0.866666666666663, 0.100000000000001, 0.40000000000002,
         1.033333333333327]
fig, ax1 = plt.subplots()
font = FontProperties(family='Times New Roman', weight='normal', size=15)
# 创建第二个y轴
ax2 = ax1.twinx()
# 绘制散点图
color_s = '#27408B'
ax2.scatter(i, delta, color=color_s,label='loss of precision(%)', marker='o')
ax2.plot(i, delta, color=color_s)
ax2.set_ylabel('loss of precision(%)',fontproperties=font)
ax2.set_ylim(0, 1.8)

# 绘制柱状图
color = '#87CEFF'
ax1.bar(i, simi, color=color, alpha=0.5,label='similarity')
ax1.set_ylabel('similarity',fontproperties=font)
ax1.set_ylim(44, 45)

# 设置x轴标签
ax1.set_xticks(range(len(i)))
ax1.set_xticklabels(x,fontproperties=font)

ax1.legend(loc='upper left', prop=font)
ax2.legend(loc='upper right', prop=font)


plt.savefig('./plot/fig-6_tmp.png')
# 显示图形
plt.show()
