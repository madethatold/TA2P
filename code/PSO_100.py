import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import itertools
import json
import os
from CIFAR_DataLoader import CifarDataManager
from vggTrimmedModel import TrimmedModel
from tqdm import tqdm
import pickle
import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

d = CifarDataManager()


class Particle:
    def __init__(self, num_dimensions):
        self.position = np.random.uniform(0, 1, num_dimensions)
        self.velocity = np.random.uniform(0, 1, num_dimensions)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')


class ParticleSwarmOptimizer:
    def __init__(self, num_particles, num_dimensions, class_indexes):
        self.particles = [Particle(num_dimensions) for _ in range(num_particles)]
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.class_indexes = class_indexes

    def optimize(self, max_iterations):
        for _ in tqdm(range(max_iterations), desc='PSO进度', colour='blue'):
            for particle in tqdm(self.particles, desc='处理Particles的进度', colour='green'):
                fitness = fitness_function(particle.position, self.class_indexes)
                if fitness < particle.best_fitness:
                    particle.best_position = particle.position
                    particle.best_fitness = fitness
                if fitness < self.best_global_fitness:
                    self.best_global_position = particle.position
                    self.best_global_fitness = fitness
                particle.velocity = self.update_velocity(particle)
                particle.position = self.update_position(particle)
                # print("paticle:", particle.position)
                # print("fitness:", fitness)
            print("这一轮particles中的best fitness", self.best_global_fitness)

    def update_velocity(self, particle):
        inertia_weight = 0.5
        cognitive_weight = 1
        social_weight = 2
        inertia_term = inertia_weight * particle.velocity
        cognitive_term = cognitive_weight * np.random.uniform(0, 1) * (particle.best_position - particle.position)
        social_term = social_weight * np.random.uniform(0, 1) * (self.best_global_position - particle.position)
        return inertia_term + cognitive_term + social_term

    def update_position(self, particle):
        return particle.position + particle.velocity


def get_gatesAll_classId(index):
    # 给label得到通道值的list
    jsonpath = "./ClassEncoding/class" + str(index) + ".json"
    res = []
    with open(jsonpath, 'r') as f:
        gate = json.load(f)
        for ii in range(len(gate)):
            res.extend(gate[ii]['shape'])
    return res


def set_params(net, ls):
    params = net.state_dict()
    # 修改参数
    params['fc.weight'] = torch.unsqueeze(torch.Tensor(ls), 0)
    # 将修改后的参数加载回模型
    net.load_state_dict(params)


def loss_(fusion_res, class_list):
    model_trim = TrimmedModel(target_class_id=class_list, multiPruning=True, pr=0.63)

    images, labels = d.train.generateSpecializedData_from_classes(class_list, 1000)
    acc, _ = model_trim.test_accuracy(images, labels)
    res = fusion_res.flatten().tolist()
    model_trim.assign_weight(res)

    for _ in range(100):
        img, lab = d.train.generateSpecializedData_random_from_classes(class_list, 1400)
        model_trim.train_model(img, lab)
    acc_tune, _ = model_trim.test_accuracy(images, labels)
    loss_value = acc - acc_tune
    return loss_value


def fitness_function_old(net, indexes):
    fusion_list = np.array([get_gatesAll_classId(indexes[0])
                               , get_gatesAll_classId(indexes[1])
                               , get_gatesAll_classId(indexes[2])
                            ]).T
    input = torch.from_numpy(fusion_list).float()
    fusion_res = net.fusion(input)  # 此处返回的是numpy
    return loss_(fusion_res, indexes)


def fitness_function(params, class_indexes):
    # net_params = params.reshape(-1, 1)
    set_params(net, params)
    return fitness_function_old(net, class_indexes)  # Your existing fitness function


class CDRP_Fusion(nn.Module):
    def __init__(self):
        super(CDRP_Fusion, self).__init__()
        self.fc = nn.Linear(3, 1, bias=False)
        init.constant_(self.fc.weight, 1.0)
        self.optimizer = optim.SGD(self.parameters(), lr=0.05, momentum=0.9)

    def forward(self, x):
        x = self.fc(x)
        return x

    def fusion(self, x_test):
        with torch.no_grad():
            y_pred = self.forward(x_test)
        return y_pred.numpy()

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
        print('Model saved to:', model_path)

    def load_model(self, model_path=None):
        self.load_state_dict(torch.load(model_path))
        print('Model loaded from:', model_path)


net = CDRP_Fusion()
num_params = sum(p.numel() for p in net.parameters())
"""执行PSO"""
#
# num_dimensions = num_params
#
# lo = [[35, 68, 85],
#       [24, 75, 86],
#       [5, 63, 83],
#       [20, 33, 48],
#       [20, 59, 66],
#       [19, 22, 27],
#       [17, 51, 94]]
#
# for class_indexes in tqdm(lo, colour='white'):
#     print("class_index", class_indexes)
#     pso = ParticleSwarmOptimizer(20, num_dimensions, class_indexes)
#     pso.optimize(max_iterations=15)
#     set_params(net, pso.best_global_position)
#     net.save_model('./for_cifar100/fusion_5/fusionNet' + ('_'.join(str(x) for x in class_indexes)) + '.pth')

'''从目录中得到indexes'''
# def extract_numbers_from_filenames(directory):
#     files = os.listdir(directory)
#     numbers_list = []
#
#     for file in files:
#         # 使用正则表达式提取文件名中的数字部分
#         numbers = re.findall(r'\d+', file)
#         numbers = [int(num) for num in numbers]  # 将提取到的数字转换为整数
#         numbers_list.append(numbers)
#
#     return numbers_list
#
# directory = './other/fusion_3/PSO'  # 替换为你的目录路径
# result = extract_numbers_from_filenames(directory)
# print(result)


'''完成遗传算法后 准备dict 稍后dict内容作图'''

dict_res = {}
samples = [[0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 1, 6], [0, 1, 7], [0, 5, 9], [1, 2, 4], [1, 2, 5], [1, 2, 9], [1, 3, 4],
           [1, 4, 5], [1, 4, 7], [1, 5, 8], [1, 6, 9], [1, 7, 8], [1, 8, 9], [2, 5, 8], [2, 6, 8], [2, 7, 9]]

acc_before = []  # 原始模型
acc_fusion = []  # PSO
acc_no_tune = []  # 未tune
acc_after = []  # sum_gate_fusion
indexes = []  # 类别标签

fusionNet = CDRP_Fusion()
for i in tqdm(range(len(samples))):
    sample_indexes = samples[i]
    print("sample classes:", sample_indexes)

    indexes.append(str(sample_indexes))

    pr = 0.824

    images, labels = d.test.generateSpecializedData_from_classes(sample_indexes, 1000)
    model = TrimmedModel(target_class_id=sample_indexes, multiPruning=True, pr=pr)

    acc_tmp, _ = model.test_accuracy(images, labels)
    print("pr={},acc before prune:{}".format(pr, acc_tmp))
    acc_before.append(acc_tmp)

    path_fusion = './for_cifar100/fusion_10/fusionNet' + ('_'.join(str(x) for x in sample_indexes)) + '.pth'
    fusionNet.load_model(path_fusion)

    fusion_list = np.array([get_gatesAll_classId(sample_indexes[0])
                               , get_gatesAll_classId(sample_indexes[1])
                               , get_gatesAll_classId(sample_indexes[2])
                               , get_gatesAll_classId(sample_indexes[3])
                               , get_gatesAll_classId(sample_indexes[4])
                               , get_gatesAll_classId(sample_indexes[5])
                               , get_gatesAll_classId(sample_indexes[6])
                               , get_gatesAll_classId(sample_indexes[7])
                               , get_gatesAll_classId(sample_indexes[8])
                               , get_gatesAll_classId(sample_indexes[9])
                            ]).T

    input = torch.from_numpy(fusion_list).float()
    fusion_res = fusionNet.fusion(input)
    fusion_res = fusion_res.flatten().tolist()

    model.assign_weight(fusion_res)

    acc_tmp_prune, _ = model.test_accuracy(images, labels)
    print("pr={},acc after fuse:{}".format(pr, acc_tmp_prune))
    acc_no_tune.append(acc_tmp_prune)

    """fine tune"""
    for _ in range(150):
        img, lab = d.test.generateSpecializedData_random_from_classes(sample_indexes, 1000)
        model.train_model(img, lab)

    acc_tmp_tune, _ = model.test_accuracy(images, labels)
    print("pr={},acc after tune:{}".format(pr, acc_tmp_tune))
    acc_fusion.append(acc_tmp_tune)

with open('fusion3_78_res.pkl', 'rb') as f:
    dict_res = pickle.load(f)

dict_res['acc_before'] = acc_before
dict_res['indexes'] = indexes
dict_res['acc_fusion'] = acc_fusion
dict_res['acc_no_tune'] = acc_no_tune
dict_res['acc_after'] = acc_after

with open('fusion3_78_res.pkl', 'wb') as f:
    pickle.dump(dict_res, f)
