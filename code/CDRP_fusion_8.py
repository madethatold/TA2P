import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from CIFAR_DataLoader import CifarDataManager
from vggTrimmedModel import TrimmedModel
import json
from tqdm import tqdm
import pickle
import itertools
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

d = CifarDataManager()


def get_gatesAll_classId(index):
    # 给label得到通道值的list
    jsonpath = "./ClassEncoding/class" + str(index) + ".json"
    res = []
    with open(jsonpath, 'r') as f:
        gate = json.load(f)
        for ii in range(len(gate)):
            res.extend(gate[ii]['shape'])
    return res


class CDRP_Fusion(nn.Module):
    def __init__(self):
        super(CDRP_Fusion, self).__init__()
        self.fc = nn.Linear(5, 1, bias=False)
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


def loss_(fusion_res, class_list):
    model_trim = TrimmedModel(target_class_id=class_list, multiPruning=True, pr=0.63)

    images, labels = d.train.generateSpecializedData_from_classes(class_list, 1000)
    acc, _ = model_trim.test_accuracy(images, labels)
    res = fusion_res.flatten().tolist()
    model_trim.assign_weight(res)

    for _ in range(100):
        img, lab = d.train.generateSpecializedData_random_from_classes(class_list, 1000)
        model_trim.train_model(img, lab)

    acc_tune, _ = model_trim.test_accuracy(images, labels)
    loss_value = acc - acc_tune
    return 1.0 / (loss_value + 1e-5)


def fitness_function(net, indexes):
    fusion_list = np.array([get_gatesAll_classId(indexes[0])
                               , get_gatesAll_classId(indexes[1])
                               , get_gatesAll_classId(indexes[2])
                               , get_gatesAll_classId(indexes[3])
                               , get_gatesAll_classId(indexes[4])]).T
    input = torch.from_numpy(fusion_list).float()
    fusion_res = net.fusion(input)  # 此处返回的是numpy
    return loss_(fusion_res, indexes)


def binary_encode(x, bits):
    return format(x, 'b').zfill(bits)


def binary_decode(x):
    return int(x, 2)


def set_params(net, ls):
    params = net.state_dict()
    # 修改参数
    params['fc.weight'] = torch.unsqueeze(torch.Tensor(ls), 0)
    # 将修改后的参数加载回模型
    net.load_state_dict(params)


net = CDRP_Fusion()
num_params = sum(p.numel() for p in net.parameters())


def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(chromosome, mutation_rate):
    # 对一串二进制串进行变异操作
    mutated_chromosome = ''
    for gene in chromosome:
        if random.random() < mutation_rate:
            mutated_chromosome += '0' if gene == '1' else '1'
        else:
            mutated_chromosome += gene
    return mutated_chromosome


def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    probabilities = [fitness / total_fitness for fitness in fitnesses]
    parent1 = random.choices(population, weights=probabilities)[0]
    parent2 = random.choices(population, weights=probabilities)[0]
    return parent1, parent2


# net_params = []
# for p in net.parameters():
#     net_params += list(p.view(-1).detach().numpy())


def generate_population(population_size, bits):
    population = []
    for i in range(population_size):
        chromosome = ''
        for j in range(num_params):
            chromosome += ''.join(random.choice(['0', '1']) for _ in range(bits))
        population.append(chromosome)
    return population


def genetic_algorithm(net, class_indexes, population_size=10, bits_per_param=7, mutation_rate=0.15,
                      max_iterations=20):
    population = generate_population(population_size, bits_per_param)
    for i in tqdm(range(max_iterations), desc='遗传算法的进度'):
        fitnesses = []
        for j in tqdm(range(population_size), desc='一次遗传算法中对population处理的进度'):  # 100
            individual = population[j]
            individual_params = []
            for k in range(num_params):  # 8
                start = k * bits_per_param
                end = (k + 1) * bits_per_param
                param_bits = individual[start:end]
                param_value = binary_decode(param_bits)
                individual_params.append(param_value)
            set_params(net, individual_params)
            fitness = fitness_function(net, indexes=class_indexes)
            fitnesses.append(fitness)
        best_individual = population[fitnesses.index(max(fitnesses))]
        print(f'Iteration {i}: Best individual: {best_individual}, Fitness: {max(fitnesses)}')
        new_population = [best_individual]
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
        population = new_population
    return best_individual


def apply_fusion(individual, net, indexes, bits_per_param=7):
    best_params = []
    for i in range(num_params):
        start = i * bits_per_param
        end = (i + 1) * bits_per_param
        param_bits = individual[start:end]
        param_value = binary_decode(param_bits)
        best_params.append(param_value)
    set_params(net, best_params)
    net.save_model('./other/for_100/fusionNet' + ('_'.join(str(x) for x in indexes)) + '.pth')


'''训练25组类别组合的融合参数'''
# uu = list(itertools.combinations(range(100), 5))
# uu_1 = []
# for c in uu:
#     sorted_u = sorted(c)
#     if sorted_u not in uu_1:
#         uu_1.append(sorted_u)
#
# lo = random.sample(uu_1, 15)
# print(lo)
# lo = [[0, 29, 63, 78, 90], [10, 32, 19, 75, 90], [12, 18, 20, 45, 67], [14, 17, 75, 76, 99],
#       [15, 19, 27, 34, 96], [18, 30, 45, 56, 98], [19, 29, 33, 44, 60], [20, 22, 23, 53, 76],
#       [23, 25,34,65, 75]]
#
# for indexes in tqdm(lo, desc='总进度---', colour='green'):
#     best_individual = genetic_algorithm(net, class_indexes=indexes)
#     apply_fusion(best_individual, net, indexes=indexes)

'''完成遗传算法后 准备dict 稍后dict内容作图'''

dict_res_3 = {}
samples = [[0, 29, 63, 78, 90], [10, 32, 19, 75, 90], [12, 18, 20, 45, 67], [14, 17, 75, 76, 99], [15, 19, 27, 34, 96]]

acc_before = []  # 原始模型
acc_fusion = []  # 我们的方法
indexes = []  # 类别标签
acc_delta = []
fusionNet = CDRP_Fusion()
for i in tqdm(range(len(samples))):
    sample_indexes = samples[i]
    print("sample classes:", sample_indexes)

    indexes.append(str(sample_indexes))

    pr = 0.9

    images, labels = d.test.generateSpecializedData_from_classes(sample_indexes, 300)
    model = TrimmedModel(target_class_id=sample_indexes, multiPruning=True, pr=pr)

    acc_tmp, _ = model.test_accuracy(images, labels)
    print("pr={},acc before prune:{}".format(pr, acc_tmp))
    acc_before.append(acc_tmp)

    path_fusion = './other/for_100/class_5_63/fusionNet' + ('_'.join(str(x) for x in sample_indexes)) + '.pth'
    fusionNet.load_model(path_fusion)

    fusion_list = np.array([get_gatesAll_classId(sample_indexes[0])
                               , get_gatesAll_classId(sample_indexes[1])
                               , get_gatesAll_classId(sample_indexes[2])
                               , get_gatesAll_classId(sample_indexes[3])
                               , get_gatesAll_classId(sample_indexes[4])]).T

    input = torch.from_numpy(fusion_list).float()
    fusion_res = fusionNet.fusion(input)
    fusion_res = fusion_res.flatten().tolist()

    model.assign_weight(fusion_res)

    # acc_tmp_prune, _ = model.test_accuracy(images, labels)
    # print("pr={},acc after prune:{}".format(pr, acc_tmp_prune))
    # acc_after.append(acc_tmp_prune)

    """fine tune"""
    for _ in range(150):
        img, lab = d.test.generateSpecializedData_random_from_classes(sample_indexes, 500)
        model.train_model(img, lab)

    acc_tmp_tune, _ = model.test_accuracy(images, labels)
    print("pr={},acc after tune:{}".format(pr, acc_tmp_tune))
    acc_fusion.append(acc_tmp_tune)
    acc_delta.append(acc_tmp_tune-acc_tmp)
print(acc_delta)
print(np.mean(acc_delta))
# #
# # with open('dict_res_7.pkl', 'rb') as f:
# #     dict_res_7 = pickle.load(f)
#
# dict_res_3['acc_before'] = acc_before
# dict_res_3['indexes'] = indexes
# dict_res_3['acc_fusion'] = acc_fusion
#
# with open('dict_res_5_cifar100.pkl', 'wb') as f:
#     pickle.dump(dict_res_3, f)

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
# directory = './other/for_100/class_5_63/'  # 替换为你的目录路径
# result = extract_numbers_from_filenames(directory)
# print(result)
