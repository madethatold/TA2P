from vggTrimmedModel import TrimmedModel
from CIFAR_DataLoader import CifarDataManager
import numpy as np
from tqdm import tqdm
import tensorflow as tf


d = CifarDataManager()
# indexes = [43,52,33,86,48]
# pr = 0.8
# images, labels = d.test.next_batch(1000)
# model = TrimmedModel(target_class_id=indexes, multiPruning=True,pr=pr)
# acc_tmp, time_tmp = model.test_accuracy(images, labels)
# print("pr={},acc before prune:{},time consumed:{}".format(pr, acc_tmp, time_tmp))
# model.assign_weight()
# acc_tmp, time_tmp = model.test_accuracy(images, labels)
# print("pr={},acc after prune:{},time consumed:{}".format(pr, acc_tmp, time_tmp))





pr = 0.6
for _ in range(10):
    indexes = np.random.randint(100, size=5)
    print(indexes)
    model = TrimmedModel(target_class_id=indexes, multiPruning=True, pr=pr)
    images, labels = d.test.next_batch(1000)
    model.assign_weight()
    acc_tmp, time_tmp = model.test_accuracy(images, labels)
    print("pr={},acc after prune:{},time consumed:{}".format(pr, acc_tmp, time_tmp))

