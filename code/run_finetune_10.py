from vggFinetuneModel_10 import FineTuneModel
from Cifar10_DataLoader import CifarDataManager
import numpy as np

'''
For fine tune model, the data label should be different:
    if the fine tune trimmed model if for classify label:[0]
    the fine tune model data labels would be of 2
    Thus, in this case, we need to change the label correspongdingly then go to train 
'''


def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


def modify_label(labels, test_classes=[0]):
    '''
    labels是test_classes中的话 就是正常的onehot
    不在 那就是返回000000001
    返回的是n*11
    '''
    test_classes.sort()
    tmp_labels = []
    for i in labels:
        if i in test_classes:
            tmp_labels.append(test_classes.index(i))
        else:
            tmp_labels.append(len(test_classes))
    return one_hot(tmp_labels, vals=len(test_classes) + 1)


data_loader = CifarDataManager()

test_images, test_labels = data_loader.test.next_batch_without_onehot(500)  # 这里的labels是n*1，不是n*10

test_labels = modify_label(test_labels, [0, 6, 2, 3])

model = FineTuneModel(target_class_id=[0, 6, 2, 3])
'''targetid的长度直接影响着net的输出有几个'''
model.assign_weight()
model.test_accuracy(test_images, test_labels)

for i in range(10):
    train_images, train_labels = data_loader.train.next_batch_without_onehot(200)
    train_labels = modify_label(train_labels, test_classes=[0, 6, 2, 3])
    model.train_model(train_images, train_labels)
    model.test_accuracy(test_images, test_labels)
