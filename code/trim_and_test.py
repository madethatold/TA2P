import tqdm

from vggTrimmedModel import TrimmedModel
from CIFAR_DataLoader import CifarDataManager
from CIFAR_DataLoader import display_cifar
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

d = CifarDataManager()

# model = TrimmedModel(target_class_id= np.random.randint(50,size=10),
#                       multiPruning=True)


# for _ in range(50):
#     # test_images, test_labels = d.test.next_batch(500)  # 从test数据集中取出200个[x,y]
#
#     test_images, test_labels = d.train.generateSpecializedData(class_id=61,count=100)
#
#     print(np.argmax(test_labels[0]))
#     model.test_accuracy_pretrim(test_images, test_labels)
#
#     model.assign_weight()
#     model.test_accuracy(test_images, test_labels)

# x = []
# acc_before = []
# acc_after = []
#
# for r in tqdm.tqdm(range(100)):
#     pr = r / 100
#     x.append(pr)
#     model = TrimmedModel(target_class_id=[29], multiPruning=False, pr=pr)
#     images, labels = d.test.generateSpecializedData([13], 500)
#     acc_before.append(model.test_accuracy(images, labels))
#     model.assign_weight()
#     acc_after.append(model.test_accuracy(images, labels))
#
# plt.plot(x, acc_before, 'b', linewidth=0.3, label='acc_before')
# plt.plot(x, acc_after, 'r', linewidth=0.3, label='acc_after')
# plt.show()

model = TrimmedModel(target_class_id=[29], multiPruning=False, pr=0)

imgs,labs = d.test.generateSpecializedData_from_classes([23,45,56],500)
acc,_ = model.test_accuracy(imgs,labs)
print(acc)