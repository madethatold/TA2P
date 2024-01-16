import random
import numpy as np
import tqdm

from vggFinetuneModel import FineTuneModel
from CIFAR_DataLoader import CifarDataManager
from CIFAR_DataLoader import display_cifar
import numpy as np
from vggNet import Model

data_loader = CifarDataManager()

indexes = np.random.randint(100, size=5)
print('classes:', indexes)
test_images, test_labels = data_loader.test.generateSpecializedData_from_classes(indexes, 2000)
model = FineTuneModel(target_class_id=indexes, prune_ratio=0.8)
# acc = model.test_accuracy(test_images, test_labels)
# print('acc before prune -----------------', acc)
# model.assign_weight()
# acc = model.test_accuracy(test_images, test_labels)
# print('acc after prune -----------------', acc)
#
# for i in tqdm.tqdm(range(50)):
#     train_images, train_labels = data_loader.train.next_batch_without_onehot(500)
#     train_labels = modify_label(train_labels, test_classes=indexes)
#     model.train_model(train_images, train_labels)
#     model.test_accuracy(test_images, test_labels)
