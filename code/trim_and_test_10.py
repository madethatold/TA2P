import random
from tqdm import tqdm
from vggTrimmedModel_10 import TrimmedModel
from Cifar10_DataLoader import CifarDataManager
import numpy as np
import matplotlib.pyplot as plt



d = CifarDataManager()


x = range(30)
acc_before = []
acc_after = []
acc_tune = []

# 选取任意的三十个组合
indexes = []
all_class = range(10)
for index in range(30):
    tmp = []
    tmp.append(random.sample(all_class, 3))
    indexes.append(tmp)
print(indexes)

for i in tqdm(range(30)):
    sample_indexes = indexes[i][0]
    print("sample classes:", sample_indexes)


    pr = 0.85

    images, labels = d.test.generateSpecializedData_from_classes(sample_indexes, 3000)
    model = TrimmedModel(target_class_id=sample_indexes, multiPruning=True, pr=pr)

    acc_tmp, time_tmp = model.test_accuracy(images, labels)
    print("pr={},acc before prune:{},time consumed:{}".format(pr, acc_tmp, time_tmp))
    acc_before.append(acc_tmp)

    model.assign_weight()

    acc_tmp_prune, time_tmp = model.test_accuracy(images, labels)
    print("pr={},acc after prune:{},time consumed:{}".format(pr, acc_tmp_prune, time_tmp))
    acc_after.append(acc_tmp_prune)

    """fine tune"""
    for _ in range(100):
        img, lab = d.train.generateSpecializedData_random_from_classes(sample_indexes, 600)
        model.train_model(img, lab)

    acc_tmp_tune, time_tmp = model.test_accuracy(images, labels)
    print("pr={},acc after tune:{},time consumed:{}".format(pr, acc_tmp_tune, time_tmp))
    acc_tune.append(acc_tmp_tune)


plt.figure(num=1,figsize=(24,8))
plt.plot(x, acc_before, 'b', linewidth=0.3, label='acc_before')
plt.plot(x, acc_after, 'r', linewidth=0.3, label='acc_after_prune')
plt.plot(x, acc_tune, 'g', linewidth=0.3, label='acc_after_prune&tune')
plt.scatter(x, acc_before, s=8, c='b')
plt.scatter(x, acc_after, s=8, c='r')
plt.scatter(x, acc_tune, s=8, c='g')
plt.legend()
plt.ylabel('acc')
plt.savefig('tmp.png')
plt.show()



#
# indexes = [1,2,3]  # 405
# pr = 0.85
# model = TrimmedModel(target_class_id=indexes, multiPruning=True, pr=pr)
# images, labels = d.test.generateSpecializedData_from_classes(indexes, 3000)
#
# acc_tmp, time_tmp = model.test_accuracy(images, labels)
# print("pr={},acc before prune:{},time consumed:{}".format(pr, acc_tmp, time_tmp))
#
# fusionNet = CDRP_Fusion(3,1)
# path_fusion = './other/Fusion'+('_'.join(str(x) for x in indexes))+'.pth'
# fusionNet.load_model()
#
# fusion_list = np.array([get_gatesAll_classId(indexes[0])
#                                , get_gatesAll_classId(indexes[1])
#                                , get_gatesAll_classId(indexes[2])]).T
# input = torch.from_numpy(fusion_list).float()
#
# res = fusionNet.fusion(input)
# res = res.flatten().tolist()
#
# model.assign_weight(res)
# acc_tmp, time_tmp = model.test_accuracy(images, labels)
# print("pr={},acc after prune:{},time consumed:{}".format(pr, acc_tmp, time_tmp))
#
# for _ in tqdm(range(100)):
#     img, lab = d.train.generateSpecializedData_random_from_classes(indexes, 1200)
#     model.train_model(img, lab)
#
# acc_tmp, time_tmp = model.test_accuracy(images, labels)
# print("pr={},acc after tune:{},time consumed:{}".format(pr, acc_tmp, time_tmp))


"""
601
95 33 85 e
95 33 85


405 
92 57 84
92 33 68 e




405

"""