# from decimal import *
import json
import os
import pickle
import random
import time
from CIFAR_DataLoader import CifarDataManager
import keras
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



class TrimmedModel:
    '''
    This class does:
        1. Load original model graph
        2. Assign new weights to layers
        3. Test Accuracy
    '''

    def __init__(self, target_class_id=[0], multiPruning=False, pr=0.85):

        self.prune_ratio = pr

        self.AllGateVariables = dict()
        self.AllGateVariableValues = list()

        self.graph = tf.Graph()
        self.build_model(self.graph)
        # print("restored the pretrained model......")
        self.restore_model(self.graph)

        self.target_class_id = target_class_id  # assign the trim class id
        self.multiPruning = multiPruning  # ready to prune for one single class or multi classes
        # self.close_sess()

        self.learning_rate = 0.01
        self.epoch = 0

    def train_model(self, input_images, input_labels):
        if self.epoch == 5: self.learning_rate /= 10
        if self.epoch == 50: self.learning_rate /= 10
        if self.epoch == 100: self.learning_rate /= 10
        self.sess.run(self.train_step, feed_dict={
            self.xs: input_images,
            self.ys_true: input_labels,
            self.lr: self.learning_rate,
            self.keep_prob: 1.0,
            self.is_training: False
        })
        self.epoch += 1

    '''
    Find mask class unit
    '''

    def mask_class_unit(self, classid):
        self.test_counter = 0
        theshold = 10
        json_path = "./ClassEncoding/class" + str(classid) + ".json"
        with open(json_path, "r") as f:
            gatesValueDict = json.load(f)
            for idx in range(len(gatesValueDict)):
                layer = gatesValueDict[idx]
                name = layer["name"]
                vec = layer["shape"]
                # process name
                name = name.split('/')[0]
                # process vec
                for i in range(len(vec)):
                    if vec[i] < 10:
                        vec[i] = 0
                    else:
                        vec[i] = 1
                layer["name"] = name
                layer["shape"] = vec

            return gatesValueDict

    '''
    mask by value
    '''

    def mask_unit_by_value_my(self, classid):
        json_path = "./ClassEncoding/class" + str(classid) + ".json"
        with open(json_path, "r") as f:
            gatesValueDict = json.load(f)
            for idx in range(len(gatesValueDict)):
                layer = gatesValueDict[idx]
                name = layer["name"]
                vec = layer["shape"]
                # process name
                name = name.split('/')[0]
                gatesValueDict[idx]["name"] = name
                gatesValueDict[idx]["shape"] = vec

        allGatesValue = []
        for idx in range(len(gatesValueDict)):
            layer = gatesValueDict[idx]
            name = layer["name"]
            vec = layer["shape"]
            allGatesValue += vec

        allGatesValue.sort()
        allGatesValue = allGatesValue[:int(len(allGatesValue) * self.prune_ratio)]
        allGatesValue = set(allGatesValue)

        result = gatesValueDict

        for idx in range(len(result)):
            layer = result[idx]
            name = layer["name"]
            vec = layer["shape"]
            # process name
            name = name.split('/')[0]
            # process vec
            for i in range(len(vec)):
                if vec[i] in allGatesValue:
                    vec[i] = 0
                else:
                    vec[i] = 1

            layer["name"] = name
            layer["shape"] = vec
        return result

    def mask_unit_by_value(self, classid):
        formulizedDict = {}
        json_path = "./ClassEncoding/class" + str(classid) + ".json"

        allGatesValue = []

        with open(json_path, "r") as f:
            gatesValueDict = json.load(f)
            for idx in range(len(gatesValueDict)):
                layer = gatesValueDict[idx]
                name = layer["name"]
                vec = layer["shape"]
                allGatesValue += vec

        allGatesValue.sort()
        allGatesValue = allGatesValue[:int(len(allGatesValue) * self.prune_ratio)]

        allGatesValue = set(allGatesValue)
        with open(json_path, "r") as f:
            gatesValueDict = json.load(f)
            for idx in range(len(gatesValueDict)):
                layer = gatesValueDict[idx]
                name = layer["name"]
                vec = layer["shape"]
                # process name
                name = name.split('/')[0]
                # process vec
                for i in range(len(vec)):
                    if vec[i] in allGatesValue or vec[i] == 0:
                        vec[i] = 0
                    else:
                        vec[i] = 1
                layer["name"] = name
                layer["shape"] = vec

            return gatesValueDict

    '''
    Fine mask class multi, merge multi-class JSONs
    '''

    def mask_class_multi(self):
        theshold = 5
        self.test_counter = 0
        ''' init the dict with class0.json '''
        multiClassGates = self.mask_class_unit(self.target_class_id[0])
        for classid in self.target_class_id:
            if (classid == self.target_class_id[0]):
                continue
            ''' Merge JSONs continuously '''
            json_path = "./ClassEncoding/class" + str(classid) + ".json"
            with open(json_path, "r") as f:
                gatesValueDict = json.load(f)
                for idx in range(len(gatesValueDict)):
                    layer = gatesValueDict[idx]
                    name = layer["name"]
                    vec = layer["shape"]
                    # process name
                    name = name.split('/')[0]
                    # process vec
                    for i in range(len(vec)):
                        if vec[i] < theshold:
                            vec[i] = 0
                        else:
                            vec[i] = 1
                    gatesValueDict[idx]["name"] = name
                    gatesValueDict[idx]["shape"] = vec

                ''' Now we merge gatesValueDict and multiClassGates '''
                for idx1 in range(len(gatesValueDict)):
                    for idx2 in range(len(multiClassGates)):
                        if (gatesValueDict[idx1]["name"] == multiClassGates[idx2]["name"]):
                            tomerge = gatesValueDict[idx1]["shape"]
                            for idx3 in range(len(tomerge)):
                                if (tomerge[idx3] == 1 and multiClassGates[idx2]["shape"][idx3] == 0):
                                    multiClassGates[idx2]["shape"][idx3] = 1
                                    self.test_counter += 1
                                else:
                                    pass
                        else:
                            pass
            print("Furthermore, class ", str(classid), " activate nums of neurons: ", str(self.test_counter))

        return multiClassGates

    def mask_class_multi_by_value(self):
        '''
        Calculate sum of multi-class scalars
        '''
        # print("RUNNING mask_class_multi_by_value.py")
        self.count = 0
        # print("Pruning Ratio: ", self.prune_ratio)
        multiClassGates = list()
        for classid in self.target_class_id:
            '''
            Merge JSONs continuously
            '''
            json_path = "./ClassEncoding/class" + str(classid) + ".json"
            with open(json_path, "r") as f:
                gatesValueDict = json.load(f)
                for idx in range(len(gatesValueDict)):
                    layer = gatesValueDict[idx]
                    name = layer["name"]
                    vec = layer["shape"]
                    # process name
                    name = name.split('/')[0]
                    gatesValueDict[idx]["name"] = name
                    gatesValueDict[idx]["shape"] = vec
                if not multiClassGates:
                    '''
                    Initialize the multiClassGates
                    '''
                    multiClassGates = gatesValueDict
                else:
                    '''
                    Now we merge gatesValueDict and multiClassGates
                    '''
                    for idx1 in range(len(gatesValueDict)):
                        for idx2 in range(len(multiClassGates)):
                            if (gatesValueDict[idx1]["name"] == multiClassGates[idx2]["name"]):
                                tomerge = gatesValueDict[idx1]["shape"]
                                for idx3 in range(len(tomerge)):
                                    multiClassGates[idx2]["shape"][idx3] += tomerge[idx3]
                            else:
                                pass
        '''
        Sort & Mask for multi-class conditions
        '''
        allGatesValue = []
        for idx in range(len(multiClassGates)):
            layer = multiClassGates[idx]
            name = layer["name"]
            vec = layer["shape"]
            allGatesValue += vec

        allGatesValue.sort()
        allGatesValue = allGatesValue[:int(len(allGatesValue) * self.prune_ratio)]
        allGatesValue = set(allGatesValue)

        result = multiClassGates

        for idx in range(len(result)):
            layer = result[idx]
            name = layer["name"]
            vec = layer["shape"]
            # process name
            name = name.split('/')[0]
            # process vec
            for i in range(len(vec)):
                if vec[i] in allGatesValue:
                    vec[i] = 0
                else:
                    vec[i] = 1
                    self.count += 1

            layer["name"] = name
            layer["shape"] = vec
        print("activate nums of neurons:", self.count)
        return result

    '''
    Assign trimmed weight to weight variables
    '''

    def merge(self, fusionRes):
        multiClassGates = fusionRes_to_gate(fusionRes=fusionRes)
        allGatesValue = []
        for idx in range(len(multiClassGates)):
            layer = multiClassGates[idx]
            name = layer["name"]
            vec = layer["shape"]
            allGatesValue += vec

        allGatesValue.sort()
        allGatesValue = allGatesValue[:int(len(allGatesValue) * self.prune_ratio)]
        allGatesValue = set(allGatesValue)

        result = multiClassGates

        for idx in range(len(result)):
            layer = result[idx]
            name = layer["name"]
            vec = layer["shape"]
            # process name
            name = name.split('/')[0]
            # process vec
            for i in range(len(vec)):
                if vec[i] in allGatesValue:
                    vec[i] = 0
                else:
                    vec[i] = 1

            layer["name"] = name
            layer["shape"] = vec

        return result

    def assign_weight(self, fusionRes=None):
        '''
        Encapsulate unit-class pruning and multi-class pruning print("PRUNE FOR CLASS", self.target_class_id)
        '''
        maskDict = []
        if (self.multiPruning == True and len(self.target_class_id) > 1):
            if fusionRes is None:
                maskDict = self.mask_class_multi_by_value()
            else:
                maskDict = self.merge(fusionRes)
        else:
            maskDict = self.mask_unit_by_value(self.target_class_id[0])

        # for li in maskDict:
        #     print(li)

        for tmpLayer in maskDict:
            if (tmpLayer["name"][0] == "C"):  # if the layer is convolutional layer
                with self.graph.as_default():
                    layerNum = tmpLayer["name"].strip("Conv")
                    name = "Conv" + layerNum + "/composite_function/kernel:0"
                    for var in tf.global_variables():
                        if var.name == name:
                            tmpWeights = self.sess.run(var)
                            tmpMask = np.array(tmpLayer["shape"])

                            tmpWeights[:, :, :, tmpMask == 0] = 0
                            assign = tf.assign(var, tmpWeights)
                            self.sess.run(assign)

                            # print(self.sess.run(self.graph.get_tensor_by_name(name))==0)
            # if (tmpLayer["name"][0] == "F"):  # if the layer is fully connected
            #     with self.graph.as_default():
            #         layerNum = tmpLayer["name"].strip("FC")
            #         name_W = "FC" + layerNum + "/W:0"
            #         name_bias = "FC" + layerNum + "/bias:0"
            #         for var in tf.global_variables():
            #             if var.name == name_W:
            #                 tmpWeights = self.sess.run(var)
            #                 tmpMask = np.array(tmpLayer["shape"])
            #
            #                 tmpWeights[:, tmpMask == 0] = 0
            #                 assign = tf.assign(var, tmpWeights)
            #                 self.sess.run(assign)
            #
            #                 # print(self.sess.run(self.graph.get_tensor_by_name(name_W))==0)
            #             if var.name == name_bias:
            #                 tmpBias = self.sess.run(var)
            #                 tmpMask = np.array(tmpLayer["shape"])
            #
            #                 tmpBias[tmpMask == 0] = 0
            #                 assign = tf.assign(var, tmpBias)
            #                 self.sess.run(assign)
            #                 # print(self.sess.run(self.graph.get_tensor_by_name(name_bias))==0)
        # print("assign finished!")
        '''
        Save the model
        '''
        # with self.graph.as_default():
        #     saver = tf.train.Saver(max_to_keep = None)
        #     saver.save(self.sess, 'vggNet/test.ckpt')

    '''
    Test Accuracy
    '''

    def test_accuracy(self, test_images, test_labels):
        start = time.time()
        ys_pred_argmax, ys_true_argmax = self.sess.run(
            [self.ys_pred_argmax, self.ys_true_argmax], feed_dict={
                self.xs: test_images,
                self.ys_true: test_labels,
                self.lr: 0.1,
                self.is_training: False,
                self.keep_prob: 1.0
            })
        end = time.time()

        count = 0
        for i in range(len(ys_pred_argmax)):
            if ys_true_argmax[i] in self.target_class_id:
                count += 1 if ys_pred_argmax[i] == ys_true_argmax[i] else 0
            else:
                count += 1 if ys_pred_argmax[i] not in self.target_class_id else 0

        test_accuracy = count / len(test_labels)
        return test_accuracy, str(end - start)

    '''
    Build VGG Network without Control Gate Lambdas
    '''

    def build_model(self, graph, label_count=100):
        with graph.as_default():
            weight_decay = 5e-4
            self.xs = tf.placeholder("float", shape=[None, 32, 32, 3])
            self.ys_true = tf.placeholder("float", shape=[None, label_count])
            self.lr = tf.placeholder("float", shape=[])
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder("bool", shape=[])

            with tf.variable_scope("Conv1", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(self.xs, 3, 64, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv2", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 64, 64, 3, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv3", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 64, 128, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv4", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 128, 128, 3, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv5", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 128, 256, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv6", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 256, 256, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv7", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 256, 256, 1, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv8", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 256, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv9", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv10", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 1, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv11", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv12", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv13", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 1, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
                current = tf.reshape(current, [-1, 512])
            with tf.variable_scope("FC14", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_fc(current, 512, 4096, self.is_training)
            with tf.variable_scope("FC15", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_fc(current, 4096, 4096, self.is_training)
            with tf.variable_scope("FC16", reuse=tf.AUTO_REUSE):
                Wfc = self.weight_variable_xavier([4096, label_count], name='W')
                bfc = self.bias_variable([label_count])
                ys_pred = tf.matmul(current, Wfc) + bfc

            '''
            Loss Function
            '''
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.ys_true, logits=ys_pred
            ))
            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            total_loss = l2_loss * weight_decay + cross_entropy

            '''
            Optimizer
            '''
            self.train_step = tf.train.MomentumOptimizer(self.lr, 0.9, use_nesterov=True).minimize(total_loss)

            '''
            Accuracy & Top-5 Accuracy
            '''
            self.ys_pred_argmax = tf.argmax(ys_pred, 1)
            self.ys_true_argmax = tf.argmax(self.ys_true, 1)
            correct_prediction = tf.equal(tf.argmax(ys_pred, 1), tf.argmax(self.ys_true, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            top5 = tf.nn.in_top_k(predictions=ys_pred, targets=tf.argmax(self.ys_true, 1), k=5)
            top_5 = tf.reduce_mean(tf.cast(top5, 'float'))

            self.init = tf.global_variables_initializer()

    '''
    Restore the original network weights
    '''

    def restore_model(self, graph):
        # # If GPU is needed
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.sess = tf.Session(graph=graph, config=config)
        # # Else if CPU needed
        # # self.sess = tf.Session(graph = graph)
        # self.sess.run(self.init)
        #
        # with graph.as_default():
        #     saver = tf.train.Saver(max_to_keep=None)
        #
        #     saver.restore(self.sess, "other/vggNet_10/augmentation.ckpt-120")
        #     # print("restored successfully!")

        savedVariable = {}

        # If GPU is needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)
        # Else if CPU needed
        # self.sess = tf.Session(graph = graph)
        self.sess.run(self.init)

        with graph.as_default():
            for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                variable = i
                name = i.name
                if name == 'pl:0':
                    continue
                if name in self.AllGateVariables:
                    continue
                if len(name) >= 8 and name[-11:] == '/Momentum:0':
                    name_prefix = name[:-11]
                    name_prefix += ':0'
                    if name_prefix in self.AllGateVariables:
                        continue
                name = i.name[:-2]
                savedVariable[name] = variable
            saver = tf.train.Saver(savedVariable)
            # saver = tf.train.Saver(max_to_keep = None)
            saver.restore(self.sess, "other/vggNet/augmentation.ckpt-120")
            # print("Restored successfully!")

    '''
    Close Session
    '''

    def close_sess(self):
        self.sess.close()

    '''
    Helper Functions: to build model
    '''

    def gate_variable(self, length, name='gate'):
        initial = tf.constant([1.0] * length)
        v = tf.get_variable(name=name, initializer=initial, trainable=False)
        self.AllGateVariables[v.name] = v
        self.AllGateVariableValues.append(v)
        return v

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer(),
                               trainable=False)

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
                               trainable=False)

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name=name, initializer=initial, trainable=False)

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='VALID')

    def conv2d(self, input, in_features, out_features, kernel_size, with_bias=False):
        W = self.weight_variable_msra([kernel_size, kernel_size, in_features, out_features], name='kernel')
        conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')

        gate = self.gate_variable(out_features)
        conv = tf.multiply(conv, tf.abs(gate))

        if with_bias:
            return conv + self.bias_variable([out_features])
        return conv

    def batch_activ_conv(self, current, in_features, out_features, kernel_size, is_training, keep_prob):
        with tf.variable_scope("composite_function", reuse=tf.AUTO_REUSE):
            current = self.conv2d(current, in_features, out_features, kernel_size)
            current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training,
                                                   updates_collections=None)
            current = tf.nn.relu(current)
            # current = tf.nn.dropout(current, keep_prob)
        return current

    def batch_activ_fc(self, current, in_features, out_features, is_training):
        Wfc = self.weight_variable_xavier([in_features, out_features], name='W')
        bfc = self.bias_variable([out_features])
        current = tf.matmul(current, Wfc) + bfc
        current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
        current = tf.nn.relu(current)
        return current


'''
        {'name': 'FC14/gate:0', 'shape': 4096},
        {'name': 'Conv7/composite_function/gate:0', 'shape': 256},
        {'name': 'Conv12/composite_function/gate:0', 'shape': 512},
        {'name': 'Conv2/composite_function/gate:0', 'shape': 64},
        {'name': 'Conv4/composite_function/gate:0', 'shape': 128},
        {'name': 'Conv1/composite_function/gate:0', 'shape': 64},
        {'name': 'Conv6/composite_function/gate:0', 'shape': 256},
        {'name': 'Conv10/composite_function/gate:0', 'shape': 512},
        {'name': 'Conv9/composite_function/gate:0', 'shape': 512},
        {'name': 'FC15/gate:0', 'shape': 4096},
        {'name': 'Conv13/composite_function/gate:0', 'shape': 512},
        {'name': 'Conv5/composite_function/gate:0', 'shape': 256},
        {'name': 'Conv3/composite_function/gate:0', 'shape': 128},
        {'name': 'Conv11/composite_function/gate:0', 'shape': 512},
        {'name': 'Conv8/composite_function/gate:0', 'shape': 512}
'''

d = CifarDataManager()


def fusionRes_to_gate(fusionRes):
    jsonpath = "./ClassEncoding/class1.json"
    res = []
    count = 0
    with open(jsonpath, 'r') as f:
        gatesValueDict = json.load(f)
        for idx in range(len(gatesValueDict)):
            layer = gatesValueDict[idx]
            name = layer["name"]
            vec = len(layer["shape"])
            # process name
            name = name.split('/')[0]
            gatesValueDict[idx]["name"] = name
            gatesValueDict[idx]["shape"] = fusionRes[count:count + vec]
            count += vec
        res = gatesValueDict
    return res


def get_gatesAll_classId(index):
    # 给label得到通道值的list
    jsonpath = "./ClassEncoding/class" + str(index) + ".json"
    res = []
    with open(jsonpath, 'r') as f:
        gate = json.load(f)
        for ii in range(len(gate)):
            res.extend(gate[ii]['shape'])
    return res
