import numpy as np
np.random.seed(1337) 

import tensorflow as tf
tf.set_random_seed(2016)

from evaluation import *
from elm import ELM
from keras.utils import np_utils

total_high_pred_train = []
total_high_pred_test = []

def generate_labels(multiTasks, Y, max_t_steps, temporal = False):
    
    dictForLabels = {}
    
    for task, classes, idx in multiTasks:

        if classes == 1:
            if temporal:
                dictForLabels[task] = time_distributed_regression_label(Y[:,idx], max_t_steps)
            else:
                dictForLabels[task] = Y[:,idx]
        else:

            if temporal:
                dictForLabels[task] = time_distributed_label(np_utils.to_categorical(Y[:,idx], classes), max_t_steps)
            else:
                dictForLabels[task] = np_utils.to_categorical(Y[:,idx], classes)
    return dictForLabels

def generate_sample_weight(multiTasks, Y_train, total_class_weights = None, verbose = True):
    num_instances = Y_train.shape[0]
    
    sample_weights_dict = {}
    
    for task, classes, idx in multiTasks:
        sample_weights = np.ones(num_instances)
        labels = Y_train[:,idx]
        labels = labels.astype(int)
            
        if total_class_weights == None:
            print("no sample weights")
        else:    
            task_class_weights = total_class_weights.get(task)
        
            if task_class_weights == None:
                print("automatic setup for sample weights for task: ", task)

                counts = np.bincount(labels)
                class_weights = counts / float(sum(counts))
                for idx in range(num_instances):
                    sample_weights[idx] = sample_weights[idx] - class_weights[labels[idx]]
            else:
                print("manual setup for sample weights: ", task, ",: ", task_class_weights)
                for idx in range(num_instances):
                    sample_weights[idx] = task_class_weights[labels[idx]]

        if verbose:
            print("samples weights for task: ", task, ", :", sample_weights)
            
        sample_weights_dict[task] = sample_weights
        
    return sample_weights_dict
    
def generate_temporal_labels(multiTasks, Y_train, Y_test, Y_valid, max_t_steps):
    dictForLabelsTemporalTest = {}
    dictForLabelsTemporalValid = {}
    dictForLabelsTemporalTrain = {}
    dictForLabelsTest = {}
    dictForLabelsValid = {}
    dictForLabelsTrain = {}

    for task, classes, idx in multiTasks:

        if classes == 1:
            if Y_train != None:
                dictForLabelsTemporalTrain[task] = time_distributed_regression_label(Y_train[:,idx], max_t_steps)
                dictForLabelsTrain[task] = Y_train[:,idx]
            if Y_test != None:
                dictForLabelsTemporalTest[task] = time_distributed_regression_label(Y_test[:,idx], max_t_steps)
                dictForLabelsTest[task] = Y_test[:,idx]
            if Y_valid != None:    
                dictForLabelsTemporalValid[task] = time_distributed_regression_label(Y_valid[:,idx], max_t_steps)
                dictForLabelsValid[task] = Y_valid[:,idx]
        else:
            if Y_train != None:
                dictForLabelsTemporalTrain[task] = time_distributed_label(np_utils.to_categorical(Y_train[:,idx], classes), max_t_steps)
                dictForLabelsTrain[task] = np_utils.to_categorical(Y_train[:,idx], classes)
            if Y_test != None:
                dictForLabelsTemporalTest[task] = time_distributed_label(np_utils.to_categorical(Y_test[:,idx], classes), max_t_steps)
                dictForLabelsTest[task] = np_utils.to_categorical(Y_test[:,idx], classes)
            if Y_valid != None:    
                dictForLabelsTemporalValid[task] = time_distributed_label(np_utils.to_categorical(Y_valid[:,idx], classes), max_t_steps)
                dictForLabelsValid[task] = np_utils.to_categorical(Y_valid[:,idx], classes)
            
    return dictForLabelsTemporalTest, dictForLabelsTemporalValid, dictForLabelsTemporalTrain, dictForLabelsTest, dictForLabelsValid, dictForLabelsTrain


def time_distributed_label(label, max_t_steps):
    nb_class = label.shape[1]
    nb_samples = label.shape[0]
    
    if max_t_steps == None:
        max_t_steps = 1
        
    dictForLabelsTime = np.zeros((nb_samples, max_t_steps, nb_class))
    for i in range(nb_samples):
        for t in range(max_t_steps):
            dictForLabelsTime[i, t, ] = label[i]
    return dictForLabelsTime

def time_distributed_regression_label(label, max_t_steps):
    nb_samples = label.shape[0]
    
    if max_t_steps == None:
        max_t_steps = 1
        
    dictForLabelsTime = np.zeros((nb_samples, max_t_steps, 1))
    for i in range(nb_samples):
        for t in range(max_t_steps):
            dictForLabelsTime[i, t, 0] = label[i]
    return dictForLabelsTime

def high_level_feature_mtl(predictions, threshold = 0.3, stl = False, main_task_id = -1):
    results = []
    total_feat = 0
    total_samples = 0
    if stl == True:
        total_samples = predictions.shape[0]
        result, nb_classes = high_level_feature_task(predictions, threshold)
        return result
    else:
        num_tasks = len(predictions)
        print("number of tasks: ", num_tasks)
        total_samples = predictions[0].shape[0]
        for task_id in range(num_tasks):
            result, nb_classes = high_level_feature_task(predictions[task_id], threshold)
            results.append((result, total_feat, total_feat + nb_classes * 4))
            total_feat = total_feat + nb_classes * 4

    #high level representation of all tasks
    if main_task_id == -1:
        feature_vecs = np.zeros((total_samples, total_feat))
        for utt_id in range(total_samples):
            for task_id in range(num_tasks):
                result = results[task_id]
                feature_vecs[utt_id][result[1]:result[2]] = result[0][utt_id]
    else:
        feature_vecs = results[main_task_id][0]
    return feature_vecs

def high_level_feature_task(predictions, threshold):
    batch_size = predictions.shape[0]
    nb_classes = predictions.shape[2]
    results = np.zeros((batch_size, nb_classes * 4))
    for utt_idx in range(batch_size):
        results[utt_idx] = high_level_feature(predictions[utt_idx], threshold)
    return results, nb_classes

#prediction result for each utturance(time * classes)
def high_level_feature(pred, threshold):
    max_idx = np.argmax(pred, 0)
    min_idx = np.argmin(pred, 0)
    nb_classes = pred.shape[1]
    max_scores = np.zeros((nb_classes))
    min_scores = np.zeros((nb_classes))
    for idx in range(len(max_idx)):
        max_scores[idx] = pred[max_idx[idx]][idx]    
    for idx in range(len(min_idx)):
        min_scores[idx] = pred[min_idx[idx]][idx]
    avg_scores = np.mean(pred, 0)    
    sums = np.sum(pred, 0) + 0.000001
    over = sum(pred > threshold)
    portions = over * avg_scores / sums
    results = np.zeros((nb_classes * 4))
    feats = (max_scores, min_scores, avg_scores, portions)
    for feat_idx in range(4):
        results[feat_idx * nb_classes : (feat_idx + 1) * nb_classes] = feats[feat_idx]
    return results


def elm_predict(model, X_train, X_test, X_valid, multiTasks, unweighted, stl, dictForLabelsTest, dictForLabelsValid, dictForLabelsTrain, hidden_num = 50, main_task_id = -1, elm_save_path = './model/elm.ckpt'):
    sess = tf.Session()

    print('elm high level feature generating')
    pred_train = model.predict([X_train])
    feat_train = high_level_feature_mtl(pred_train, stl = stl, main_task_id = main_task_id)
    print('high level feature dim for train: ', feat_train.shape[1])

    #add total features
    add_high_feature(feat_train, multiTasks, dictForLabelsTrain, total_high_pred_train)

    pred_test = model.predict([X_test])
    feat_test = high_level_feature_mtl(pred_test, stl = stl, main_task_id = main_task_id)
    
    #add total features
    add_high_feature(feat_test, multiTasks, dictForLabelsTest, total_high_pred_test)

    print('high level feature dim for test: ', feat_test.shape[1])
    
    if X_valid != None:
        pred_valid = model.predict([X_valid])
        feat_valid = high_level_feature_mtl(pred_valid, stl = stl, main_task_id = main_task_id)

    scores = []
    for task, classes, idx in multiTasks:
        elm = ELM(sess, feat_train.shape[0], feat_train.shape[1], hidden_num, dictForLabelsTrain[task].shape[1], task = str(task))
        
        print('elm training')
        elm.feed(feat_train, dictForLabelsTrain[task])
        elm.save(elm_save_path + "." + str(task) + ".elm.ckpt")

        print('elm testing')
        labels = dictForLabelsTest[task]
        if unweighted:
            preds = elm.test(feat_test)
            scores.append(unweighted_recall(preds, labels, task))
        else:
            acc = elm.test(feat_test, labels)
            scores.append(acc)

        if X_valid != None:
            print('elm validating')
            labels = dictForLabelsValid[task]
            if unweighted:
                preds = elm.test(feat_valid)
                scores.append(unweighted_recall(preds, labels, task))
            else:
                acc = elm.test(feat_valid, labels)
                scores.append(acc)
    return scores

def elm_load_predict(model, X_test, multiTasks, unweighted, stl, dictForLabelsTest, hidden_num = 50, main_task_id = -1, elm_load_path = './model/elm.ckpt'):
    sess = tf.Session()

    print('elm high level feature generating')
    pred_test = model.predict([X_test])
    feat_test = high_level_feature_mtl(pred_test, stl = stl, main_task_id = main_task_id)

    print('high level feature dim: ', feat_test.shape[1])
    
    scores = []
    for task, classes, idx in multiTasks:
        elm = ELM(sess, feat_test.shape[0], feat_test.shape[1], hidden_num, dictForLabelsTest[task].shape[1])
        
        print('elm loading')
        elm.load(elm_load_path)

        print('elm testing')
        if unweighted:
            labels = dictForLabelsTest[task]
            preds = elm.test(feat_test)
            scores.append(unweighted_recall(preds, labels, task))
        else:
            acc = elm.test(feat_test, labels)
            scores.append(acc)
    return scores

def add_high_feature(feat_train, multiTasks, dictForLabelsTrain, total_data):
    high_feat_label = np.zeros((feat_train.shape[0], feat_train.shape[1] + len(multiTasks)))
    print("high level feat shape: ", feat_train.shape)
    print("high level label shape: ", len(multiTasks))

    high_feat_label[:,0:feat_train.shape[1]] = feat_train

    id = 0
    for task, classes, idx in multiTasks:
        high_feat_label[:,feat_train.shape[1] + id] = np.argmax(dictForLabelsTrain[task],1)
        id = id + 1
    
    total_data.append(high_feat_label)

def total_write_high_feature(file, data):
    f_handle = open(file,'w')
    for line in data:
        np.savetxt(f_handle,line)

    del data[:]
    f_handle.close()