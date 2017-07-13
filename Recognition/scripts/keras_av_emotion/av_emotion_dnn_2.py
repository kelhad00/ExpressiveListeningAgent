from __future__ import print_function
import numpy as np
np.random.seed(1337) 

import tensorflow as tf
tf.set_random_seed(2016)

import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, LSTM, GRU, Dropout, Activation, TimeDistributed, merge
from keras.engine.topology import Merge
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
from keras import backend as K
from keras.layers.core import Highway
from keras.layers.core import Permute
from keras.layers import Input, Convolution3D,Convolution2D,Convolution1D, MaxPooling3D,MaxPooling2D, MaxPooling1D, GlobalAveragePooling1D, Flatten, ZeroPadding2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger,TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
import keras.utils.io_utils
import argparse
import h5py
import sys
from sklearn.metrics import f1_score,recall_score,confusion_matrix
from evaluation import *
from high_level import *
from conv_highway import Conv2DHighway
from conv1d_highway import Conv1DHighway
from conv3d_highway import Conv3DHighway
from keras.models import load_model
from keras.utils.visualize_util import plot

def load_premodel(model, model_path):
    if model_path != None:
        print("Pre-trained model:", model_path)
        premodel = load_model(model_path)
        premodel.summary()
    
        premodel.save_weights('./temp.w.h5')
    
        print("loading weights........")
        model.load_weights('./temp.w.h5', by_name=True)
    
        del premodel


def utt_feature_ext(utt_feat_ext_layer, input):
    return utt_feat_ext_layer.predict(input)

def total_write_utt_feature(file, total_utt_features):
    f_handle = open(file,'w')
    
    np.savetxt(f_handle, total_utt_features)

    f_handle.close()

def compose_utt_feat(feature, multiTasks, labels, onehotvector = False):
    high_feat_label = np.zeros((feature.shape[0], feature.shape[1] + len(multiTasks)))
    print("high level feat shape: ", feature.shape)
    print("high level label shape: ", labels.shape, "multitasks: ", len(multiTasks), ": ", str(multiTasks))

    high_feat_label[:,0:feature.shape[1]] = feature

    id = 0
    for task, classes, idx in multiTasks:
        if onehotvector:
            high_feat_label[:,feature.shape[1] + id] = np.argmax(labels[:,idx],1)
        else:
            high_feat_label[:,feature.shape[1] + id] = labels[:,idx]
        id = id + 1

    return high_feat_label

def load_data_in_range(data_path, start, end, f):
    X_audio = np.array(keras.utils.io_utils.HDF5Matrix(data_path, "a_feat", start, end))
    X_video = np.array(keras.utils.io_utils.HDF5Matrix(data_path, "v_feat", start, end))
    Y = np.array(keras.utils.io_utils.HDF5Matrix(data_path, "label", start, end))
    return X_audio, X_video, Y

def compose_idx(args_train_idx, args_test_idx, args_valid_idx, args_ignore_idx, args_adopt_idx, args_kf_idx):
    train_idx = []
    test_idx = []
    valid_idx = []
    ignore_idx = []
    adopt_idx = []
    kf_idx = []
    if args_train_idx:
        if ',' in args_train_idx:
            train_idx = args_train_idx.split(',')
        elif ':' in args_train_idx:
            indice = args_train_idx.split(':')
            for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                train_idx.append(idx)
        else:
            train_idx = args_train_idx.split(",")

    if args_test_idx:
        if ',' in args_test_idx:
            test_idx = args_test_idx.split(',')
        elif ':' in args_test_idx:
            indice = args_test_idx.split(':')
            for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                test_idx.append(idx)
        else:
            test_idx = args_test_idx.split(",")

    if args_ignore_idx:
        if ',' in args_ignore_idx:
            ignore_idx = args_ignore_idx.split(',')
        elif ':' in args_ignore_idx:
            indice = args_ignore_idx.split(':')
            for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                ignore_idx.append(idx)
        else:
            ignore_idx = args_ignore_idx.split(",")

    if args_valid_idx:
        if ',' in args_valid_idx:
            valid_idx = args_valid_idx.split(',')
        elif ':' in args_valid_idx:
            indice = args_valid_idx.split(':')
            for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                valid_idx.append(idx)
        else:
            valid_idx = args_valid_idx.split(",")
    if args_adopt_idx:
            if ',' in args_adopt_idx:
                adopt_idx = args_adopt_idx.split(',')
            elif ':' in args_adopt_idx:
                indice = args_adopt_idx.split(':')
                for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                    adopt_idx.append(idx)
            else:
                adopt_idx = args_adopt_idx.split(",")
    if args_kf_idx:
        kf_idx = args_kf_idx.split(",")
    kf_idx = set(kf_idx)

    return train_idx, test_idx, valid_idx, ignore_idx, adopt_idx, kf_idx

def train_adopt_evaluate(model, multiTasks, X_train, X_test, X_valid, X_adopt, Y_train, Y_test, Y_valid, Y_adopt, a_max_t_steps, callbacks, utt_level = False, stl = False, unweighted = True, model_save_path = './model/model', evaluation = True, r_valid = 0.0, epochs = 10, batch_size = 128, class_weights = None, reg = False):
    
    if reg:
        sample_weights = None
    else:
        if class_weights != None:
            sample_weights = generate_sample_weight(multiTasks, Y_train, class_weights)
            print("sample weight:, ", sample_weights)
        else:
            sample_weights = None
    
    print("a_max_t_steps: ", a_max_t_steps)
    
    main_model_path = model_save_path + ".h5"

    dictForLabelsTemporalTest, dictForLabelsTemporalValid, dictForLabelsTemporalTrain, dictForLabelsTest, dictForLabelsValid, dictForLabelsTrain = generate_temporal_labels(multiTasks, Y_train, Y_test, Y_valid, a_max_t_steps)
    
    if len(X_adopt[0]) != 0:
        dictForLabelsTemporalAdopt = generate_labels(multiTasks, Y_adopt, a_max_t_steps, True)
        dictForLabelsAdopt = generate_labels(multiTasks, Y_adopt, a_max_t_steps, False)
        
    if utt_level:
        if r_valid == 0.0:
            model.fit(X_train, dictForLabelsTrain, batch_size = batch_size, nb_epoch=epochs, validation_data=(X_valid, dictForLabelsValid), callbacks=callbacks, sample_weight = sample_weights)
        else:
            print("batch_size: ", batch_size)
            print("epochs: ", epochs)
            print("r_valid: ", r_valid)
            model.fit(X_train, dictForLabelsTrain, batch_size = batch_size, nb_epoch=epochs, validation_split = r_valid, callbacks=callbacks, sample_weight = sample_weights)
        
        if len(X_adopt[0]) != 0:
            model.fit(X_adopt, dictForLabelsAdopt, batch_size = batch_size, nb_epoch=epochs, validation_split = r_valid, callbacks=callbacks, sample_weight = sample_weights)
            
        if evaluation:
            predictions = model.predict(X_test)
            if reg:
                scores = regression_task(predictions, dictForLabelsTest, multiTasks)
            elif unweighted:
                scores = unweighted_recall_task(predictions, dictForLabelsTest, multiTasks)
            
            scores = model.evaluate(X_test, dictForLabelsTest, verbose=0)
            print("evaluated matrix: ", scores)
    else:
        print("temporal output is not supported yet")
        
    if model_save_path != '':
        model.save(main_model_path)

    if evaluation:
        print(scores)
        return scores


parser = argparse.ArgumentParser()

#training params
parser.add_argument("-b", "--batch", dest= 'batch', type=int, help="batch size", default=128)
parser.add_argument("-n_sub_b", "--nb_sub_batch", dest= 'nb_sub_batch', type=int, help="nb_sub_batch", default=10)
parser.add_argument("-e", "--epoch", dest= 'epoch', type=int, help="number of epoch", default=50)
parser.add_argument("-p", "--patience", dest= 'patience', type=int, help="patience size", default=5)
parser.add_argument("-d", "--dropout", dest= 'dropout', type=float, help="dropout", default=0.5)
parser.add_argument("-lr", "--learing_rate", dest= 'learingRate', type=float, help="learingRate", default=0.001)
parser.add_argument("-l2", "--l2reg", dest= 'l2reg', type=float, help="l2reg", default=0.01)
parser.add_argument("-cw", "--class_w", dest= 'class_w', type=str, help="class weights (e.g. arousal:0.5:1.0:0.5,valence:0.5:1.0:0.1)")

#dnn
parser.add_argument("-nn", "--node_size", dest= 'node_size', type=int, help="DNN node_size", default=128)
parser.add_argument("-dnn_depth", "--dnn_depth", dest= 'dnn_depth', type=int, help="depth of convolutional layers", default = 3)
parser.add_argument("-f_dnn_depth", "--f_dnn_depth", dest= 'f_dnn_depth', type=int, help="depth of feature dnn", default = 2)
parser.add_argument("-p_dnn_depth", "--p_dnn_depth", dest= 'p_dnn_depth', type=int, help="depth of post dnn", default = 2)
parser.add_argument("-av_dnn_depth", "--av_dnn_depth", dest= 'av_dnn_depth', type=int, help="depth of audio/visual dnn", default = 2)
parser.add_argument("-av_nn", "--av_node_size", dest= 'av_node_size', type=int, help="DNN av_node_size", default=512)

#convolution
parser.add_argument("-n_row", "--nb_row", dest= 'nb_row', type=str, help="length of row for 2d convolution", default="10,5")
parser.add_argument("-n_col", "--nb_col", dest= 'nb_col', type=str, help="length of column for 2d convolution", default="40,20")
parser.add_argument("-n_time", "--nb_time", dest= 'nb_time', type=str, help="nb_time for 3d convolution", default="40,20")
parser.add_argument("-l_filter", "--len_filter", dest= 'len_filter', type=str, help="filter length for 1d convolution", default="100,80")
parser.add_argument("-n_filter", "--nb_filter", dest= 'nb_filter', type=str, help="nb_filter", default="40,20")
parser.add_argument("-stride", "--stride", dest= 'sub_sample', type=int, help="stride (how many segment a filter shifts for each time", default=1)
parser.add_argument("-pool", "--pool", dest= 'l_pool', type=str, help="pool", default="2,2")
parser.add_argument("-pool_t", "--pool_t", dest= 'pool_t', type=str, help="pool", default="2")
parser.add_argument("-pool_r", "--pool_r", dest= 'pool_r', type=str, help="pool", default="2")
parser.add_argument("-pool_c", "--pool_c", dest= 'pool_c', type=str, help="pool", default="2")

#lstm
parser.add_argument("-c_len", "--context_len", dest= 'context_len', type=int, help="context_len", default=5)
parser.add_argument("-nb_sample", "--nb_total_sample", dest= 'nb_total_sample', type=int, help="nb_total_sample")
parser.add_argument("-cs", "--cell_size", dest= 'cell_size', type=int, help="LSTM cell_size", default=256)
parser.add_argument("-a_t_max", "--a_t_max", dest= 'a_t_max', type=int, help="max length of time sequence")
parser.add_argument("-v_t_max", "--v_t_max", dest= 'v_t_max', type=int, help="max length of time sequence")

parser.add_argument("-w_feat_layer", "--feature_ext_name", dest= 'feature_ext_name', type=str, help="write feat layer")

#image
parser.add_argument("-img_size", "--img_size", dest= 'img_size', type=int, help="pixels of image", default = 48)
parser.add_argument("-v_cs", "--video_cell_size", dest= 'video_cell_size', type=int, help="cell size of video RNN", default = 128)

#cv 
parser.add_argument("-dt", "--data", dest= 'data', type=str, help="data")
parser.add_argument("-kf", "--k_fold", dest= 'k_fold', type=int, help="random split k_fold")
parser.add_argument("-n_cc", "--n_cc", dest= 'cc', type=str, help="cc (0,1,2,3,4)")
parser.add_argument("-test_idx", "--test_idx", dest= 'test_idx', type=str, help="(0,1,2,3,4)")
parser.add_argument("-train_idx", "--train_idx", dest= 'train_idx', type=str, help="(0,1,2,3,4)")
parser.add_argument("-valid_idx", "--valid_idx", dest= 'valid_idx', type=str, help="(0,1,2,3,4)")
parser.add_argument("-ignore_idx", "--ignore_idx", dest= 'ignore_idx', type=str, help="(0,1,2,3,4)")
parser.add_argument("-adopt_idx", "--adopt_idx", dest= 'adopt_idx', type=str, help="Use train_idx together. Train data is first used then, adopted to this data(0,1,2,3,4)")
parser.add_argument("-kf_idx", "--kf_idx", dest= 'kf_idx', type=str, help="(0,1,2,3,4)")
parser.add_argument("-r_valid", "--r_valid", dest= 'r_valid', type=float, help="validation data rate from training", default=0.0)
parser.add_argument("-mt", "--multitasks", dest= 'multitasks', type=str, help="multi-tasks (name:classes:idx:(cost_function):(weight)", default = 'acted:2:0::,arousal:2:1::')
parser.add_argument("-ot", "--output_file", dest= 'output_file', type=str, help="output.txt")
parser.add_argument("-sm", "--save_model", dest= 'save_model', type=str, help="save model", default='./model/model')

#pretrained
parser.add_argument("-alm", "--a_load_model", dest= 'a_load_model', type=str, help="load pre-trained model for audio")
parser.add_argument("-vlm", "--v_load_model", dest= 'v_load_model', type=str, help="load pre-trained model for video")
parser.add_argument("-frozen", "--frozen", dest= 'frozen', type=str, help="(0,1,2,3,4)")
parser.add_argument("-unloaded", "--unloaded", dest= 'unloaded', type=str, help="(0,1,2,3,4)")

parser.add_argument("-log", "--log_file", dest= 'log_file', type=str, help="log file", default='./output/log.txt')
parser.add_argument("-w_feat", "--w_feat", dest= 'w_feat', type=str, help="write feat file")

parser.add_argument("--conv", help="frame level convolutional network for 2d or 1d",
                    action="store_true")
parser.add_argument("--conv_3d", help="frame level convolutional network for 3d",
                    action="store_true")
parser.add_argument("--r_conv_3d", help="frame level convolutional network for 3d",
                    action="store_true")
parser.add_argument("--conv_hw_3d", help="frame level convolutional network for 3d",
                    action="store_true")

parser.add_argument("--r_conv", help="frame level residual convolutional network",
                    action="store_true")

parser.add_argument("--f_dnn", help="frame level dnn requiring elm, otherwise it calculates frame-level performances",
                    action="store_true")
parser.add_argument("--f_highway", help="frame level highway network requiring elm, otherwise it calculates frame-level performances",
                    action="store_true")
parser.add_argument("--f_conv_highway", help="frame level 2d convolutional highway network requiring elm, otherwise it calculates frame-level performances",
                    action="store_true")
parser.add_argument("--f_residual", help="frame level residual network requiring elm, otherwise it calculates frame-level performances",
                    action="store_true")

parser.add_argument("--f_lstm", help="frame level lstm requiring elm, otherwise it calculates frame-level performances",
                    action="store_true")
parser.add_argument("--u_lstm", help="utterance level lstm",
                    action="store_true")
parser.add_argument("--u_dnn", help="utterance level dnn after u_lstm",
                    action="store_true")
parser.add_argument("--u_hw", help="utterance level highway after u_lstm",
                    action="store_true")
parser.add_argument("--u_residual", help="utterance level residual after u_lstm",
                    action="store_true")
parser.add_argument("--g_pool", help="global pooling for temporal features",
                    action="store_true")

parser.add_argument("--v_u_lstm", help="vidual utterance level lstm",
                    action="store_true")
parser.add_argument("--v_u_gru", help="vidual utterance level lstm",
                    action="store_true")
parser.add_argument("--g_pool_rnn", help="utterance level global average pooled rnn",
                    action="store_true")

parser.add_argument("--headerless", help="headerless in feature file?",
                    action="store_true")
parser.add_argument("--default", help="default training",
                    action="store_true")
parser.add_argument("--log_append", help="append log or not",
                    action="store_true")
parser.add_argument("--unweighted", help="unweighted evaluation",
                    action="store_true")
parser.add_argument("--tb", help="tensorboard",
                    action="store_true")

parser.add_argument("--reg", help="regression evaluation",
                    action="store_true")
                    
args = parser.parse_args()

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
    

patience = args.patience
batch_size = args.batch
epochs = args.epoch
nb_sub_batch = args.nb_sub_batch
cell_size = args.cell_size
node_size = args.node_size
kfold = args.k_fold
l2_reg = args.l2reg
dropout = args.dropout
learing_rate = args.learingRate
tasks = args.multitasks 
   
output_file = args.output_file
log_file = args.log_file
data_path = args.data
# Convolution
nb_time = [] 
for time in args.nb_time.split(','):
    nb_time.append(int(time))

nb_row = [] 
for row in args.nb_row.split(','):
    nb_row.append(int(row))
    
nb_col = [] 
for col in args.nb_col.split(','):
    nb_col.append(int(col))
    
nb_filter = [] 
for filter in args.nb_filter.split(','):
    nb_filter.append(int(filter))

len_filter = []
for length in args.len_filter.split(','):
    len_filter.append(int(length))

l_pool = []
for length in args.l_pool.split(','):
    l_pool.append(int(length))

pool_t = []
for length in args.pool_t.split(','):
    pool_t.append(int(length))


pool_r = []
for length in args.pool_r.split(','):
    pool_r.append(int(length))

pool_c = []
for length in args.pool_c.split(','):
    pool_c.append(int(length))

if args.conv or args.r_conv:
    if len(nb_row) == len(nb_col) and len(nb_col) == len(nb_filter) and len(nb_filter) == args.dnn_depth:
        print("correct setup for convolution")
    else:
        print("wrong setup for convolution, number of layers should match the number of setups for column, row, filters")
        print("depth: ", args.dnn_depth, "row: ", len(nb_row), "col: ", len(nb_col), "filter: ", len(nb_filter))
        exit()
    
sub_sample = args.sub_sample

save_model = args.save_model
    
n_cc = []
if args.cc:
    if ',' in args.cc:
        n_cc = args.cc.split(',')
    elif ':' in args.cc:
        indice = args.cc.split(':')
        for idx in range(int(indice[0]), int(indice[1]), +1):
            n_cc.append(idx)
    else:
        n_cc = args.cc.split(',')
    
    print('total cv: ', len(n_cc))


r_valid = args.r_valid

#utt level features
total_utt_features = []

#compose idx
train_idx, test_idx, valid_idx, ignore_idx, adopt_idx, kf_idx = compose_idx(args.train_idx, args.test_idx, args.valid_idx, args.ignore_idx, args.adopt_idx, args.kf_idx)

large_corpus_mode = False
if args.nb_total_sample:
    print("very large corpus mode")
    large_corpus_mode = True
    a_train_csv, v_train_csv, train_lab = load_data_in_range(data_path, 0, 1)
else:
    with h5py.File(data_path,'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = hf.get('a_feat')
        a_train_csv = np.array(data)
        data = hf.get('v_feat')
        v_train_csv = np.array(data)
        data = hf.get('label')
        train_lab = np.array(data)
        print('Shape of the array audio feat: ', a_train_csv.shape)
        print('Shape of the array video feat: ', v_train_csv.shape)
        print('Shape of the array lab: ', train_lab.shape)
        if len(n_cc) or len(test_idx) > 0:
            start_indice = np.array(hf.get('start_indice'))
            end_indice = np.array(hf.get('end_indice'))
            print('Shape of the indice for start: ', start_indice.shape)

input_type = "1d"

#2d input
if len(a_train_csv.shape) == 5:
    input_dim = a_train_csv.shape[4]
    context_len = a_train_csv.shape[3]
    if args.conv_3d or args.conv_hw_3d or args.r_conv_3d:
        input_type = "3d"
    else:    
        input_type = "2d"
else:#1d input
    input_dim = a_train_csv.shape[2]

a_max_t_steps = a_train_csv.shape[1]

nameAndClasses = tasks.split(',')
multiTasks = []
dictForCost = {}
dictForWeight = {}
dictForEval = {}

for task in nameAndClasses:
    params = task.split(':')
    
    name = "av_" + params[0]
        
    multiTasks.append((name, int(params[1]), int(params[2])))
    if int(params[1]) == 1:    #regression problem
        dictForEval[name] = 'mean_squared_error'
    else:
        dictForEval[name] = 'accuracy'
    if params[3] != '':
        dictForCost[name] = params[3]
    else:
        dictForCost[name] = 'categorical_crossentropy'  
    if params[4] != '':
        dictForWeight[name] = float(params[4])
    else:
        dictForWeight[name] = 1.

if len(multiTasks) > 1:
    stl = False
    print('MTL')       
else:
    stl = True
    print('STL')

#class weights

if args.class_w:
    class_weights = {}
    if(args.class_w.startswith("auto") == True):
        print("class weights will be automatically set.")
    else:
        for weights in args.class_w.split(","):
            temp = weights.split(":")
            task = temp[0]
            w_s = np.zeros(len(temp) - 1)
            for idx in range(1, len(temp)):
                w_s[idx - 1] = float(temp[idx])
            class_weights[task] = w_s

        print("class manual weights: ", class_weights)
    
else:
    class_weights = None
    print("no class weights")
    
#callbacks
callbacks = []
callbacks.append(EarlyStopping(monitor='val_loss', patience=patience))
if args.log_file:
    csv_logger = CSVLogger(log_file + ".csv", separator='\t')
    callbacks.append(csv_logger)

if args.tb:
    callbacks.append(TensorBoard(log_dir='./logs', histogram_freq=2, write_graph=True, write_images=True))

adam = Adam(lr=learing_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=0.5)

print('Creating Model')

#audio input 
if input_type == "3d":
    inputs = Input(shape=(1, args.a_t_max, context_len, input_dim),name='audio_feat')
elif input_type == "2d":
    inputs = Input(shape=(args.a_t_max, 1, context_len, input_dim),name='audio_feat')
elif input_type == "1d":
    inputs = Input(shape=(args.a_t_max, input_dim),name='audio_feat')

#batch normalisation
batchNorm = BatchNormalization(mode = 2)

#utterance level modelling
if args.f_dnn or args.f_lstm or args.f_highway or args.f_residual:
    utt_model = False
else:
    utt_model = True

#2d or 1d
if input_type == "2d" or input_type == "1d":
    t_inputs = TimeDistributed(batchNorm)(inputs)

if input_type == "3d":
    t_inputs = inputs
    if args.conv_3d:
        for d in range(args.dnn_depth):
            t_inputs = Convolution3D(W_regularizer=l2(l2_reg),nb_filter=nb_filter[d],dim_ordering='th',
                                    kernel_dim1=nb_time[d],
                                    kernel_dim2=nb_row[d],
                                    kernel_dim3=nb_col[d],
                                    border_mode='same',
                                    activation='relu',
                                    subsample=(sub_sample,sub_sample,sub_sample), name = 'conv-' + str(d))(t_inputs)
            t_inputs = MaxPooling3D(pool_size = (pool_t[d], pool_r[d], pool_c[d]), dim_ordering = 'th', name = 'pool-' + str(d))(t_inputs)
    elif args.conv_hw_3d:
        for d in range(args.dnn_depth):
            t_inputs = Conv3DHighway(W_regularizer=l2(l2_reg),nb_filter=nb_filter[d],dim_ordering='th',
                                    kernel_dim1=nb_time[d],
                                    kernel_dim2=nb_row[d],
                                    kernel_dim3=nb_col[d],
                                    border_mode='same',
                                    activation='relu',
                                    subsample=(sub_sample,sub_sample,sub_sample), name = 'conv-hw-' + str(d))(t_inputs)
            t_inputs = MaxPooling3D(pool_size = (pool_t[d], pool_r[d], pool_c[d]), dim_ordering = 'th', name = 'pool-' + str(d))(t_inputs)
    elif args.r_conv_3d:
        t_inputs = Convolution3D(W_regularizer=l2(l2_reg),nb_filter=nb_filter[0],dim_ordering='th',
                                    kernel_dim1=nb_time[0],
                                    kernel_dim2=nb_row[0],
                                    kernel_dim3=nb_col[0],
                                    border_mode='same',
                                    activation='relu',
                                    subsample=(sub_sample,sub_sample,sub_sample), name = 'conv-' + str(0))(t_inputs)
        t_inputs = MaxPooling3D(pool_size = (pool_t[0], pool_r[0], pool_c[0]), dim_ordering = 'th', name = 'pool-' + str(0))(t_inputs)
        o_inputs = t_inputs
        for d in range(1, args.dnn_depth):
            t_inputs = Convolution3D(W_regularizer=l2(l2_reg),nb_filter=nb_filter[d],dim_ordering='th',
                                    kernel_dim1=nb_time[d],
                                    kernel_dim2=nb_row[d],
                                    kernel_dim3=nb_col[d],
                                    border_mode='same',
                                    activation='relu',
                                    subsample=(sub_sample,sub_sample,sub_sample), name = 'r-conv-' + str(d))(t_inputs)
            #residual merge and pooling at the end
            t_inputs = merge([o_inputs, t_inputs], mode ='sum', name="c_merge_" + str(d))
            t_inputs = MaxPooling3D(pool_size = (pool_t[d], pool_r[d], pool_c[d]), dim_ordering = 'th', name = 'pool-' + str(d))(t_inputs)
            o_inputs = t_inputs
    if args.u_lstm or args.f_lstm or args.f_dnn or args.f_residual or args.f_highway:
        t_inputs = Permute((2,1,3,4))(t_inputs)
        t_inputs = TimeDistributed(Flatten())(t_inputs)
    else:    
        t_inputs = Flatten()(t_inputs)

elif input_type == "2d":
    if args.conv:
        for d in range(args.dnn_depth):
            t_inputs = TimeDistributed(Convolution2D(W_regularizer=l2(l2_reg),nb_filter=nb_filter[d],dim_ordering='th',
                                nb_row=nb_row[d],
                                nb_col=nb_col[d],
                                border_mode='same',
                                activation='relu',
                                subsample=(sub_sample,sub_sample)), name = 'conv-' + str(d))(t_inputs)
            t_inputs = TimeDistributed(MaxPooling2D(pool_size = (pool_r[d], pool_c[d]), dim_ordering = 'th'), name = 'pool-' + str(d))(t_inputs)
    elif args.r_conv:
        
        t_inputs = TimeDistributed(Convolution2D(W_regularizer=l2(l2_reg),nb_filter=nb_filter[0],dim_ordering='th',
                                nb_row=nb_row[0],
                                nb_col=nb_col[0],
                                border_mode='same',
                                activation='relu',
                                subsample=(sub_sample,sub_sample)), name = 'r-conv-' + str(0))(t_inputs)
        t_inputs = TimeDistributed(MaxPooling2D(pool_size = (pool_r[0], pool_c[0]), dim_ordering = 'th'), name = 'pool-' + str(0))(t_inputs)
        o_inputs = t_inputs
        for d in range(1, args.dnn_depth):
            t_inputs = TimeDistributed(Convolution2D(W_regularizer=l2(l2_reg),nb_filter=nb_filter[d],dim_ordering='th',
                                nb_row=nb_row[d],
                                nb_col=nb_col[d],
                                border_mode='same',
                                activation='relu',
                                subsample=(sub_sample,sub_sample)), name = 'r-conv-' + str(d))(t_inputs)
            #residual merge and pooling at the end
            t_inputs = merge([o_inputs, t_inputs], mode ='sum', name="c_merge_" + str(d))
            t_inputs = TimeDistributed(MaxPooling2D(pool_size = (pool_r[d], pool_c[d]), dim_ordering = 'th'), name = 'pool-' + str(d))(t_inputs)
            o_inputs = t_inputs
    elif args.f_conv_highway:
        for d in range(args.dnn_depth):
            t_inputs = TimeDistributed(Conv2DHighway(W_regularizer=l2(l2_reg),nb_filter=nb_filter[d],dim_ordering='th',
                                nb_row=nb_row[d],
                                nb_col=nb_col[d],
                                border_mode='same',
                                activation='relu'), name = 'conv-hw-' + str(d))(t_inputs)
            t_inputs = TimeDistributed(MaxPooling2D(pool_size = (pool_r[d], pool_c[d]), dim_ordering = 'th'), name = 'pool-' + str(d))(t_inputs)

    #for all layers, flatten is necessary for 2d inputs
    t_inputs = TimeDistributed(Flatten())(t_inputs)
elif input_type == "1d":#ID convolution
    for d in range(args.dnn_depth):
        if args.conv:
            t_inputs = Convolution1D(nb_filter=nb_filter[d],
                        filter_length=len_filter[d],
                        border_mode='valid',
                        activation='relu',
                        subsample_length=sub_sample)(t_inputs)
            t_inputs = MaxPooling1D(pool_length = l_pool[d])(t_inputs)
        elif args.f_conv_highway:
            t_inputs = Conv1DHighway(nb_filter=nb_filter[d],
                        filter_length=len_filter[d],
                        border_mode='valid',
                        activation='relu',
                        subsample_length=sub_sample)(t_inputs)
            t_inputs = MaxPooling1D(pool_length = l_pool[d])(t_inputs)

#for residual, highway dimension reduction
if args.f_highway or args.f_residual:
    t_inputs = TimeDistributed(Dense(node_size, W_regularizer=l2(l2_reg), activation = 'relu'), name = 'dim')(t_inputs)

print("input shape for last feature before temporal model", K.int_shape(t_inputs))
a_max_t_steps = K.int_shape(t_inputs)[1]

for d in range(args.f_dnn_depth):
    if args.f_dnn:
        t_inputs = TimeDistributed(Dense(node_size, W_regularizer=l2(l2_reg), activation = 'relu'), name = 'fc-' + str(d))(t_inputs)
        t_inputs = TimeDistributed(Dropout(dropout))(t_inputs)
    elif args.f_highway:
        t_inputs = TimeDistributed(Highway(W_regularizer=l2(l2_reg), activation = 'relu'), name = 'hw-' + str(d))(t_inputs)
        t_inputs = TimeDistributed(Dropout(dropout))(t_inputs)
    elif args.f_residual:
        d_1 = TimeDistributed(Dense(node_size, W_regularizer=l2(l2_reg), activation = 'relu'), name = 'res-' + str(d)+ "-0")(t_inputs)
        d_2 = TimeDistributed(Dropout(dropout), name = 'd-' + str(d) + "-0")(d_1)
        t_inputs = merge([d_2, t_inputs], mode ='sum', name="c3d_merge_" + str(d))

#global pooling for utterance level lstm
if args.g_pool and args.u_lstm:
    global_inputs = GlobalAveragePooling1D(name='global_pooling')(t_inputs)
#LSTM
if args.f_lstm:
    t_inputs = LSTM(cell_size, return_sequences = True, W_regularizer=l2(l2_reg), dropout_W=dropout, dropout_U=dropout, name = 'f-lstm-0')(t_inputs)
    t_inputs = LSTM(cell_size, return_sequences = True, W_regularizer=l2(l2_reg), dropout_W=dropout, dropout_U=dropout, name = 'f-lstm-1')(t_inputs)
elif args.u_lstm:
    t_inputs = LSTM(cell_size, return_sequences = True, W_regularizer=l2(l2_reg), dropout_W=dropout, dropout_U=dropout, name = 'u-lstm-0')(t_inputs)
    if args.g_pool_rnn:
        t_inputs = LSTM(cell_size, return_sequences = True, W_regularizer=l2(l2_reg), dropout_W=dropout, dropout_U=dropout, name = 'u-lstm-1')(t_inputs)    
        t_inputs = GlobalAveragePooling1D(name='a_global_pooling')(t_inputs)
    else:
        t_inputs = LSTM(cell_size, return_sequences = False, W_regularizer=l2(l2_reg), dropout_W=dropout, dropout_U=dropout, name = 'u-lstm-1')(t_inputs)
elif input_type != "3d" and args.u_dnn:
    t_inputs = Flatten()(t_inputs)    
    if args.g_pool:
        t_inputs = merge([global_inputs, t_inputs], mode = 'concat', name= "merged_a_g_pool_lstm")

if args.u_hw or args.u_residual:
    t_inputs = Dense(node_size, W_regularizer=l2(l2_reg), activation = 'relu', name = 'dim')(t_inputs)

#post dnn for u-lstm or 3d convolution
if utt_model:
    for d in range(args.p_dnn_depth):
        if args.u_dnn:
            layer_name = 'u-fc-' + str(d)
            t_inputs = Dense(node_size, W_regularizer=l2(l2_reg), activation = 'relu', name = layer_name)(t_inputs)
            t_inputs = Dropout(dropout)(t_inputs)
        elif args.u_hw:
            layer_name = 'u-hw-' + str(d)
            t_inputs = Highway(W_regularizer=l2(l2_reg), activation = 'relu', name = layer_name)(t_inputs)
            t_inputs = Dropout(dropout)(t_inputs)
        elif args.u_residual:
            layer_name = 'u_merge_' + str(d)
            d_1 = Dense(node_size, W_regularizer=l2(l2_reg), activation = 'relu', name = 'u_res-' + str(d)+ "-0")(t_inputs)
            d_2 = Dropout(dropout, name = 'd-' + str(d) + "-0")(d_1)
            t_inputs = merge([d_2, t_inputs], mode ='sum', name=layer_name)
else:
    for d in range(args.p_dnn_depth):
        if args.u_dnn:
            layer_name = 'u-fc-' + str(d)
            t_inputs = TimeDistributed(Dense(node_size, W_regularizer=l2(l2_reg), activation = 'relu'), name = layer_name)(t_inputs)
            t_inputs = TimeDistributed(Dropout(dropout))(t_inputs)
        elif args.u_hw:
            layer_name = 'u-hw-' + str(d)
            t_inputs = TimeDistributed(Highway(W_regularizer=l2(l2_reg), activation = 'relu'), name = layer_name)(t_inputs)
            t_inputs = TimeDistributed(Dropout(dropout))(t_inputs)
        elif args.u_residual:
            layer_name = 'u_merge_' + str(d)
            d_1 = TimeDistributed(Dense(node_size, W_regularizer=l2(l2_reg), activation = 'relu'), name = 'u_res-' + str(d) + "-0")(t_inputs)
            d_2 = TimeDistributed(Dropout(dropout), name = 'd-' + str(d) + "-0")(d_1)
            t_inputs = merge([d_2, t_inputs], mode ='sum', name=layer_name)


#video model
v_inputs = Input(shape=(args.v_t_max, 1, args.img_size, args.img_size),name='video_feat')

v = TimeDistributed(ZeroPadding2D((1,1), input_shape=(1, args.img_size, args.img_size)), name = "v_0")(v_inputs)
v = TimeDistributed(Convolution2D(32, 3, 3, activation='relu'), name = "v_1")(v)
v = TimeDistributed(ZeroPadding2D((1,1)), name = "v_2")(v)
v = TimeDistributed(Convolution2D(32, 3, 3, activation='relu'), name = "v_3")(v)
v = TimeDistributed(MaxPooling2D((2,2), strides=(2,2)), name = "v_4")(v)

v = TimeDistributed(ZeroPadding2D((1,1)), name = "v_5")(v)
v = TimeDistributed(Convolution2D(64, 3, 3, activation='relu'), name = "v_6")(v)
v = TimeDistributed(ZeroPadding2D((1,1)), name = "v_7")(v)
v = TimeDistributed(Convolution2D(64, 3, 3, activation='relu'), name = "v_8")(v)
v = TimeDistributed(MaxPooling2D((2,2), strides=(2,2)), name = "v_9")(v)

v = TimeDistributed(ZeroPadding2D((1,1)), name = "v_10")(v)
v = TimeDistributed(Convolution2D(128, 3, 3, activation='relu'), name = "v_11")(v)
v = TimeDistributed(ZeroPadding2D((1,1)), name = "v_12")(v)
v = TimeDistributed(Convolution2D(128, 3, 3, activation='relu'), name = "v_13")(v)
v = TimeDistributed(ZeroPadding2D((1,1)), name = "v_14")(v)
v = TimeDistributed(Convolution2D(128, 3, 3, activation='relu'), name = "v_15")(v)
v = TimeDistributed(MaxPooling2D((2,2), strides=(2,2)), name = "v_16")(v)

v = TimeDistributed(Flatten())(v)
v = TimeDistributed(Dense(1024, activation='relu'), name = "v_17")(v)
v = TimeDistributed(Dropout(0.5), name = "v_18")(v)
v = TimeDistributed(Dense(512, activation='relu'), name = "v_19")(v)
v = TimeDistributed(Dropout(0.5), name = "v_20")(v)

if args.v_u_lstm:
    v = LSTM(args.video_cell_size, return_sequences = True, W_regularizer=l2(l2_reg), dropout_W=dropout, dropout_U=dropout, name = 'v-u-lstm-0')(v)

    if args.g_pool_rnn:
        v = LSTM(args.video_cell_size, return_sequences = True, W_regularizer=l2(l2_reg), dropout_W=dropout, dropout_U=dropout, name = 'v-u-lstm-1')(v)
        v = GlobalAveragePooling1D(name='av_global_pooling')(v)
    else:
        v = LSTM(args.video_cell_size, return_sequences = False, W_regularizer=l2(l2_reg), dropout_W=dropout, dropout_U=dropout, name = 'v-u-lstm-1')(v)
elif args.v_u_gru:
    v = GRU(args.video_cell_size, return_sequences = True, dropout_W=dropout, name = 'v-u-gru-0')(v)

    if args.g_pool_rnn:
        v = GRU(args.video_cell_size, return_sequences = True, dropout_W=dropout, name = 'v-u-gru-1')(v)
        v = GlobalAveragePooling1D(name='av_global_pooling')(v)
    else:
        v = GRU(args.video_cell_size, return_sequences = False, dropout_W=dropout, name = 'v-u-gru-1')(v)
else:
    v = GlobalAveragePooling1D(name='av_global_pooling')(v)

av_inputs = merge([t_inputs, v], mode ='concat', name="a_v_concate")

for i in range(args.av_dnn_depth):
    av_inputs = Dense(args.av_node_size, activation='relu', name = "av_fcn_" + str(i))(av_inputs)
    av_inputs = Dropout(args.dropout, name = "av_drop_"+ str(i))(av_inputs)

#output tasks
predictions = []

for task, classes, idx in multiTasks:
    #frame level modelling using high-level feature ELM
    if args.f_dnn or args.f_lstm or args.f_highway or args.f_residual:
        if classes == 1: #regresssion problem
            predictions.append(TimeDistributed(Dense(classes), name=task)(av_inputs))
        else:
            predictions.append(TimeDistributed(Dense(classes, activation='softmax'),name=task)(av_inputs))
    else:
        #utterance level
        if classes == 1: #regresssion problem
            predictions.append(Dense(classes, name=task)(av_inputs))
        else:
            predictions.append(Dense(classes, activation='softmax',name=task)(av_inputs))

model = Model(input=[inputs, v_inputs], output=predictions)
model.summary()

#freezing and unloading
frozen_layer_list = []
if args.frozen:
    if ',' in args.frozen:
        frozen_layer_list = args.frozen.split(',')
    elif ':' in args.frozen:
        indice = args.frozen.split(':')
        for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
            frozen_layer_list.append(idx)
    else:
        frozen_layer_list = args.frozen.split(",")

unload_layer_list = []
if args.unloaded:
    if ',' in args.unloaded:
        unload_layer_list = args.unloaded.split(',')
    elif ':' in args.unloaded:
        indice = args.unloaded.split(':')
        for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
            unload_layer_list.append(idx)
    else:
        unload_layer_list = args.unloaded.split(",")

for idx in frozen_layer_list:
    print("layer: ", model.layers[int(idx)].name, " is frozen")
    model.layers[int(idx)].trainable = False    

for idx in unload_layer_list:
    print("unloaded layer: ", model.layers[int(idx)].name)
    model.layers[int(idx)].name = "unloaded_" + model.layers[int(idx)].name

print("after frozen and unloaded setups")
model.summary()

#load a pretrained model
if args.a_load_model:
    load_premodel(model, args.a_load_model)

if args.v_load_model:
    load_premodel(model, args.v_load_model)
    
#compile model
model.compile(loss=dictForCost, optimizer=adam, metrics=dictForEval)
print("Final model, layers n = ", len(model.layers))
model.summary()

#utterance level modelling and feature extraction mode
if utt_model and args.w_feat:
    if args.feature_ext_name:
        utt_feat_ext_layer = Model(input=model.input, output=model.get_layer(args.feature_ext_name).output)
    else:
        utt_feat_ext_layer = Model(input=model.input, output=model.layers[n_layers - len(multiTasks) - 1].output)
  
#logging function
test_writer = open(output_file, 'a', 0)

if len(test_idx) > 0 :

    test_indice = []
    valid_indice = []
    adopt_indice = []
    remove_indice = []

    for cid in ignore_idx:
        print("cross-validation ignore: ", cid)
        start_idx = start_indice[int(cid)]
        end_idx = end_indice[int(cid)]
        
        if start_idx == 0 and end_idx == 0:
            continue
            
        for idx in range(int(start_idx), int(end_idx), + 1):
            remove_indice.append(idx)

    for cid in test_idx:
        print("cross-validation test: ", cid)
        start_idx = start_indice[int(cid)]
        end_idx = end_indice[int(cid)]
        
        if start_idx == 0 and end_idx == 0:
            continue
            
        for idx in range(int(start_idx), int(end_idx), + 1):
            test_indice.append(idx)
            remove_indice.append(idx)

    for cid in valid_idx:
        print("cross-validation valid: ", cid)
        start_idx = start_indice[int(cid)]
        end_idx = end_indice[int(cid)]
        
        if start_idx == 0 and end_idx == 0:
            continue
            
        for idx in range(int(start_idx), int(end_idx), + 1):
            remove_indice.append(idx)
            valid_indice.append(idx)
    
    for cid in adopt_idx:
        print("cross-adoptation adopt: ", cid)
        start_idx = start_indice[int(cid)]
        end_idx = end_indice[int(cid)]
        
        if start_idx == 0 and end_idx == 0:
            continue
            
        for idx in range(int(start_idx), int(end_idx), + 1):
            remove_indice.append(idx)
            adopt_indice.append(idx)
            
    if len(train_idx):
        train_indice = []
        for cid in train_idx:
            print("cross-validation train: ", cid)
            start_idx = start_indice[cid]
            end_idx = end_indice[cid]
        
            if start_idx == 0 and end_idx == 0:
                continue
            
            for idx in range(int(start_idx), int(end_idx), + 1):
                train_indice.append(idx)

        a_X_train = a_train_csv[train_indice]
        v_X_train = v_train_csv[train_indice]  
        Y_train = train_lab[train_indice]
    else:
        a_X_train = np.delete(a_train_csv, remove_indice, axis=0)
        v_X_train = np.delete(v_train_csv, remove_indice, axis=0)
        Y_train = np.delete(train_lab, remove_indice, axis=0)

    #test set
    a_X_test = a_train_csv[test_indice]
    v_X_test = v_train_csv[test_indice]  
    Y_test = train_lab[test_indice]
    
    #adopt set
    a_X_adopt = a_train_csv[adopt_indice]
    v_X_adopt = v_train_csv[adopt_indice]  
    Y_adopt = train_lab[adopt_indice]
    
    #valid set
    if len(valid_indice) == 0:
        a_X_valid = a_X_test
        v_X_valid = v_X_test
        Y_valid = Y_test
    else:
        a_X_valid = a_train_csv[valid_indice]  
        v_X_valid = v_train_csv[valid_indice]  
        Y_valid = train_lab[valid_indice]
        r_valid = 0.0

    print('audio train shape: ', a_X_train.shape)
    print('video train shape: ', v_X_train.shape)
    print('audio test shape: ', a_X_test.shape)
    print('video test shape: ', v_X_test.shape)

    scores = train_adopt_evaluate(model, multiTasks, [a_X_train,v_X_train], [a_X_test,v_X_test], [a_X_valid,v_X_valid], [a_X_adopt,v_X_adopt], Y_train, Y_test, Y_valid, Y_adopt, a_max_t_steps, callbacks, utt_level = utt_model, stl = stl, unweighted = args.unweighted, model_save_path = save_model, r_valid = r_valid, epochs = epochs, batch_size = batch_size, class_weights = class_weights, reg = args.reg)

    result = str(scores).replace('[','').replace(']','').replace(', ','\t')
    test_writer.write( result + "\n")

    if args.w_feat:
        test_feat = utt_feature_ext(utt_feat_ext_layer, [a_X_test,v_X_test])
        train_feat = utt_feature_ext(utt_feat_ext_layer, [a_X_train,v_X_train])
        if len(a_X_adopt) > 0:
            adopt_feat = utt_feature_ext(utt_feat_ext_layer, [a_X_adopt,v_X_adopt])
            total_write_utt_feature(args.w_feat + '.adopt.csv', compose_utt_feat(adopt_feat, multiTasks, Y_adopt))
        total_write_utt_feature(args.w_feat + '.test.csv', compose_utt_feat(test_feat, multiTasks, Y_test))            
        total_write_utt_feature(args.w_feat + '.train.csv', compose_utt_feat(train_feat, multiTasks, Y_train))
        
if args.reg:
    print('collected ccc')
    total_write_ccc(test_writer)
else:
    print('collected confusion matrix')
    print(str(total_cm))
    test_writer.write('collected confusion matrix\n')
    test_writer.write(str(total_cm) + '\n')
    total_write_cm(test_writer)    

plot(model, to_file = save_model + ".png")
test_writer.close()
