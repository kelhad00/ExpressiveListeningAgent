from __future__ import print_function
import numpy as np
np.random.seed(1337) 

import tensorflow as tf
tf.set_random_seed(2016)

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Activation, TimeDistributed, merge
from keras.engine.topology import Merge
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
from keras import backend as K
from keras.layers.core import Highway
from keras.layers.core import Permute
from keras.layers import Input, Convolution3D,Convolution2D,Convolution1D, MaxPooling3D,MaxPooling2D, MaxPooling1D, GlobalAveragePooling1D, Flatten
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
from keras.models import load_model

def uw_categorical_accuracy(y_true, y_pred, k = 2):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)

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

def load_data_in_range(data_path, start, end, feat = 'feat', label = 'label'):
    X = np.array(keras.utils.io_utils.HDF5Matrix(data_path, feat, start, end))
    Y = np.array(keras.utils.io_utils.HDF5Matrix(data_path, label, start, end))
    return X, Y

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
    
parser = argparse.ArgumentParser()
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
parser.add_argument("-f_layer", "--freeze_layer", dest = 'f_layer', type=int, help="freeze a loaded model, upto N (ex. 2 = upto 2 layers, -1=fully update, 0=fully frozen)", default = -1)

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
parser.add_argument("-t_max", "--t_max", dest= 't_max', type=int, help="max length of time sequence")

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

parser.add_argument("-alm", "--a_load_model", dest= 'a_load_model', type=str, help="load pre-trained model for audio")
parser.add_argument("-vlm", "--v_load_model", dest= 'v_load_model', type=str, help="load pre-trained model for video")

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
parser.add_argument("--u_lstm", help="utterance level lstm do not require elm",
                    action="store_true")
parser.add_argument("--u_dnn", help="utterance level dnn after u_lstm do not require elm",
                    action="store_true")
parser.add_argument("--u_hw", help="utterance level highway after u_lstm do not require elm",
                    action="store_true")
parser.add_argument("--u_residual", help="utterance level residual after u_lstm do not require elm",
                    action="store_true")
parser.add_argument("--g_pool", help="global pooling for temporal features",
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

parser.add_argument("--decoding", help="decoding using a loaded model", action="store_true")
parser.add_argument("--lm_new_output", help="using a loaded model but make a new softmax layer; pretrained models and new models have different classes", action="store_true")
parser.add_argument("--reverse_adopt", help="freeze layers from top to bottom", action="store_true")
                    
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
    train_csv, train_lab = load_data_in_range(data_path, 0, 1)
else:
    with h5py.File(data_path,'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = hf.get('feat')
        train_csv = np.array(data)
        data = hf.get('label')
        train_lab = np.array(data)
        print('Shape of the array feat: ', train_csv.shape)
        print('Shape of the array lab: ', train_lab.shape)
        if len(n_cc) or len(test_idx) > 0:
            start_indice = np.array(hf.get('start_indice'))
            end_indice = np.array(hf.get('end_indice'))
            print('Shape of the indice for start: ', start_indice.shape)

input_type = "1d"

#2d input
if len(train_csv.shape) == 5:
    input_dim = train_csv.shape[4]
    context_len = train_csv.shape[3]
    if args.conv_3d or args.conv_hw_3d or args.r_conv_3d:
        input_type = "3d"
    else:    
        input_type = "2d"
else:#1d input
    input_dim = train_csv.shape[2]

max_t_steps = train_csv.shape[1]

nameAndClasses = tasks.split(',')
multiTasks = []
dictForCost = {}
dictForWeight = {}
dictForEval = {}

for task in nameAndClasses:
    params = task.split(':')
    name = params[0]
        
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


a_inputs = Input(shape=(args.t_max, 1, context_len, input_dim),name='audio_feat')
v_inputs = Input(shape=(1, 48, 48),name='video_feat')


if args.a_load_model and args.v_load_model:
    print("Pre-trained audio model:", args.a_load_model)
    audio_premodel = load_model(args.a_load_model)

    for task, classes, idx in multiTasks:
        print("pretrained output node: ", task)
        audio_premodel.layers.pop()

    audio_premodel.summary()

    print("Pre-trained video model:", args.v_load_model)
    video_premodel = load_model(args.v_load_model)
    n_layers = len(video_premodel.layers)
    video_premodel.layers.pop()

    video_premodel.summary()

    a_model_len = len(audio_premodel.layers)
    v_model_len = len(video_premodel.layers)    

    top_audio_layer = audio_premodel.layers[a_model_len - 1]

    top_video_layer = video_premodel.layers[v_model_len - 1]
    
    concat_a_v_layer = merge([top_audio_layer.output, top_video_layer.output], mode = 'concat')
    x = Dense(64, activation='relu')(concat_a_v_layer)
    x = Dense(64, activation='relu')(x)
    prediction = Dense(4, activation='softmax')(x)

    audio_premodel.set_input(a_inputs)
    video_premodel.set_input(v_inputs)
    model = Model(input=[a_inputs, v_inputs], output=prediction)

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics="accuracy")
    print("New model: ")
    model.summary()

