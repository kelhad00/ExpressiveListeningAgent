#!/bin/python
# python2.7
# Train and evaluate (audio-visual) model for the prediction of arousal / valence / liking
# The complexity of the SVM regressor is optimised.
# The performance on the development set in terms of CCC, PCE, and MSE is appended to corresponding text files (results.txt, results_pcc.txt, results_mse.txt).
# The predicitions on the test set are written into the folder specified by the variable 'path_test_predictions'.
# 
# 
# Contact: maximilian.schmitt@uni-passau.de
# Based on the original script, more functions were added (e.g. feature dimension reduction)

import os
import fnmatch
import numpy as np
import argparse
import sys
from sys     import argv
from sklearn import svm
from sklearn.decomposition import PCA, IncrementalPCA, SparsePCA, KernelPCA
from sklearn.lda import LDA
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import RadiusNeighborsRegressor

from load_features     import load_all
from calc_scores       import calc_scores
from write_predictions import write_predictions

def build_model(args, C, seed):
    if args.dc_tree:
        model = DecisionTreeRegressor(random_state=seed)
    elif args.nn_radius:
        model = RadiusNeighborsRegressor(radius=1.0)
    else:
        model = svm.LinearSVR(C=complexities[comp],random_state=seed)
    
    return model

parser = argparse.ArgumentParser()

parser.add_argument("-path_train", "--path_train", dest= 'path_train', type=str, help="path train (only if all data speaker sets are merged)")
parser.add_argument("-path_devel", "--path_devel", dest= 'path_devel', type=str, help="path devel (only if all data speaker sets are merged)")


parser.add_argument("-path_test", "--path_test_predictions", dest= 'path_test_predictions', type=str, help="prediction folder", default = "test_predictions/")
parser.add_argument("-path_audio", "--path_audio_features", dest= 'path_audio_features', type=str, help="path_audio_features", default = "audio_features_xbow_6s/")
parser.add_argument("-path_video", "--path_video_features", dest= 'path_video_features', type=str, help="path_video_features", default = "video_features_xbow_6s/")
parser.add_argument("-path_text", "--path_text_features", dest= 'path_text_features', type=str, help="path_text_features", default = "text_features_xbow_6s/")
parser.add_argument("-path_labels", "--path_labels", dest= 'path_labels', type=str, help="path_labels", default = "labels/")

parser.add_argument("-path_save_train_feat", "--path_save_train_feat", dest= 'path_save_train_feat', type=str, help="path to save train features")
parser.add_argument("-path_save_devel_feat", "--path_save_devel_feat", dest= 'path_save_devel_feat', type=str, help="path to save devel features")


parser.add_argument("-delay", "--delay", dest= 'delay', type=float, help="delay(Sec)", default = 0.0)
parser.add_argument("--audio", help="audio", action="store_true")
parser.add_argument("--video", help="video", action="store_true")
parser.add_argument("--text", help="text", action="store_true")

parser.add_argument("--arousal", help="arousal", action="store_true")
parser.add_argument("--valence", help="valence", action="store_true")
parser.add_argument("--liking", help="liking", action="store_true")

parser.add_argument("--test_label",help="test labels are available", action="store_true")
parser.add_argument("--pca", help="pca", action="store_true")
parser.add_argument("--ipca", help="ipca", action="store_true")
parser.add_argument("--kpca", help="kpca", action="store_true")
parser.add_argument("--spca", help="spca", action="store_true")

parser.add_argument("--lda", help="lda", action="store_true")
parser.add_argument("--dc_tree", help="dc_tree", action="store_true")
parser.add_argument("--nn_radius", help="nn_radius", action="store_true")

parser.add_argument("-pl_dim", "--pl_dim", dest= 'pl_dim', type=int, help="PCA or LDA dimensions", default = 30)
parser.add_argument("-ipca_batch", "--ipca_batch", dest= 'ipca_batch', type=int, help="Incremental PCA batch size", default = 10)

parser.add_argument("--write_result", help="write_result", action="store_true")
args = parser.parse_args()

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

# Set folders here
path_test_predictions = args.path_test_predictions

if args.test_label:
    b_test_available = True
else:
    b_test_available = False  # If the test labels are not available, the predictions on test are written into the folder 'path_test_predictions'

# Folders with provided features and labels
path_audio_features = args.path_audio_features
path_video_features = args.path_video_features
path_text_features  = args.path_text_features
path_labels         = args.path_labels

sr_labels = 0.1

delay = args.delay
if args.audio:
    b_audio = True
else:
    b_audio = False
if args.video:
    b_video = True
else:
    b_video = False
if args.text:
    b_text = True
else:
    b_text = False

path_features = []
if b_audio:
    path_features.append( path_audio_features )
if b_video:
    path_features.append( path_video_features )
if b_text:
    path_features.append( path_text_features )

if not b_test_available and not os.path.exists(path_test_predictions):
    os.mkdir(path_test_predictions)

if args.path_train and args.path_devel:
    data = np.genfromtxt (args.path_train, delimiter="\t")
    Train = data[:,0:(data.shape[1] - 2)]
    Train_L = data[:,(data.shape[1] - 2):(data.shape[1])]

    data = np.genfromtxt (args.path_devel, delimiter="\t")
    Devel = data[:,0:(data.shape[1] - 2)]
    Devel_L = data[:,(data.shape[1] - 2):(data.shape[1])]
else:
    # Compensate the delay (quick solution)
    shift = int(np.round(delay/sr_labels))
    shift = np.ones(len(path_features),dtype=int)*shift

    files_train = fnmatch.filter(os.listdir(path_features[0]), "Train*")  # Filenames are the same for audio, video, text & labels
    files_devel = fnmatch.filter(os.listdir(path_features[0]), "Devel*")
    files_test  = fnmatch.filter(os.listdir(path_features[0]), "Test*")

    # Load features and labels
    Train   = load_all( files_train, path_features, shift )
    Devel   = load_all( files_devel, path_features, shift )
    Train_L = load_all( files_train, [ path_labels ] )  # Labels are not shifted
    Devel_L = load_all( files_devel, [ path_labels ] )

if b_test_available:
    Test   = load_all( files_test, path_features, shift )
    Test_L = load_all( files_test, [ path_labels ] )  # Test labels are not available in the challenge
else:
    Test   = load_all( files_test, path_features, shift, separate=True )  # Load test features separately to store the predictions in separate files

print("Original features")
print("Train feature shape: ", Train.shape)
print("Train_L feature shape: ", Train_L.shape)

print("Devel feature shape: ", Devel.shape)
print("Devel_L feature shape: ", Devel_L.shape)
print("Test feature shape: ", Test.shape)
if b_test_available:
    print("Test_L feature shape: ", Test_L.shape)

if args.pca:
    print("PCA transforming...")
    pca = PCA(n_components=args.pl_dim)
    Train = pca.fit_transform(Train)
    Devel = pca.fit_transform(Devel)
elif args.spca:
    print("SparsePCA transforming...")
    pca = SparsePCA(n_components=args.pl_dim)
    Train = pca.fit_transform(Train)
    Devel = pca.fit_transform(Devel)
elif args.kpca:
    print("KernelPCA transforming...")
    pca = KernelPCA(n_components=args.pl_dim)
    Train = pca.fit_transform(Train)
    Devel = pca.fit_transform(Devel)
elif args.ipca:
    print("i-PCA transforming...")
    ipca = IncrementalPCA(batch_size=args.ipca_batch, copy=True, n_components=args.pl_dim, whiten=True)
    Train = ipca.fit_transform(Train)
    Devel = ipca.fit_transform(Devel)
elif args.lda:
    print("LDA transforming...")
    lda = LDA(n_components=args.pl_dim)
    
    if args.arousal:
        labels = Train_L[:,0]
    elif args.valence:
        labels = Train_L[:,1]
    elif args.liking:
        labels = Train_L[:,2]
        
    lda = lda.fit(Train, labels) #learning the projection matrix
    Train = lda.transform(Train)
    Devel = lda.transfrom(Devel)
    
print("After feature transformation")
print("Train feature shape: ", Train.shape)
print("Train_L feature shape: ", Train_L.shape)

print("Devel feature shape: ", Devel.shape)
print("Devel_L feature shape: ", Devel_L.shape)

if args.path_save_train_feat:
    np.savetxt(args.path_save_train_feat,  np.append(Train, Train_L, axis=1), delimiter=',')
if args.path_save_devel_feat:
    np.savetxt(args.path_save_devel_feat,  np.append(Devel, Devel_L, axis=1), delimiter=',')

print("Test feature shape: ", Test.shape)
if b_test_available:
    print("Test_L feature shape: ", Test_L.shape)
    
# Run liblinear (scikit-learn)
# Optimize complexity
num_steps = 16
complexities = np.logspace(-15,0,num_steps,base=2.0)  # 2^-15, 2^-14, ... 2^0

scores_devel_A = np.empty((num_steps,3))
scores_devel_V = np.empty((num_steps,3))
scores_devel_L = np.empty((num_steps,3))

seed = 0

print("Finding optimal params")
for comp in range(0,num_steps):
    print("Finding optimal param: ", complexities[comp])

    if args.arousal:
        model_A = build_model(args, complexities[comp], seed)#svm.LinearSVR(C=complexities[comp],random_state=seed)
        model_A.fit(Train,Train_L[:,0])
        predA = model_A.predict(Devel)
        scores_devel_A[comp,:] = calc_scores(Devel_L[:,0],predA)
    
    if args.valence:
        model_V = build_model(args, complexities[comp], seed)#svm.LinearSVR(C=complexities[comp],random_state=seed)
        model_V.fit(Train,Train_L[:,1])
        predV = model_V.predict(Devel)
        scores_devel_V[comp,:] = calc_scores(Devel_L[:,1],predV)
    
    if args.liking:    
        model_L = build_model(args, complexities[comp], seed)#svm.LinearSVR(C=complexities[comp],random_state=seed)
        model_L.fit(Train,Train_L[:,2])
        predL = model_L.predict(Devel)
        scores_devel_L[comp,:] = calc_scores(Devel_L[:,2],predL)

ind_opt_A = np.argmax(scores_devel_A[:,0])
ind_opt_V = np.argmax(scores_devel_V[:,0])
ind_opt_L = np.argmax(scores_devel_L[:,0])
comp_opt_A = complexities[ind_opt_A]
comp_opt_V = complexities[ind_opt_V]
comp_opt_L = complexities[ind_opt_L]

# Run on train+devel with optimum complexity and predict on the test set
TrainDevel   = np.concatenate((Train, Devel), axis=0)
TrainDevel_L = np.concatenate((Train_L, Devel_L), axis=0)

if args.pca:
    print("PCA transforming...")
    pca = PCA(n_components=args.pl_dim)
    TrainDevel = pca.fit_transform(TrainDevel)
elif args.spca:
    print("SparsePCA transforming...")
    pca = SparsePCA(n_components=args.pl_dim)
    TrainDevel = pca.fit_transform(TrainDevel)
elif args.kpca:
    print("KernelPCA transforming...")
    pca = KernelPCA(n_components=args.pl_dim)
    TrainDevel = pca.fit_transform(TrainDevel)
elif args.ipca:
    print("i-PCA transforming...")
    ipca = IncrementalPCA(batch_size=args.ipca_batch, copy=True, n_components=args.pl_dim, whiten=True)
    TrainDevel = ipca.fit_transform(TrainDevel)
elif args.lda:
    print("LDA transforming...")
    lda = LDA(n_components=args.pl_dim)
    if args.arousal:
        labels = TrainDevel[:,0]
    elif args.valence:
        labels = TrainDevel[:,1]
    elif args.liking:
        labels = TrainDevel[:,2]
    
    lda = lda.fit(TrainDevel, labels) #learning the projection matrix
    TrainDevel = lda.transform(TrainDevel)

print("Finding optimal param: ", comp_opt_A, ", ", comp_opt_V, ", ", comp_opt_L)

if args.arousal:
    model_A = build_model(args, comp_opt_A, seed) #svm.LinearSVR(C=comp_opt_A,random_state=seed)
    model_A.fit(TrainDevel,TrainDevel_L[:,0])
if args.valence:
    model_V = build_model(args, comp_opt_A, seed) #svm.LinearSVR(C=comp_opt_V,random_state=seed)
    model_V.fit(TrainDevel,TrainDevel_L[:,1])
if args.liking:
    model_L = build_model(args, comp_opt_A, seed) #svm.LinearSVR(C=comp_opt_L,random_state=seed)
    model_L.fit(TrainDevel,TrainDevel_L[:,2])

if b_test_available:
    if args.arousal:
        predA = model_A.predict(Test)
        score_test_A = calc_scores(Test_L[:,0],predA)
    if args.valence:
        predV = model_V.predict(Test)
        score_test_V = calc_scores(Test_L[:,1],predV)
    if args.liking:
        predL = model_L.predict(Test)
        score_test_L = calc_scores(Test_L[:,2],predL)
else:
    for f in range(0,len(files_test)):
        
        if args.pca:
            Test[f] = pca.fit_transform(Test[f])
        elif args.spca:
            pca = SparsePCA(n_components=args.pl_dim)
            Test[f] = pca.fit_transform(Test[f])
        elif args.kpca:
            pca = KernelPCA(n_components=args.pl_dim)
            Test[f] = pca.fit_transform(Test[f])
        elif args.ipca:
            ipca = IncrementalPCA(batch_size=args.ipca_batch, copy=True, n_components=args.pl_dim, whiten=True)
            Test[f] = ipca.fit_transform(Test[f])
        elif args.lda:
            Test[f] = lda.transform(Test[f])
            
        if args.arousal:
            predA = model_A.predict(Test[f])
        if args.valence:
            predV = model_V.predict(Test[f])
        if args.liking:
            predL = model_L.predict(Test[f])

        if args.write_result:
            predictions = np.array([predA,predV,predL])
            write_predictions(path_test_predictions,files_test[f],predictions,sr_labels)


# Print scores (CCC, PCC, RMSE) on the development set
if args.arousal:
    print("Arousal devel (CCC,PCC,RMSE):")
    print(scores_devel_A[ind_opt_A,:])
if args.valence:
    print("Valence devel (CCC,PCC,RMSE):")
    print(scores_devel_V[ind_opt_V,:])
if args.liking:
    print("Liking  devel (CCC,PCC,RMSE):")
    print(scores_devel_L[ind_opt_L,:])

if args.write_result:
    if b_test_available:
        result_ccc  = [ scores_devel_A[ind_opt_A,0], score_test_A[0], scores_devel_V[ind_opt_V,0], score_test_V[0], scores_devel_L[ind_opt_L,0], score_test_L[0] ]
        result_pcc  = [ scores_devel_A[ind_opt_A,1], score_test_A[1], scores_devel_V[ind_opt_V,1], score_test_V[1], scores_devel_L[ind_opt_L,1], score_test_L[1] ]
        result_rmse = [ scores_devel_A[ind_opt_A,2], score_test_A[2], scores_devel_V[ind_opt_V,2], score_test_V[2], scores_devel_L[ind_opt_L,2], score_test_L[2] ]
        print("Arousal test (CCC,PCC,RMSE):")
        print(score_test_A)
        print("Valence test (CCC,PCC,RMSE):")
        print(score_test_V)
        print("Liking  test (CCC,PCC,RMSE):")
        print(score_test_L)
    else:
        # Write only the scores for the development set
        result_ccc  = [ scores_devel_A[ind_opt_A,0], scores_devel_V[ind_opt_V,0], scores_devel_L[ind_opt_L,0] ]
        result_pcc  = [ scores_devel_A[ind_opt_A,1], scores_devel_V[ind_opt_V,1], scores_devel_L[ind_opt_L,1] ]
        result_rmse = [ scores_devel_A[ind_opt_A,2], scores_devel_V[ind_opt_V,2], scores_devel_L[ind_opt_L,2] ]

    # Write scores into text files
    with open("results_ccc.txt", 'a') as myfile:
        myfile.write("Arousal Valence Liking\n")
        myfile.write(str(result_ccc) + '\n')
    with open("results_pcc.txt", 'a') as myfile:
        myfile.write("Arousal Valence Liking\n")
        myfile.write(str(result_pcc) + '\n')
    with open("results_rmse.txt", 'a') as myfile:
        myfile.write("Arousal Valence Liking\n")
        myfile.write(str(result_rmse) + '\n')
