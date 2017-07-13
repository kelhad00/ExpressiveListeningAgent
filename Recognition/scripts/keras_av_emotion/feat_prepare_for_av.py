from __future__ import print_function
import numpy as np
import argparse
import csv
import h5py
import random
import feature_utility as fu
import os, sys
import cv2
np.random.seed(1337)  # for reproducibility

parser = argparse.ArgumentParser()
parser.add_argument("-input", "--input", dest= 'input', type=str, help="meta file including a list of feature files and labels")
parser.add_argument("-mt", "--multitasks", dest= 'multitasks', type=str, help="multi-tasks (idx:idx:..)", default = '3:4:5:6:7')
parser.add_argument("-f_idx", "--feat_idx", dest= 'feat_idx', type=int, help="feature index (e.g. 8)", default = '8')
parser.add_argument("-img_idx", "--img_idx", dest= 'img_idx', type=int, help="image folder index (e.g. 3)", default = '9')
parser.add_argument("-c_idx", "--c_idx", dest= 'c_idx', type=int, help="cross-validation index (e.g. 3)", default = '3')
parser.add_argument("-n_cc", "--n_cc", dest= 'n_cc', type=int, help="number of cross corpora", default = '0')

parser.add_argument("-a_m_steps", "--a_max_time_steps", dest= 'a_max_time_steps', type=int, help="maximum time steps", default = '50')
parser.add_argument("-v_m_steps", "--v_max_time_steps", dest= 'v_max_time_steps', type=int, help="maximum time steps", default = '50')
parser.add_argument("-c_len", "--context_length", dest= 'context_length', type=int, help="context window length", default = '1')

parser.add_argument("-out", "--output", dest= 'output', type=str, help="output file in HDF5", default="./output")
parser.add_argument("-f_delim", "--feat_delim", dest= 'feat_delim', type=str, help="feat_delim ", default=";")
#parser.add_argument("-img_base_dir", "--img_base_dir", dest= 'img_base_dir', type=str, help="img_base_dir ", default="./")


parser.add_argument("--two_d", help="two_d",
                    action="store_true")
parser.add_argument("--three_d", help="three_d",
                    action="store_true")
parser.add_argument("--headerless", help="headerless in feature file?",
                    action="store_true")
parser.add_argument("--evaluation", help="evaluation",
                    action="store_true")
parser.add_argument("--no_image", help="no_image",
                    action="store_true")


args = parser.parse_args()

if args.input == None:
	print('please specify an input meta file')
	exit(1)

meta_file = open(args.input, "r")
f_idx = args.feat_idx
n_cc = args.n_cc

audio_max_t_steps = args.a_max_time_steps
video_max_t_steps = args.v_max_time_steps

input_dim = -1
feat_delim = args.feat_delim
context_length = args.context_length
half_length = int(context_length / 2)

#parsing 
count = -1
lines = []

for line in meta_file:
	line = line.rstrip()
	if count == -1:
		count = count + 1
		continue
	params = line.split('\t')
	if input_dim == -1:
		feat_file = params[f_idx]
		feat_data = np.genfromtxt (feat_file, delimiter=feat_delim)
		if len(feat_data.shape) == 1:
			input_dim = 1
		else:
			input_dim = feat_data.shape[1]
	
	lines.append(line)
	count = count + 1

#randomise	
if n_cc == 0:
	random.shuffle(lines)

n_samples = count

labels = args.multitasks.split(':')
n_labels = len(labels)

audio_max_t_steps = int(audio_max_t_steps / context_length)

if args.two_d:
	X_audio = np.zeros((n_samples, audio_max_t_steps, 1, context_length, input_dim))
elif args.three_d:
	X_audio = np.zeros((n_samples, 1, audio_max_t_steps, context_length, input_dim))
else:
	X_audio = np.zeros((n_samples, audio_max_t_steps, input_dim * context_length))

X_video = np.zeros((n_samples, video_max_t_steps, 1, 48, 48))

Y = np.zeros((n_samples, n_labels))

indice_map = {}

print('input dim: ' + str(input_dim))
print('number of samples: '+ str(n_samples))
print('number of labels: '+ str(n_labels))
print('max steps: '+ str(audio_max_t_steps))
print('context windows: '+ str(context_length))
print('half length', half_length)
print('audio shape', X_audio.shape)
print('video shape', X_video.shape)

#actual parsing
meta_file.seek(0)
idx = 0

for line in lines:
	line = line.rstrip()
	params = line.split('\t')
	#feature file
	feat_file = params[f_idx]
	feat_data = np.genfromtxt (feat_file, delimiter=feat_delim)

	cid = int(params[args.c_idx])

	indice = indice_map.get(cid)
	if indice == None:
		indice = [idx]
		indice_map[cid] = indice
	else:
		indice.append(idx)

	#2d with context windows
	if args.two_d:
		for t_steps in range(audio_max_t_steps):
			if t_steps * context_length < feat_data.shape[0] - context_length:
				if input_dim != 1 and feat_data.shape[1] != input_dim:
					print('inconsistent dim: ', feat_file)
					break
				for c in range(context_length):
					X_audio[idx, t_steps, 0, c, ] = feat_data[t_steps * context_length + c]
	elif args.three_d:#3d with context windows
		for t_steps in range(audio_max_t_steps):
			if t_steps * context_length < feat_data.shape[0] - context_length:
				if input_dim != 1 and feat_data.shape[1] != input_dim:
					print('inconsistent dim: ', feat_file)
					break
				for c in range(context_length):
					X_audio[idx, 0, t_steps, c, ] = feat_data[t_steps * context_length + c]
	##1d but context windows copy features into time slots
	elif context_length == 1:
		for t_steps in range(audio_max_t_steps):
			if t_steps < feat_data.shape[0]:
				if input_dim != 1 and feat_data.shape[1] != input_dim:
					print('inconsistent dim: ', feat_file)
					break
				X_audio[idx, t_steps,] = feat_data[t_steps]
	else:#1d but context windows
		for t_steps in range(audio_max_t_steps):
			if t_steps * context_length < feat_data.shape[0] - context_length:
				if input_dim != 1 and feat_data.shape[1] != input_dim:
					print('inconsistent dim: ', feat_file)
					break
				for c in range(context_length):
					X_audio[idx, t_steps, c * input_dim: (c + 1) * input_dim] = feat_data[t_steps * context_length + c]
	#TODO
	if args.no_image:
		print("no image")
	else:
		#1. read images from folder
		image_folder = params[args.img_idx]
		print("image_folder", image_folder)
	    #retreive files
		onlyfiles = next(os.walk(image_folder))[2]
		onlyfiles.sort()

	        
		#for empty folder
		if len(onlyfiles) == 0:
			print("There is no frames in ", image_folder)
			print("We leave it as zeros for masking")
		else:
			gap = len(onlyfiles) / video_max_t_steps

			#2. gray scale & resize
			for v_idx in range(video_max_t_steps):
				frame_idx = v_idx * gap
				full_path = image_folder + "/" + onlyfiles[frame_idx]
				print(feat_file, ", chosen frame: ", full_path)
				img = fu.preprocessing(cv2.imread(full_path))
				frame = np.expand_dims(img, axis=0)
		    	frame = np.expand_dims(frame, axis=0)
		    	#3. put that into temporal windows
		    	X_video[idx, v_idx] = frame
	
	#copy labels
	for lab_idx in range(n_labels):
		Y[idx, lab_idx] = params[int(labels[lab_idx])]

	idx = idx + 1
	print("processing: ", idx, " :", feat_file)
print('successfully write samples: ' + str(idx))

h5_output = args.output + '.h5'


if n_cc > 0:
	idx = 0
	# loading and constructing each fold in memory takes too much time.
	index_list = []
	start_indice = np.zeros((n_cc))
	end_indice = np.zeros((n_cc))

	if args.two_d:
		X_audio_ordered = np.zeros((n_samples, audio_max_t_steps, 1, context_length, input_dim))
	elif args.three_d:
		X_audio_ordered = np.zeros((n_samples, 1, audio_max_t_steps, context_length, input_dim))
	else:
		X_audio_ordered = np.zeros((n_samples, audio_max_t_steps, input_dim * context_length))
	
	X_video_ordered = np.zeros((n_samples, video_max_t_steps, 1, 48, 48))

	Y_ordered = np.zeros((n_samples, n_labels))

	start_idx = 0
	end_idx = 0
	for cid, indice in indice_map.items():
		#print('indice', indice)
		if indice == None:
			continue
		X_audio_temp = X_audio[indice]
		Y_temp = Y[indice]
		end_idx = start_idx + X_audio_temp.shape[0]
		print('shape', X_audio_temp.shape)
		start_indice[idx] = start_idx
		end_indice[idx] = end_idx
		print("corpus: ", idx, " starting from: ", start_idx, " ends: ", end_idx)
		X_audio_ordered[start_idx:end_idx] = X_audio_temp
		X_video_ordered[start_idx:end_idx] = X_video[indice]

		Y_ordered[start_idx:end_idx] = Y_temp
		start_idx = end_idx
		idx = idx + 1
		
	print("shape of audio feat: ", X_audio_ordered.shape)
	print("shape of video feat: ", X_video_ordered.shape)
	print("shape of label: ", Y_ordered.shape)
	with h5py.File(h5_output, 'w') as hf:
		hf.create_dataset('a_feat', data= X_audio_ordered)
		hf.create_dataset('v_feat', data= X_video_ordered)
		hf.create_dataset('label', data= Y_ordered)
		hf.create_dataset('start_indice', data=start_indice)
		hf.create_dataset('end_indice', data=end_indice)
	print('total cv: ' + str(len(start_indice)))	
	
else:
	print("shape of a_feat: ", X_audio.shape)
	print("shape of v_feat: ", X_video.shape)
	print("shape of label: ", Y.shape)
	with h5py.File(h5_output, 'w') as hf:
		hf.create_dataset('a_feat', data=X_audio)
		hf.create_dataset('v_feat', data=X_video)
		hf.create_dataset('label', data=Y)

meta_file.close()
