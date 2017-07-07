'''
=======================================================================
Animated plot of the 3d face landmarks sequence given csv file as input
=======================================================================
Author: Yong Zhao
@eNTERFACE2017--Project: End-to-End Listening Agent for Audio-Visual Emotional and Naturalistic Interactions
'''
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import pandas as pd
import glob

def get3d_data(file_name,file_type='openface'):
    point_all = []
    if file_type == 'openface':
        raw_data = pd.read_csv(file_name)
        #find the location of 3d landmarks
        a = raw_data.columns.get_loc(' X_0')
        b = raw_data.columns.get_loc(' Z_67')
        data = raw_data.ix[:, a:b+1]
        for nr_points in range(68):
            shape_x = data.iloc[0:, nr_points]
            shape_y = data.iloc[0:, nr_points + 68]
            shape_z = data.iloc[0:, nr_points + 136]
            point_one = np.stack((shape_x, shape_z, shape_y))  # switch the position of y and z to have a right view
            point_all.append(point_one)
    elif file_type == 'optitrack':
        raw_data = pd.read_csv(file_name,delimiter=';',header=None,usecols=range(6,105))
        data = raw_data
        for nr_points in range(0,99,3):
            shape_x = data.iloc[0:, nr_points]
            shape_y = data.iloc[0:, nr_points + 1]
            shape_z = data.iloc[0:, nr_points + 2]
            point_one = np.stack((shape_x, shape_y, shape_z))
            point_all.append(point_one)
    n_frames = data.shape[0]
    return point_all,n_frames

def concat_point(exp1, exp2):
    nr_frame1 = exp1[0].shape[1]
    nr_frame2 = exp2[0].shape[1]
    points_concat = []
    for nr_point in range(68):
        #concatenate the two expression sequences landmark by landmark
        points_concat.append(np.concatenate((exp1[nr_point], exp2[nr_point][:, :-1]), axis=1))
        for t in range(nr_frame2-1):
            #for the 2nd expression, we first compute the variations between each two frames, then compute a new expression
            #sequence based on the last frame of the 1st expression
            variation = exp2[nr_point][:, t + 1] - exp2[nr_point][:, 0]
            #points_concat[nr_point][:,nr_frame1 + t] = exp1[nr_point][:,0] + variation
            points_concat[nr_point][:, nr_frame1 + t] = exp1[nr_point][:, nr_frame1-1] + variation
    return points_concat

def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        line.set_data(data[0:2, num-1:num])
        line.set_3d_properties(data[2, num-1:num])
    return lines


fig = plt.figure()
ax = p3.Axes3D(fig)

csv_lst = glob.glob("data\*.csv") #the csv file is created by openface

# Input 1 expression file
#csv_file = csv_lst[0]  #choose csv file from data folder
#point_all,n = get3d_data(csv_file)

# Input 2 expression files
csv_file1 = csv_lst[0]  #choose csv file from data folder
csv_file2 = csv_lst[2]  #choose csv file from data folder
expression1,n1 = get3d_data(csv_file1)
expression2,n2 = get3d_data(csv_file2)
expression_all = concat_point(expression1, expression2)
n=n1+n2-1

#point_all,n = get3d_data(csv_file,'optitrack')

lines = [ax.plot(point[0, 0:1], point[1, 0:1], point[2, 0:1],'k.')[0] for point in expression_all]

#Set axes properties, note that the position of shape_y and shape_z have been switched
#ax.set_xlim3d([-60, 140])

ax.set_xlabel('X')

#ax.set_ylim3d([360, 480])

ax.set_ylabel('Y')

#ax.set_zlim3d([60,100]) #invert the axe
ax.set_zlim3d([100,-60])
ax.set_zlabel('Z')

ax.set_title('3D Plot Test')

# Creating the Animation object
# 'interval is millisecond
ani = animation.FuncAnimation(fig, update_lines, n, fargs=(expression_all, lines),
                              interval=33, blit=False)

plt.show()