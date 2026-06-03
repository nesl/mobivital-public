import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def self_normalize(mat):
    max_val = np.amax(mat)
    min_val = np.amin(mat)
    if max_val == min_val:
        return np.zeros(mat.shape)
    mat = (mat - min_val) / (max_val - min_val) * 2 - 1
    return mat

mode = 'tripod'
rootdir = f"./dataset/mobivital/{mode}/"
save_folder = './data_final/'

if not os.path.exists(save_folder):
    os.mkdir(save_folder)
filenames = os.listdir(rootdir)

X_uwb_train = []
y_breath_train = []

# Construct training dataset
for i in tqdm(range(len(filenames))):
    dataset = np.array([])
    training_user = "ABCDEFKL"
    
    skip_curr_file = 1
    for each in training_user:
        if ("user" + each) in filenames[i] and ".csv" in filenames[i]:
            skip_curr_file = 0
    if skip_curr_file==1:
        continue
    
    with open(os.path.join(rootdir,filenames[i])) as csvfile:
        reader = csv.reader(csvfile)
        dataset = np.array(list(reader)).astype(np.float32)     
        y_heart_train = self_normalize(dataset[:,-1]) # not used
        if len(y_heart_train)!=1500:
            continue
        y_breath_train.append(self_normalize(dataset[:,-2]))  
        X_uwb_train.append(dataset[:,12:132] + 1j * dataset[:,132:252])

X_uwb_train = np.array(X_uwb_train)
y_breath_train = np.array(y_breath_train)

with open(os.path.join(save_folder, f'training_breath_{mode}_data.npy'), 'wb') as f:
    np.save(f, X_uwb_train)
    np.save(f, y_breath_train)


X_uwb_test = []
y_breath_test = []

# Construct testing dataset
for i in tqdm(range(len(filenames))):
    dataset = np.array([])
    testing_user = "GHIJ"
    
    skip_curr_file = 1
    for each in testing_user:
        if ("user" + each) in filenames[i] and ".csv" in filenames[i]:
            skip_curr_file = 0
    if skip_curr_file==1:
        continue
    
    with open(os.path.join(rootdir,filenames[i])) as csvfile:
        reader = csv.reader(csvfile)
        dataset = np.array(list(reader)).astype(np.float32)     
        y_heart_test = self_normalize(dataset[:,-1]) # not used
        if len(y_heart_test)!=1500:
            continue
        y_breath_test.append(self_normalize(dataset[:,-2]))  
        X_uwb_test.append(dataset[:,12:132] + 1j * dataset[:,132:252])

X_uwb_test = np.array(X_uwb_test)
y_breath_test = np.array(y_breath_test)

with open(os.path.join(save_folder, f'testing_breath_{mode}_data.npy'), 'wb') as f:
    np.save(f, X_uwb_test)
    np.save(f, y_breath_test)

