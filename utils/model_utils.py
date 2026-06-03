import numpy as np
import torch
from tqdm import tqdm

from utils.dataset import MyDataset

    
def mad_2d(data):
    mean_vals = np.mean(data, axis=1)
    bar = np.transpose(np.array([mean_vals for i in range(data.shape[1])]))
    return np.mean(np.abs(data - bar), axis=1)


def self_normalize(mat):
    max_val = np.amax(mat)
    min_val = np.amin(mat)
    if max_val == min_val:
        return np.zeros(mat.shape)
    mat = (mat - min_val) / (max_val - min_val) * 2 - 1
    return mat


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sequence_transforms(seq):
    #return [transform(seq, method) for method in ['abs', 'real', 'imag', 'phase']]
    return [transform(seq, method) for method in ['abs', 'phase']]

    #return [self_normalize(np.abs(seq)), self_normalize(np.real(seq)), self_normalize(np.imag(seq)), self_normalize(np.unwrap(np.angle(seq)))]


def transform(seq, method):
    if method == 'abs':
        return self_normalize(np.abs(seq))
    #elif method == 'real':
    #    return self_normalize(np.real(seq))
    #elif method == 'imag':
    #    return self_normalize(np.imag(seq))
    elif method == 'phase':
        return self_normalize(np.unwrap(np.angle(seq)))
    else:
        raise ValueError('invalid method')

def generate_top_sequences(X_uwb, y_breath, top=3):
    top_seqs = []
    for i in range(len(y_breath)):
        curr_seqs = []
        for bins in range(0,120):
            complex_seq = X_uwb[i,:,bins] 
            for seq in sequence_transforms(complex_seq):
                corr_score = np.corrcoef(seq, self_normalize(y_breath[i]))[0][1]
                curr_seqs.append([corr_score, seq])
        curr_seqs = sorted(curr_seqs, key=lambda x: x[0], reverse=True)[:top]
        top_seqs += [seq[1] for seq in curr_seqs]
    
                
    #top_seqs.append(self_normalize(y_breath[i]))
    top_seqs = np.array(top_seqs)
    return top_seqs

def generate_sequences(X_uwb, y_breath, corr_threshold=0.9):
    above_thresh_seqs = []
    below_thresh_seqs = []
    for i in range(len(y_breath)):
        for bins in range(0, 120):
            complex_seq = X_uwb[i,:,bins] 
            
            for seq in sequence_transforms(complex_seq):
                corr_score = np.corrcoef(seq, self_normalize(y_breath[i]))
                if corr_score[0][1] > corr_threshold:
                    above_thresh_seqs.append(seq)
                elif corr_score[0][1] < -corr_threshold:
                    below_thresh_seqs.append(seq)
                   
        above_thresh_seqs.append(self_normalize(y_breath[i]))
    above_thresh_seqs = np.array(above_thresh_seqs)
    below_thresh_seqs = np.array(below_thresh_seqs)
    return above_thresh_seqs, below_thresh_seqs


def generate_dataset(X_uwb, y_breath, batch_size, history_len, future_len, corr_threshold, top=False):
    X = []
    y = []
    steps_forward = future_len

    if top:
        training_seqs = generate_top_sequences(X_uwb, y_breath)
    else:
        training_seqs, _ = generate_sequences(X_uwb, y_breath, corr_threshold)

    for i in range(len(training_seqs)):
        seq = training_seqs[i,...]
        start_idx = 0
        end_idx = start_idx + history_len + future_len
        while end_idx <= len(seq):
            X.append(seq[start_idx:start_idx+history_len])
            y.append(seq[start_idx+history_len:end_idx])
            start_idx += steps_forward
            end_idx = start_idx + history_len + future_len

    X = np.array(X)
    y = np.array(y)
    dataset = MyDataset(X, y)
    return dataset


def generate_dataloader(X_uwb, y_breath, batch_size, history_len, future_len, corr_threshold, shuffle=True, top=False):
    dataset = generate_dataset(X_uwb, y_breath, batch_size, history_len, future_len, corr_threshold, top=top)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    return dataloader
