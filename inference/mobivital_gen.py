import os
import numpy as np
from tqdm import tqdm

import torch

import numpy as np
from utils.models import *
from utils.model_utils import self_normalize, sequence_transforms, softmax

from utils.peak_width_inverter import invert_detector


from tqdm import tqdm, trange
from einops import rearrange

import argparse
import json


seed = 1234
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, default='checkpoints')
parser.add_argument('--model_name', type=str, default='lstm_pred')
parser.add_argument('-c', '--corr', type=float, default=0.9)

parser.add_argument('-b', '--bar', type=float, default=0.8)
parser.add_argument('--params_file', type=str, default='checkpoints/optimal_params.json')
parser.add_argument('--mode', type=str, default='tripod')

args = parser.parse_args()

args.model_name = f'{args.model_name}_{args.mode}_{args.corr}.pth'


if args.params_file:
    with open(args.params_file, 'r') as f:
        params = json.load(f)
        for key, value in params.items():
            setattr(args, key, value)
            
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_invert_prob(sequence):
    return invert_detector(sequence)


def get_best_sequence(original_sequence, model, history_len=args.history_length, future_len=args.future_length):
    original_sequence = rearrange(original_sequence, "t c -> c t")
    sequences = []
    for s in original_sequence:
        transformed = sequence_transforms(s)
        sequences += [transformed[0], transformed[-1]]

    filtered_sequences = []
    index_map = {}
    counter = 0
    for i, seq in enumerate(sequences):
        if get_invert_prob(seq) < args.bar:
            index_map[counter] = i
            counter += 1
            filtered_sequences.append(seq)

    sequences = np.array(filtered_sequences)
    # (36, 1500)
    steps_forward = future_len
    
    X = []
    y = []
    for i in range(len(sequences)):
        seq = sequences[i,...]
        start_idx = 0
        end_idx = start_idx + history_len + future_len
        while end_idx <= len(seq):
            X.append(seq[start_idx:start_idx+history_len])
            y.append(seq[start_idx+history_len:end_idx])
            start_idx += steps_forward
            end_idx = start_idx + history_len + future_len
    X = np.array(X)
    y = np.array(y)
    with torch.no_grad():
        inputs = torch.from_numpy(X).to(device).float()
        labels = torch.from_numpy(y).to(device).float()
        outputs = model(inputs)
    losses = []
    for sample in range(len(labels)):
        losses.append(np.corrcoef(outputs[sample].cpu().numpy(), labels[sample].cpu().numpy())[0,1])
    losses = np.array(losses)
    l = (1500 - history_len) // future_len
    losses = rearrange(losses, "(c l) -> c l", l=l)
    min_loss_idx = np.argmax(np.sum(losses, axis=1))
    best_corr = np.sum(losses[min_loss_idx])
    return sequences[min_loss_idx], index_map[min_loss_idx]


def self_normalize(mat):
    max_val = np.amax(mat)
    min_val = np.amin(mat)
    if max_val == min_val:
        return np.zeros(mat.shape)
    mat = (mat - min_val) / (max_val - min_val) * 2 - 1
    return mat

test_users = {'G', 'H', 'I', 'J'}
data_folder = f'./dataset/mobivital/{args.mode}/'

def inference(autoreg_model):
    score = 0
    count = 0
    store_path = os.path.join('inference', 'methods')
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    with open(os.path.join('inference', 'methods', f'{args.mode}_mobivital_pre_invert_{args.corr}.txt'), 'w') as f:
        for file in tqdm(os.listdir(data_folder)):
            user = file.split('_')[1][-1]
            if user not in test_users:
                continue
            data = np.genfromtxt(os.path.join(data_folder, file), delimiter=',').astype(np.float32)
            if len(data) != 1500:
                continue
            #uwb = data[:, 12:252]
            #0, 1, 2, 3, 4, 5 are imu
            #6, 7, 8, 9, 10, 11 are camera
            uwb = data[:,12:132] + 1j * data[:,132:252]
            breath = self_normalize(data[:, 252])
            heart = self_normalize(data[:, 253])
            best_sequence, best_idx = get_best_sequence(uwb, autoreg_model)
   
            invert_bit = 0
            score += np.corrcoef(self_normalize(breath), best_sequence)[0][1]
            count += 1
            
            if best_idx % 2 == 0:
                method = 'abs'
            else:
                method = 'phase'
            bin = best_idx // 2
            save_name = f"{file},{bin},{method},{invert_bit}"
            f.write(save_name + '\n')
        
    print(score/count)

def main():
    model = LSTMMultiStep(args.hidden_size, args.num_layers, args.future_length)
    model.load_state_dict(torch.load(os.path.join(args.model_folder, args.model_name), map_location=device))
    model.to(device)
    inference(model)
            
if __name__ == '__main__':
    main()
