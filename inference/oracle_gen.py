import os
import numpy as np
from tqdm import tqdm

import torch

import numpy as np
from utils.models import *
from utils.model_utils import self_normalize, sequence_transforms, softmax


from tqdm import tqdm, trange

import argparse


seed = 1234
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('-b', '--bar', type=float, default=0.8)
parser.add_argument('--mode', type=str, default='tripod')

args = parser.parse_args()
            
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def self_normalize(mat):
    max_val = np.amax(mat)
    min_val = np.amin(mat)
    if max_val == min_val:
        return np.zeros(mat.shape)
    mat = (mat - min_val) / (max_val - min_val) * 2 - 1
    return mat

test_users = {'G', 'H', 'I', 'J'}
data_folder = f'/data/mobivital/{args.mode}/'
def inference():   
    score = 0 
    count = 0
    with open(os.path.join('inference', 'methods', f'{args.mode}_oracle.txt'), 'w') as f:
        for file in tqdm(os.listdir(data_folder)):
            user = file.split('_')[1][-1]
            if user not in test_users:
                continue
            data = np.genfromtxt(os.path.join(data_folder, file), delimiter=',').astype(np.float32)
            if len(data) != 1500:
                continue
            uwb = data[:,12:132] + 1j * data[:,132:252]
            breath = self_normalize(data[:, 252])
            heart = self_normalize(data[:, 253])
            
            sequences = []
            for j in range(0, 120):
                sequences += sequence_transforms(uwb[:, j])
            
            best_corr = -1
            best_idx = 0
            invert = 0
            for i, sequence in enumerate(sequences):
                corr = np.corrcoef(self_normalize(breath), sequence)[0][1]
                if corr > best_corr:
                    best_corr = corr
                    best_idx = i
                    invert = 0
                if -corr > best_corr:
                    best_corr = -corr
                    best_idx = i
                    invert = 1
            
            score += best_corr
            count += 1
            if best_idx % 2 == 0:
                method = 'abs'
            else:
                method = 'phase'
                
            bin = best_idx // 2

            save_name = f"{file},{bin},{method},{invert}"
            f.write(save_name + '\n')
            
    print(score/count)
    print(count)
        
       
def main():    
    inference()
            
if __name__ == '__main__':
    main()