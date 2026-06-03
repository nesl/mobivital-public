import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from utils.model_utils import self_normalize, transform


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--methods_file', type=str, default='tripod_mobivital_pre_invert_0.9.txt')
parser.add_argument('--methods_folder', type=str, default='inference/methods/')
parser.add_argument('--mat', action=argparse.BooleanOptionalAction, default=False)

parser.add_argument('-d', '--data_folder', type=str, default='./dataset/mobivital/tripod/')
parser.add_argument('--save_file', type=str, default='scores.csv')
parser.add_argument('--invert', action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

def sequence_itr(methods_file, data_folder):
    with open(methods_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            file, bin, method, invert = row
            bin, invert = int(bin), int(invert)
            if args.mat:
                bin -= 1
            data = np.genfromtxt(os.path.join(data_folder, file), delimiter=',').astype(np.float32)
            if len(data) != 1500:
                    continue
            gt = self_normalize(data[:, 252])
            heart = self_normalize(data[:, 253])

            uwb = data[:,12:132] + 1j * data[:,132:252]
            sequence = uwb[:, bin]
            transformed_sequence = transform(sequence, method)
            if invert == 1 and args.invert:
                transformed_sequence = -transformed_sequence

            yield transformed_sequence, gt, file
        
            
def evaluate(methods_file, data_folder):
    save_file = os.path.join(args.methods_folder, args.save_file)
    
    if not os.path.exists(save_file):
        save_df = pd.DataFrame(columns=[args.methods_file])
    else:
        save_df = pd.read_csv(save_file, index_col=0)
        
    scores_dict = {}
                
    total_score = 0
    count = 0
    for sequence, gt, file in tqdm(sequence_itr(methods_file, data_folder)):
        score = np.corrcoef(self_normalize(gt), sequence)[0][1]
        
        scores_dict[file] = score
        
        total_score += score
        count += 1
    
    # assign scores_dict to save_df column using file as index
    if not args.invert:
        save_df[args.methods_file + '_no_inv'] = pd.Series(scores_dict)
    else:
        save_df[args.methods_file] = pd.Series(scores_dict)
    save_df.to_csv(save_file)
    
    return total_score / count


def main():
    print(evaluate(os.path.join(args.methods_folder, args.methods_file), args.data_folder))

if __name__ == '__main__':
    main()
