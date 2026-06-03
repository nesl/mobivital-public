import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm, trange
from training.utils.model_utils import generate_dataloader
from training.utils.models import LSTMMultiStep

import argparse
import json
import os

seed = 1234
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epochs', type=int, default=200)
parser.add_argument('-l', '--lr', type=float, default=5e-4)
parser.add_argument('-b', '--batch_size', type=int, default=64)

parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--history_length', type=int, default=500)
parser.add_argument('--future_length', type=int, default=100)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--params_file', type=str, default='checkpoints/optimal_params.json')

parser.add_argument('--data_folder', type=str, default='./data_final')
parser.add_argument('--save_folder', type=str, default='./checkpoints')
parser.add_argument('--model_name', type=str, default='lstm_pred')

parser.add_argument('--mode', type=str, default='tripod')
parser.add_argument('-c', '--corr_threshold', type=float, default=0.9)
parser.add_argument('--top', action='store_true')
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

if args.params_file:
    with open(args.params_file, 'r') as f:
        params = json.load(f)
        for key, value in params.items():
            setattr(args, key, value)
            
print(args.__dict__)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

def get_loss(model, dataloader):
    criterion = nn.MSELoss()
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
    return running_loss / len(dataloader)


def train(model, train_dataloader, test_dataloader, lr, epochs, save_path):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_epoch = 0
    for epoch in trange(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        if epoch % 10 == 0:       
            print(f'Epoch:{epoch} loss: {(running_loss / len(train_dataloader)):.3f}')
        
    print('Finished Training')
    torch.save(model.state_dict(), save_path)


def main():
    with open(f'{args.data_folder}/training_breath_{args.mode}_data.npy', 'rb') as train_file:
        X_train = np.load(train_file)
        y_train = np.load(train_file)

    with open(f'{args.data_folder}/testing_breath_{args.mode}_data.npy', 'rb') as test_file:
        X_test = np.load(test_file)
        y_test = np.load(test_file)
        
    train_dataloader = generate_dataloader(X_train, y_train, batch_size=args.batch_size, history_len=args.history_length, future_len=args.future_length, corr_threshold=args.corr_threshold, top=args.top)
    test_dataloader = generate_dataloader(X_test, y_test, batch_size=args.batch_size, history_len=args.history_length, future_len=args.future_length, corr_threshold=args.corr_threshold, shuffle=False, top=args.top)

    model = LSTMMultiStep(args.hidden_size, args.num_layers, args.future_length)

    model.to(device)
    
    train(model, train_dataloader, test_dataloader, lr=args.lr, epochs=args.epochs, save_path=os.path.join(args.save_folder, f'{args.model_name}_{args.mode}_{args.corr_threshold}.pth'))
    

if __name__ == '__main__':
    main()
