import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from utils.models import *
from utils.model_utils import self_normalize, sequence_transforms, softmax, generate_dataloader, generate_dataset

from tqdm import tqdm, trange
from einops import rearrange
from matplotlib import pyplot as plt
import pickle
import json

from collections import defaultdict
from mango.domain.distribution import loguniform
from mango import scheduler, Tuner

from joblib.externals.loky.backend.context import get_context

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open('data_dev/training_breath_data.npy', 'rb') as train_file:
    X_train = np.load(train_file)
    y_train = np.load(train_file)
    

with open(f'data_dev/testing_breath_data.npy', 'rb') as f:
    X_test = np.load(f)
    y_test = np.load(f)


def get_best_sequence(original_sequence, model, history_len, future_len, device):
    original_sequence = rearrange(original_sequence, "t c -> c t")
    sequences = []
    for s in original_sequence:
        sequences += sequence_transforms(s)
    sequences = np.array(sequences)
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
        inputs = torch.from_numpy(X).to(device)
        labels = torch.from_numpy(y).to(device)
        outputs = model(inputs)
    losses = []
    for sample in range(len(labels)):
        losses.append(np.corrcoef(outputs[sample].cpu().numpy(), labels[sample].cpu().numpy())[0,1])
    losses = np.array(losses)
    l = (1500 - history_len) // future_len
    losses = rearrange(losses, "(c l) -> c l", l=l)
    min_loss_idx = np.argmax(np.sum(losses, axis=1))
    best_corr = np.sum(losses[min_loss_idx])
    return sequences[min_loss_idx]
    

def inference(autoreg_model, X_uwb, y_breath, history_len, future_len, device):
    total_score = 0
    for i in range(len(y_breath)):
        best_sequence = get_best_sequence(X_uwb[i], autoreg_model, history_len, future_len, device)
        # convert the best sequence to features for invert model
        best_sequence = np.expand_dims(best_sequence, axis=0)
        ground_truth = np.corrcoef(self_normalize(y_breath[i]), best_sequence)[0][1]

        total_score += ground_truth

    return total_score / len(y_breath)


def train(model, train_dataloader, lr, epochs):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
    return model

'''
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

def train_on_device(device_id, seed, params, return_dict):
    # Set random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Select device
    device = torch.device(f'cuda:{device_id}')

    # Initialize data and model
    train_dataloader = generate_dataloader(
        X_train, y_train, params['batch_size'], 
        history_len=params['history_length'], 
        future_len=params['future_length'], 
        corr_threshold=0.9, shuffle=True
    )
    model = LSTMMultiStep(params['hidden_size'], params['num_layers'], params['future_length'])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(params['epochs']):
        running_loss = 0.0
        for data in train_dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    # Evaluate the model
    model.eval()
    score = inference(model, X_test, y_test, params['history_length'], params['future_length'], device)
    return_dict[seed] = score

import time
@scheduler.serial
def objective(**params):
    random_seeds = [0, 1, 2]
    # start time
    start_time = time.time()
    mp.set_start_method('spawn', force=True)  # Use spawn start method for CUDA compatibility
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    # Spawn a process for each seed
    for i, seed in enumerate(random_seeds):
        p = mp.Process(target=train_on_device, args=(i+1, seed, params, return_dict))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Calculate the average score from all seeds
    total_score = sum(return_dict.values()) / len(random_seeds)

    end_time = time.time()
    
    # Log results
    with open('params.txt', 'a') as f:
        f.write(str(params) + f" Score: {total_score}\n")
    print(str(params) + f" Score: {total_score}" + f" Time: {end_time - start_time}")

    return total_score
'''

@scheduler.serial
def objective(**params):
    epochs = 50
    random_seeds = [0, 1, 2]

    #total_score = 0
    epoch_dict = defaultdict(int)
    for seed in tqdm(random_seeds):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        train_dataloader = generate_dataloader(X_train, y_train, params['batch_size'], history_len=params['history_length'], future_len=params['future_length'], corr_threshold=0.9, shuffle=True)
        model = LSTMMultiStep(params['hidden_size'], params['num_layers'], params['future_length'])
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.MSELoss()

        for epoch in range(1, epochs+1):  # loop over the dataset multiple times
            running_loss = 0.0
            model.train()
            for data in train_dataloader:
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            if epoch % 5 == 0:
                model.eval()
                epoch_dict[epoch] += inference(model, X_test, y_test, params['history_length'], params['future_length'], device=device)

        min_epoch = min(epoch_dict, key=epoch_dict.get)
        total_score = epoch_dict[min_epoch]
        params['epochs'] = min_epoch
        
    with open('params.txt', 'a') as f:
        f.write(str(params) + f" Score: {total_score/len(random_seeds)}\n")
        
    print(str(params) + f" Score: {total_score/len(random_seeds)}")

    return total_score / len(random_seeds)


def main():
    conf_dict = dict(num_iteration=10, domain_size=10000, initial_random=3)

    param_space = dict(batch_size=[32, 64, 128],
                       hidden_size=range(256, 513, 32),
                       num_layers=[1, 2],
                       history_length=range(100, 401, 100),
                       future_length=range(25, 101, 25),
                       lr=loguniform(-5, 2),
                       #epochs=[50]
                    )    
    tuner = Tuner(param_space, objective, conf_dict)
    results = tuner.maximize()
    print(f'Optimal value of parameters: {results["best_params"]} and objective: {results["best_objective"]}')
    with open("optimal_params.json", "w") as outfile: 
        json.dump(results["best_params"], outfile)

if __name__ == '__main__':
    main()