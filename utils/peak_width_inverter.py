from scipy.signal import savgol_filter
from scipy.signal import find_peaks, peak_widths
import numpy as np
def invert_detector(seq):
    seq = savgol_filter(seq, window_length=101, polyorder=5)
    peaks, _ = find_peaks(seq, prominence=0.1)
    w1, _,_,_ = peak_widths(seq, peaks)
    peaks_inv, _ = find_peaks(-1*seq, prominence=0.1)
    w2, _,_,_ = peak_widths(-1*seq, peaks_inv)
    if len(w1) > 0 and len(w2) > 0:
        if np.mean(w1) > np.mean(w2): # peaks are narrower when inverted
            # we should invert
            return 1
  
    return 0

############
# Sample usage
if __name__ == '__main__':
    from training.utils.model_utils import generate_sequences
    data_folder = "data"

    with open(f'{data_folder}/training_breath_data.npy', 'rb') as train_file:
        X_train = np.load(train_file)
        y_train = np.load(train_file)

    with open(f'{data_folder}/testing_breath_data.npy', 'rb') as test_file:
        X_test = np.load(test_file)
        y_test = np.load(test_file)

    p_th = 0.85

    above_thresh_seq_train, below_thresh_seq_train = generate_sequences(X_train, y_train, corr_threshold=p_th)

    X_train = np.array(np.concatenate((above_thresh_seq_train, below_thresh_seq_train), axis=0))
    y_train = np.array([0]*len(above_thresh_seq_train) + [1]*len(below_thresh_seq_train))

    # 1 means a signal needs to be inverted
   
    invert_prob = []
    for each in X_train:  
        invert_prob.append(invert_detector(each))
        y_pred = np.array(invert_prob)
    
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_train, y_pred))
