import numpy as np
import scipy
from scipy.fft import fft
from scipy.signal import detrend
from matplotlib import pyplot as plt


def calc_snr(seq):
    seq = detrend(seq)
    # plt.plot(seq)
    # plt.show()
    Fs = 50
    T = 1/Fs
    L = len(seq)
    t = np.arange(0,L) * T
    f = Fs*np.arange(0,L//2)/L
    Y = fft(seq)
    P2 = np.abs(Y) **2
    P1 = P2[0:L//2]
    sig_power = np.sum(P1[6:22]) # [0.2, 0.7Hz]
    no = np.sum(P1[1:]) - sig_power
    snr = 10*np.log10(sig_power/no)
    return snr

if __name__ == "__main__":
    with open('training_breath_data.npy', 'rb') as train_file:
        X_uwb_train = np.load(train_file)
        y_breath_train = np.load(train_file)
        y_heart_train= np.load(train_file)

    seq = abs(X_uwb_train[562,:,26])
    print(calc_snr(seq))
