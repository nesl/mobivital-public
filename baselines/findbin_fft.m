function [object_inx] = findbin_fft(uwb_mat)

hhi = zeros( size(uwb_mat, 2),1);

for bin = 1:1:size(uwb_mat, 2)
% fft analysis
    signal = uwb_mat(:,bin);
    signal = signal - mean(signal);

    Fs = 50;
    T = 1/Fs;             % Sampling period
    L = size(signal, 1); % Length of signal
    t = (0:L-1).*T;        % Time vector
    f = Fs*(0:(round(L/2)))/L;

    Y = fft(signal);
    P2 = abs(Y);  % P2 = abs(Y/L);
    P1 = P2(1:round(L/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);

    breath_engy = sum(P1(4:13));
    total_engy = sum(P1);

    hhi(bin) = breath_engy / total_engy;
end

[sorted_hhi, object_inx] = sort(hhi,'descend');

end

