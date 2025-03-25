function [should_inv, res2, res1] = should_be_inverted(sig_selected)
    order = 5;
    framelen= 101;
    sig_smoothed = sgolayfilt(sig_selected,order,framelen);

    [pks,locs,w,p] = findpeaks(sig_smoothed, 'MinPeakProminence',0.1);
    [pks2,locs2,w2,p2] = findpeaks(-sig_smoothed, 'MinPeakProminence',0.1);
    should_inv = 0;
    
    res2 = mean(w2);
    res1 = mean(w);
    if (mean(w2) < mean(w))
       should_inv = 1;
    end
end

