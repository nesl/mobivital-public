function [object_inx] = findbreathbin(iqmat)

    %iqmat: (fs*30, bincnt)
    sig_baseband_dec = iqmat;

    % 30 seconds dec bg
    breath_win = 30;
    %breath_win_samples = breath_win * fps;
    %sig_baseband_dec_bg = sig_baseband_dec - mean(sig_baseband_dec);

    %smooth
    smooth_win = 16;
    for i = 1 : size(sig_baseband_dec,2)
        sig_smooth(:,i) = smooth(sig_baseband_dec(: ,i), smooth_win);
    end

    % clutters suppression, exponential average 
    clutter_coef = 0.9;

    % DC remove
    sig_dc_remove = detrend(sig_smooth);

    % initial sig_bg
    sig_bg = sig_dc_remove;
    for i = 1 : size(sig_dc_remove, 2)
        for j = 2 : size(sig_dc_remove, 1)
            sig_bg( j,i) = sig_dc_remove( j - 1 ,i) * (1 - clutter_coef) + sig_dc_remove(j, i);
        end
    end

    sig_bg_remove = sig_dc_remove -  sig_bg;
    fft_half_len = round(size(sig_bg_remove, 1)/2);
    sig_fft = abs(fft(sig_bg_remove(2:fft_half_len, :)));

    param.guard_cells = [0 5];
    param.training_cells = [0 10];
    param.threshold = 1;
    param.alg = 'CA-CFAR';

    cfar_out = detector_CFAR_v26(sig_fft, param);
    [pks,locs_y,object_inx] = peaks2(cfar_out,'SortStr','descend');

    % disp(object_inx(1))
    if length(object_inx) < 1
        fprintf('[no object]');
        return;
    end

end
