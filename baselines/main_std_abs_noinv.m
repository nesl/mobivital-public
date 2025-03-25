fileID = fopen('tripod_std_abs_noinv.txt','w');
root_folder = "PATH-TO-DATASET/tripod";
listing = dir(root_folder);
for i = 3:length(listing)
    
    if ~(contains(listing(i).name, "userG") || contains(listing(i).name, "userH") || contains(listing(i).name, "userI") ||contains(listing(i).name, "userJ"))
        continue
    end
        
    dataset = csvread(root_folder + "/" + listing(i).name);

    uwb_mat = dataset(:, 13:13+120-1) + dataset(:, 13+120:end-2) * 1j;
	uwb_mat = abs(uwb_mat);

    object_inx = findbreathbin_std(uwb_mat);
    disp(object_inx(1))
    idx = object_inx(1);
    
    for ii = 1:length(object_inx)
        idx = object_inx(ii);
        proposed_method = "abs";
        if ~(idx <10)
            break;
        end      
    end
    
    sig_selected = self_normalize(uwb_mat(:, idx));
    
    order = 5;
    framelen= 101;
    sig_smoothed = sgolayfilt(sig_selected,order,framelen);

    % [pks,locs,w,p] = findpeaks(sig_smoothed, 'MinPeakProminence',0.1);
    % [pks2,locs2,w2,p2] = findpeaks(-sig_smoothed, 'MinPeakProminence',0.1);
    should_inv = 0;
    % if (mean(w2) < mean(w))
    %    should_inv = 1;
    % end
    
    fprintf(fileID,listing(i).name + "," + idx + "," + ...
       proposed_method + "," + should_inv + "\n");
end
