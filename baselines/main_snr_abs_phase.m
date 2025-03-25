fileID = fopen('tripod_snr_abs+phase_preinv.txt','w');
root_folder = "PATH-TO-DATASET/tripod";
listing = dir(root_folder);
for i = 3:length(listing)
    
    if ~(contains(listing(i).name, "userG") || contains(listing(i).name, "userH") || contains(listing(i).name, "userI") ||contains(listing(i).name, "userJ"))
        continue
    end
        
    dataset = csvread(root_folder + "/" + listing(i).name);

    uwb_mat = dataset(:, 13:13+120-1) + dataset(:, 13+120:end-2) * 1j;
    uwb_phase = zeros(size(uwb_mat));
    for col = 1:120
        uwb_phase(:,col) = unwrap(phase(uwb_mat(:,col)));
    end

    uwb_mat = [abs(uwb_mat) uwb_phase];
    
    object_inx = findbin_fft(uwb_mat);
    idx = -1;

    for ii = 1:length(object_inx)
        temp_idx = object_inx(ii);
        sig_selected = self_normalize(uwb_mat(:, temp_idx));
        [should_inv, w2, w] = should_be_inverted(sig_selected);
        if should_inv
           continue % lose its candidacy
        else
           idx = temp_idx; 
           break
        end
    end
    if idx == -1
        disp("This should not happen! Check your code.")
    end
        
    proposed_method = "abs";
    if (idx > 120 && idx <= 240) 
        idx = idx - 120;
        proposed_method = "phase";
    end
    sig_selected = self_normalize(uwb_mat(:, idx));
    
    should_inv = 0;
    fprintf(fileID,listing(i).name + "," + idx + "," + ...
       proposed_method + "," + should_inv + "\n");
end
