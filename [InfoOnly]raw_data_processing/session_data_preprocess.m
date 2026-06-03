function [] = session_data_preprocess(root_folder, folder, session_name, desiredFs)
    referenceFs = 50;
    exp_name = root_folder + "/" + folder + "/" + session_name;
    if contains(session_name, "stable")
        file_export_dir = "YOUR_DEST_PATH/stable/";
    elseif contains(session_name, "selfie")
        file_export_dir = "YOUR_DEST_PATH/selfie/"; % No longer valid
    else
        file_export_dir = "YOUR_DEST_PATH/tripod/";
    end
    disp(file_export_dir)

    %%% 
    imu_data = csvread(exp_name + "_imu.csv");
    imu_timestamp = imu_data(:,1);
    imu_data = imu_data(:,2:end);

    %%% 
    realsense_content = csvread(exp_name + "_realsense.csv");
    realsense_timestamp = realsense_content(:,1);
    ext_mat = realsense_content(:,2:end);
    realsense_length  = length(realsense_content);
    realsense_data = zeros(realsense_length, 6);
    for i = 1:1:realsense_length
        trans_mat = [ext_mat(i,1) ext_mat(i,2) ext_mat(i,3) ext_mat(i,4);
                     ext_mat(i,5) ext_mat(i,6) ext_mat(i,7) ext_mat(i,8);
                     ext_mat(i,9) ext_mat(i,10) ext_mat(i,11) ext_mat(i,12)
                     ext_mat(i,13) ext_mat(i,14) ext_mat(i,15) ext_mat(i,16)];
    %     disp(trans_mat)
        trans_mat = inv(trans_mat);
    %     disp(rotm2eul(trans_mat(1:3,1:3)))
    %     disp(trans_mat(1:3,4)')
    %     disp("*********")
        realsense_data(i,:) = [rotm2eul(trans_mat(1:3,1:3)) trans_mat(1:3,4)'];
    end

    %%% 
    neulog_data = csvread(exp_name + "_neulog.csv");
    neulog_timestamp = neulog_data(:,1);
    neulog_data = neulog_data(:,2:end);
    % breath_wave = neulog_data(:,2);
    % heart_wave = neulog_data(:,3);

    %%%  
    raw_iq = csvread(exp_name + "_uwb.csv");
    uwb_timestamp = raw_iq(:,1);
    uwb_data = raw_iq(:,2:end);
    [timesteps, iqs] = size(raw_iq);
    [uwb_data, uwb_timestamp] = resample(uwb_data, uwb_timestamp,referenceFs);
    % raw_iq = raw_iq(:,2:end);
    % uwb_data = uwb_data(:,1:iqs/2) + 1j.* uwb_data(:,1+iqs/2:iqs);
    % [uwb_data, uwb_timestamp] = resample(uwb_data, uwb_timestamp,referenceFs);


    %%% 
    start_time = max([uwb_timestamp(1), neulog_timestamp(1), imu_timestamp(1), realsense_timestamp(1)]);
    end_time = min([uwb_timestamp(end), neulog_timestamp(end), imu_timestamp(end), realsense_timestamp(end)]);
    if (end_time - start_time < 0)
        disp("You forgot to align the time when doing experiments!!!")
        return
    end


    timestamps_pool = neulog_timestamp(neulog_timestamp>=start_time & neulog_timestamp<=end_time);
    
    neulog_data = neulog_data(neulog_timestamp>=start_time & neulog_timestamp<=end_time,:);
%     [neulog_data, timestamps_pool] = resample(neulog_data, timestamps_pool, referenceFs);

    [ans, start_idx] = min(abs(timestamps_pool(1) - imu_timestamp));
    [imu_data, imu_timestamp] = resample(imu_data(start_idx:end,:), imu_timestamp(start_idx:end), referenceFs);

    [ans, start_idx] = min(abs(timestamps_pool(1) - realsense_timestamp));
    [realsense_data, realsense_timestamp] = resample(realsense_data(start_idx:end,:), realsense_timestamp(start_idx:end), referenceFs);


    idx = 1;
    fileidx = 0;
    data = [];
    while (idx < length(timestamps_pool)-10) % -10 to dirty-fix the out-of-bounds error
        [ans, nearest_uwb_frame_idx] = min(abs(uwb_timestamp - timestamps_pool(idx)));
        data_of_curr_step = [imu_data(idx,:) realsense_data(idx,:) uwb_data(nearest_uwb_frame_idx,:) neulog_data(idx,:)];
        if (uwb_timestamp(nearest_uwb_frame_idx) - timestamps_pool(idx) > 1/referenceFs ...
                ||  imu_timestamp(idx) - timestamps_pool(idx) > 1/referenceFs  ||  realsense_timestamp(idx) - timestamps_pool(idx) > 1/referenceFs )
            fprintf("Timestamp Misalignment Happened\n")
            disp(realsense_timestamp(idx) - timestamps_pool(idx) > 1/referenceFs)
        end
        data = [data; data_of_curr_step];
        
        if (timestamps_pool(idx+1) - timestamps_pool(idx) > 1/referenceFs + 0.01)
            if (length(data)>1000)
                filename = file_export_dir + session_name + "_" + string(fileidx) + ".csv";
                data = resample(data, desiredFs, referenceFs);
                writematrix(data, filename);
                fileidx = fileidx + 1;
                disp("TimeElapsed: " + string(timestamps_pool(idx) - timestamps_pool(1)) + " seconds.")
            end
            data = [];
        end
        idx = idx + 1;
    end
%     if (length(data)>1000)
%         [ans, nearest_uwb_frame_idx] = min(abs(uwb_timestamp - timestamps_pool(idx)));
%         data_of_curr_step = [imu_data(idx,:) realsense_data(idx,:) uwb_data(nearest_uwb_frame_idx,:) neulog_data(idx,:)];
%         data = [data; data_of_curr_step];
%         filename = file_export_dir + session_name + "_" + string(fileidx) + ".csv";
%         data = resample(data, desiredFs, referenceFs);
%         writematrix(data, filename);
%     end

end

