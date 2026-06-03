desiredFs = 50;
root_folder = "YOUR_PATH/0418";
listing = dir(root_folder);
for i = 3:length(listing)
    folder = listing(i).name;
    if contains(folder, "archived")
        continue
    end
    fprintf("Entering Folder: " + folder + "\n");
    
    inner_listing = dir(root_folder + "/" + folder);
    for j = 1:length(inner_listing)
        if contains(inner_listing(j).name, "_realsense.csv")
            session_name = erase(inner_listing(j).name, "_realsense.csv");
            disp("Preprocessing:" + folder +"/"+session_name);
            session_data_preprocess(root_folder, folder, session_name, desiredFs)
        end 
    end
end
