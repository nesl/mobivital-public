function [data_array] = self_normalize(input_array)
    data_array = (input_array - min(input_array,[],"all")) / (max(input_array,[],"all") - min(input_array,[],"all"));
    