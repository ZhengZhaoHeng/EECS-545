function [ loss ] = getLoss(data, label, w)
%getLoss 此处显示有关此函数的摘要
%   此处显示详细说明

h_w = h(data, w)';

loss = sum(-label .* log(h_w) - (1 - label) .* log(1 - h_w)); 

end