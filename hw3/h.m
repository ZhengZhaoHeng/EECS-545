function [ out ] = h(data, w)
%H 此处显示有关此函数的摘要
%   此处显示详细说明

out = 1 ./ (1 + exp(-w * data'));

end

