function [ loss ] = getLoss(data, label, w)
%getLoss �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

h_w = h(data, w)';

loss = sum(-label .* log(h_w) - (1 - label) .* log(1 - h_w)); 

end