function [ out ] = h(data, w)
%H �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

out = 1 ./ (1 + exp(-w * data'));

end

