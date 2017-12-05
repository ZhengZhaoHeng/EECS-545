function ip = kernel(u, v, type)
%KERNEL �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

switch type
    case 1
        ip = (dot(u, v, 2) + 1).^2;
    case 2
        ip = (dot(u, v, 2) + 1).^4;
    case 3
        ip = (dot(u, v, 2) + 1).^8;
    case 4
        sigma = 1;
        ip = exp(-sum((u-v).^2, 2) / (2 * sigma^2));
end

end

