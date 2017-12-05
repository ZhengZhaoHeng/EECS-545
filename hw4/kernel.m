function ip = kernel(u, v, type)
%KERNEL 此处显示有关此函数的摘要
%   此处显示详细说明

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

