function grad = tanh_grad(x)
    grad = 1 - tanh(x).^2;
end