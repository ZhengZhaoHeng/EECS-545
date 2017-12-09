function grad = sigmoid_grad(x)

    grad = sigmoid(x) * (1 - sigmoid(x));

end 
