function [probs, predictions] = prediction(W, X)
% Input
% W: container map, containing all paramters for the network. It's
% similar to a dictionary in Python.
% X: matrix, input data

% Output
% probs: vector, output from sigmoid function
% predictions: vector, containing predictions for each data point

z_h1 = X * W('W_input_h1') + W('b_input_h1');
a_h1 = tanh(z_h1);

z_h2 = a_h1 * W('W_h1_h2') + W('b_h1_h2');
a_h2 = tanh(z_h2);

output = a_h2 * W('W_h2_output') + W('b_h2_output');
probs = sigmoid(output);

% assign 1 to 
predictions = zeros(size(output));
predictions(probs > 0.5) = 1;

end