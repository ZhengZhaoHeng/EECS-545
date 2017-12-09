% This is the master script for construting and training the neural
% networks. Your work is to implement the gradient descent. 
clear all;
clc;
load data;
rng(42);
%% Construct the network and initialize the parameters
input_dim = 2;
nodes_hidden_1 = 3;
nodes_hidden_2 = 3;
output_dim = 1;

keys = {'W_input_h1', 'b_input_h1', 'W_h1_h2', 'b_h1_h2', 'W_h2_output', 'b_h2_output'};
params = {rand(input_dim, nodes_hidden_1), rand(1, nodes_hidden_1),...
    rand(nodes_hidden_1, nodes_hidden_2), rand(1, nodes_hidden_2),...
    rand(nodes_hidden_2, output_dim), rand(1, output_dim)};
W = containers.Map(keys, params);

%% # Train using Gradient Descent
% training parameters
learning_rate = 0.005;
no_epochs = 2000;

for epoch = 1:no_epochs

%     % Forward propagation
    z_h1 = X * W('W_input_h1') + W('b_input_h1');
    a_h1 = tanh(z_h1);

    z_h2 = a_h1 * W('W_h1_h2') + W('b_h1_h2');
    a_h2 = tanh(z_h2);

    output = a_h2 * W('W_h2_output') + W('b_h2_output');

    probs = sigmoid(output);

    % Backward propagation. YOUR WORK GOES HERE. 
    % Hint: first implement the gradient for each parameter, then apply
    % gradient descent.
    
    %
    % your code......
    %
    
    E_theta = mean((-y + probs) .* a_h2);
    E_b4 = mean(-y + probs);
    
    E_h2 = repmat(-y +probs, 1, 3) .* repmat(W('W_h2_output')', 200, 1);
    E_w2 = zeros(3, 3);
    for i = 1 : 200
        E_w2 = E_w2 + 1.0 / 200 * (repmat(E_h2(i, :), 3, 1) .* ((1 - a_h2(i, :).^2)' * a_h1(i, :)));
    end
    E_b2 = mean(E_h2 .* (1 - a_h2 .^ 2));
        
    E_h1 = E_h2 * W('W_h1_h2');
    E_w1 = zeros(2, 3);
    for i = 1 : 200
        E_w1 = E_w1 + 1.0 / 200 * (repmat(E_h2(i, :), 2, 1)'.* ((1 - a_h1(i, :).^2)' * X(i, :)))';
    end
    E_b1 = mean(E_h1 .* (1 - a_h1.^2));
    
    W('W_h2_output') = W('W_h2_output') - learning_rate * E_theta';
    W('b_h2_output') = W('b_h2_output') - learning_rate * E_b4;
        
    W('W_h1_h2') = W('W_h1_h2') - learning_rate * E_w2;
    W('b_h1_h2') = W('b_h1_h2') - learning_rate * E_b2;
    
    W('W_input_h1') = W('W_input_h1') - learning_rate * E_w1;
    W('b_input_h1') = W('b_input_h1') - learning_rate * E_b1;
    
    % WORK ENDS HERE.
    % calculate and show the results every 100 iteration.
    loss = cross_entropy_loss(probs, y);    
  
    if mod(epoch, 100) == 0
        fprintf('Loss after epoch %i is %f \n', epoch, loss);
    end
end 
% plot_decision_boundary(W, X, y);