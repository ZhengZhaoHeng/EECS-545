% load digits_training_data.csv;
% load digits_training_labels.csv;
% load digits_test_data.csv;
% load digits_test_labels.csv;

% digits_training_labels = digits_training_labels - 8;
% digits_test_labels = digits_test_labels - 8;

dim = size(digits_training_data, 2);
w = zeros(dim, 1);
num_train = size(digits_training_data ,1);
num_test = size(digits_test_data, 1);
b = 0;
num_iter = 10000;
C = 3;
ita = 0.001;
training_acc_batch = [];
test_acc_batch = [];
for i = 1 : num_iter
    lr = ita / ( 1 + i * ita);
    training_pred = digits_training_data * w + b;
    indicators = (digits_training_labels .* training_pred) < 1;
    w_grad = w - (C * sum(repmat(indicators .* digits_training_labels, 1, dim) .* digits_training_data))';
    b_grad = - C * sum(indicators .* digits_training_labels);
    w = w - lr * w_grad;
    b = b - lr * b_grad;
    training_pred = digits_training_data * w + b;
    training_pred = ((training_pred > 0) - 0.5) * 2;
    test_pred = digits_test_data * w + b;
    test_pred = ((test_pred > 0) - 0.5) * 2;
    training_acc_batch = [training_acc_batch sum(training_pred == digits_training_labels) / num_train];
    test_acc_batch = [test_acc_batch, sum(test_pred == digits_test_labels) / num_test];
    if mod(i, 1000) == 0
        i
    end
end