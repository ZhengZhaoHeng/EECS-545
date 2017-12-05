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
training_acc_sgd = [];
test_acc_sgd = [];
for i = 1 : num_iter
    lr = ita / ( 1 + i * ita);
    for j = randperm(num_train)
        training_pred_single = digits_training_data(j, :) * w + b;
        indicators = (digits_training_labels(j, :) * training_pred_single) < 1;
        w_grad = 1.0 / num_train * w - (C * indicators * digits_training_labels(j, :) * digits_training_data(j, :))';
        b_grad = - C * indicators * digits_training_labels(j, :);
        w = w - lr * w_grad;
        b = b - lr * b_grad;
    end
    training_pred = digits_training_data * w + b;
    training_pred = ((training_pred > 0) - 0.5) * 2;
    test_pred = digits_test_data * w + b;
    test_pred = ((test_pred > 0) - 0.5) * 2;
    training_acc_sgd = [training_acc_sgd sum(training_pred == digits_training_labels) / num_train];
    test_acc_sgd = [test_acc_sgd sum(test_pred == digits_test_labels) / num_test];
    if mod(i, 1000) == 0
        training_acc_sgd(i)
    end
end