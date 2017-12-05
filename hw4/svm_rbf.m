% load digits_training_data.csv;
% load digits_training_labels.csv;
% load digits_test_data.csv;
% load digits_test_labels.csv;

% digits_training_labels = digits_training_labels - 8;
% digits_test_labels = digits_test_labels - 8;

svm_model = fitcsvm(digits_training_data, digits_training_labels, 'KernelFunction', 'RBF', 'OptimizeHyperparameters', 'auto');
