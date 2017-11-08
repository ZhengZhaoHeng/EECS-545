

training_label = label(1:500, :);
training_data = diabetesscale(1:500, :);
test_label = label(501:768, :);
test_data = diabetesscale(501:768, :);

constant = linspace(0.1, 2, 20);
losses = [];

for i = 1 : 20
    rng(42);
    model = fitcsvm(training_data, training_label, 'KernelFunction', 'linear', 'KernelScale', 1, 'BoxConstraint', constant(i), 'CrossVal', 'on', 'KFold', 5);
    loss = kfoldLoss(model);
    losses = [losses loss];
end
[value, idx] = min(losses);
best_c = constant(idx);
best_c_model = fitcsvm(training_data, training_label,  'KernelFunction', 'linear', 'KernelScale', 1, 'BoxConstraint', best_c);
pred = predict(best_c_model, test_data);
fprintf('Soft-Margin SVM Test Accuracy: %.2f%%\n', sum(pred==test_label) / 268 * 100);

hard_margin_model = fitcsvm(training_data, training_label, 'KernelFunction', 'linear', 'KernelScale', 1, 'BoxConstraint', 1e6);
pred = predict(hard_margin_model, test_data);
fprintf('Hard-Margin SVM Test Accuracy: %.2f%%\n', sum(pred==test_label) / 268 * 100);
