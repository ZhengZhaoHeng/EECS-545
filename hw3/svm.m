rng(34132);

training_label = label(1:500, :);
training_data = diabetesscale(1:500, :);

constant = linspace(0.1, 2, 20);
losses = [];

for i = 1 : 20
    model = fitcsvm(training_data, training_label, 'KernelFunction', 'linear', 'KernelScale', 1, 'BoxConstraint', constant(i), 'CrossVal', 'on', 'KFold', 5);
    loss = kfoldLoss(model);
    losses = [losses loss];
end
[value, idx] = min(losses);
best_c = constant(idx);
model = fitcsvm(training_data, training_label,  'KernelFunction', 'linear', 'KernelScale', 1, 'BoxConstraint', best_c);
p = predict(model, training_data);
sum(p==training_label) / 500