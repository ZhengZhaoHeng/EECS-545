
w = [0 0 0];
train_data = [ones(size(train_x0)) train_x0 train_x1];
train_label = train_y;

test_data = [ones(size(test_x0)) test_x0 test_x1];
test_label = test_y;

loss = zeros(2, 1);
loss(1) = getLoss(train_data, train_label, w);
loss(2) = loss(1) + 1;

eps = 1e-8;
training_losses = [loss(1)];
test_losses = [getLoss(test_data, test_label, w)];

while (abs(loss(1) - loss(2)) > eps)
    loss(1) = loss(2);

    h_w = h(train_data, w)';
    delta_w = sum(-train_data .* repmat(train_label - h_w, 1, 3));
    hessian_w = zeros(3, 3);
    for i = 1 : size(train_data, 1)
        hessian_w = hessian_w + train_data(i, :)' * train_data(i, :) * h_w(i) * (1 - h_w(i));
    end
    w = w - (inv(hessian_w) * delta_w')';
    loss(2) = getLoss(train_data, train_label, w);
    test_losses = [test_losses getLoss(test_data, test_label, w)];
    training_losses = [training_losses, loss(2)];
end

fprintf('Takes %d iterations to converge.\n', size(training_losses, 2));
fprintf('%f\n', w);
x = 1 : size(training_losses, 2);
plot(x, training_losses, x, test_losses);
legend('train', 'test');


