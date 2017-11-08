
w = [0 0 0];
train_data = [ones(size(train_x0)) train_x0 train_x1];
train_label = train_y;

loss = zeros(2, 1);
loss(1) = getLoss(train_data, train_label, w);
loss(2) = loss(1) + 1;

eps = 1e-8;

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
    fprintf('%f\n', loss(2));
end


