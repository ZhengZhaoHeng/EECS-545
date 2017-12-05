load synthetic.csv;

X = synthetic(:, 1);
Y = synthetic(:, 2);

w_mm = zeros(size(X, 2), 1);
b_mm = 0;
epsilon = 1e-8;

while 1
    r = Y - w_mm'*X - b_mm;
    C = 1 ./ sqrt(1 + r.^2);
    [w_temp, b_temp] = weighted_least_square(X, Y, C);
    if (norm(w_temp - w_mm) + abs(b_temp - b_mm)) < epsilon
        break;
    end
    w_mm = w_temp;
    b_mm = b_temp;
end

[w_ols, b_ols] = weighted_least_square(X, Y, ones(size(X, 1), 1));

plot_X = min(X) : 0.01 : max(X);
figure(1);
hold on;
scatter(X, Y);
plot(plot_X, 10 * plot_X + 5, '-', 'Linewidth', 2);
plot(plot_X, w_mm * plot_X + b_mm, '--', 'Linewidth', 2);
plot(plot_X, w_ols * plot_X + b_ols, '-.', 'Linewidth', 2);
legend('Data', 'True Line', 'Robust Linear Regression', 'Ordinary Least Square');

