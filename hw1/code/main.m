X = double(rgb2gray(imread('harvey-saturday-goes7am.jpg')));
[U, S, V] = svd(X);

k = [2 10 40];

for i=1:3
    app_x = zeros(size(X));
    sum_num = 0;
    for j = 1:k(i)
        app_x = app_x + S(j, j) * U(:, j) * V(:, j)';
        sum_num = sum_num + 1 + size(U(:, j)) + size(V(:, j));
    end
    s = sprintf('%d_app.jpg', k(i));
    imwrite(app_x / 256, s);
    ratio = norm(X - app_x, 'fro') / norm(X, 'fro');
    fprintf('Top %d approximation, error: %f\n', k(i), ratio);
    fprintf('numbers to store: %d\n', sum_num(1));
end