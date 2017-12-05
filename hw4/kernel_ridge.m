load auto_mpg_test.csv;
load auto_mpg_train.csv;

num_train = size(auto_mpg_train, 1);
num_test = size(auto_mpg_test, 1);
K = zeros(num_train, num_train);

predictions = zeros(num_test, 1);
lambda = 1;

for kernel_type = 1 : 4
    
    for i = 1 : num_train
        K(i, :) = kernel(repmat(auto_mpg_train(i, 2:8), num_train, 1), auto_mpg_train(:, 2:8), kernel_type);
    end
    
    coefficient = auto_mpg_train(:, 1)' / (K + lambda * eye(num_train));

    K_x = zeros(num_train, 1);
    for i = 1 : num_test
        predictions(i) = coefficient * kernel(repmat(auto_mpg_test(i, 2:8), num_train, 1), auto_mpg_train(:, 2:8), kernel_type);
    end

    RMSE = sqrt(mean((auto_mpg_test(:, 1) - predictions).^2));
    fprintf('RMSE on kernel type %d: %f\n', kernel_type, RMSE);    
end

