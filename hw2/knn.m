% load mnist_data;

test_size = 100;
training_size = size(train, 1);
k = 13;

%test_set = test(randsample(size(test, 1), test_size), :);
%distance = zeros(test_size, training_size);
%cputime
% for i = 1 : test_size
%     for j = 1 : training_size
%         distance(i, j) = norm(test_set(i, 2:785) - train(j, 2:785), 2);
%     end
% end
%cputime

[num val] = sort(distance, 2);
count = zeros(test_size, 10);
class = zeros(test_size, 1);
for i = 1 : test_size
    for j = 1 : k
        if train(val(i, j), 1) == 0
            count(i, 10) = count(i, 10) + 1;
        else
            count(i, train(val(i, j), 1)) = count(i, train(val(i, j), 1)) + 1;
        end
        [value, index] = max(count(i, :));
        if index == 10
            class(i) = 0;
        else
            class(i) = index;
        end
    end
end

class_internal = knnclassify(test_set(:, 2:785), train(:, 2:785), train(:, 1), k, 'euclidean');
sum(class_internal == class)
accuracy = sum(class == test_set(:, 1)) / test_size;
fprintf('Accuracy: %.2f%%\n', accuracy * 100);