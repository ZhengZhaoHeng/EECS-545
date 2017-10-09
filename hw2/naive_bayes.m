data_path = './spam_classification_matlab/';
training_size = 200;

vec_file_name = sprintf('DENSE.TRAIN.X.%d', training_size);
label_file_name = sprintf('DENSE.TRAIN.Y.%d', training_size);
%vec_file_name = 'DENSE.TRAIN.X';
%label_file_name = 'DENSE.Y.TRAIN.Y';
test_vec_file_name = 'DENSE.TEST.X';
test_label_file_name = 'DENSE.TEST.Y';
token_file = 'TOKENS_LIST';

num_tokens = 1448;

document_matrix = csvread(strcat(data_path, vec_file_name));
document_labels = csvread(strcat(data_path, label_file_name));
document_labels = (document_labels + 3) ./2; % 1: not spam, 2: spam
test_matrix = csvread(strcat(data_path, test_vec_file_name));
test_labels = csvread(strcat(data_path, test_label_file_name));
token_list = textscan(fopen(strcat(data_path, token_file)), '%d %s');

training_size = size(document_labels, 1);
num_tokens = size(document_matrix, 2);
test_size = size(test_labels, 1);

% Learning prior probilities
p_w = ones(num_tokens, 2);

for l = 1 : 2
    for i = 1 : num_tokens
        for j = 1:training_size
            if document_labels(j) == l
                p_w(i, l) = p_w(i, l) + document_matrix(j, i);
            end
        end
    end
    p_w(:,l) = p_w(:,l) ./ sum(p_w(:,l));
end

p_y = log([sum(document_labels == 1), sum(document_labels == 2)] / size(document_labels, 1));
p_w = log(p_w);

indicative = p_w(:,2) - p_w(:,1);
[num val] = sort(indicative, 'descend');
for i = 1 : 5
    fprintf('%d %s %f\n', val(i), char(token_list{2}(val(i))), num(i));
end

% classifying


p_d = zeros(test_size, 2);

class = zeros(test_size, 1);

for i = 1 : test_size
    for l = 1 : 2
        for j = 1 : num_tokens
            p_d(i, l) = p_d(i, l) + test_matrix(i, j) * p_w(j, l);
        end
    end
    if (p_d(i, 1) + p_y(1) > p_d(i, 2) + p_y(2))
        class(i) = -1;
    else
        class(i) = 1;
    end
end

err_rate = sum(class ~= test_labels) / test_size;
fprintf('error rate: %.3f%%\n', err_rate * 100);


