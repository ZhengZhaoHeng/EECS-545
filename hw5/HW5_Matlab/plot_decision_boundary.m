function plot_decision_boundary(W, X, y)

    % Set min and max values and give it some padding
    x1_min = min(X(:,1)) - 0.5;
    x1_max = max(X(:,1)) + 0.5;
    x2_min = min(X(:,2)) - 0.5;
    x2_max = max(X(:,2)) + 0.5;
    h = 0.01;
    
    %  Generate a grid of points with distance h between them
    [xx1, xx2] = meshgrid(x1_min:h:x1_max, x2_min:h:x2_max);
    
    % Predict the function value for the whole grid
    num_points = size(xx1, 1) * size(xx1,2);
    [~, Z] = prediction(W, [reshape(xx1,[num_points, 1]), reshape(xx2,[num_points, 1])]);
    Z = reshape(Z, size(xx1));
    

    contourf(xx1, xx2, Z);
    colormap('jet');
    hold on;
    scatter(X(y==0, 1), X(y==0, 2), 30, 'yellow', 'filled');
    scatter(X(y==1, 1), X(y==1, 2), 30, 'green', 'filled');
    

end

