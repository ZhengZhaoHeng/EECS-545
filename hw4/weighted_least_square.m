function [w, b] = weighted_least_square(X, Y, C)

X_tilde = X - C' * X / sum(C);
Y_tilde = Y - C' * Y / sum(C);

w = inv(X_tilde' * diag(C) * X_tilde) * X_tilde' * diag(C) * Y_tilde;
b = C' * Y / sum(C) - w' * (C'* X / sum(C));

end

