function loss = cross_entropy_loss(t_hat, t)
    loss = -sum(t .* log(t_hat) + (1-t) .* log(1 - t_hat));
end