function dot_prod = rbf_kernel(x, x_prime)
    rbf_gamma = get_global_gamma();
    dot_prod = exp(-rbf_gamma * norm(x - x_prime)^2);