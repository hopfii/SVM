function dot_prod = polynomial_kernel(x, x_prime)
    a = get_global_a_kernel();
    b = get_global_b_kernel();
    q = get_global_exponent_kernel();

    dot_prod = (b + a .* x_prime * x')^q;