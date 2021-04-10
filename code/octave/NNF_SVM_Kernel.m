function z = NNF_SVM_Kernel(x1, x2)
    bias = get_global_bias();
    alpha = get_global_alpha();
    y = get_global_y();
    x = get_global_x();
    k_func = get_global_kernel_func();
    polysum = 0;
    for i=1:max(size(x))
        polysum = polysum + alpha(i) * y(i) * feval(k_func, [x1,x2], x(i,:));
    end
    z =  sign(polysum + bias);
    % z = sign(sum(alpha * y * feval(k_func, [x1,x2], x) + bias));