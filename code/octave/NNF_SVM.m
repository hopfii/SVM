function z = NNF_SVM(x1, x2)
    weight = get_global_weight();
    bias = get_global_bias();
    z = (weight * [x1;x2] + bias)