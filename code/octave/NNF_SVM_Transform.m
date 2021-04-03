function z = NNF_SVM_Transform(x1, x2)
    x = Dimensionality_Transform([x1, x2])';
    weight = get_global_weight();
    bias = get_global_bias();
    z = (weight * x + bias);