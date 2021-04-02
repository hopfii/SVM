% helper function for Contour plots
% Input parameters:
% x1 = first input value
% x2 = second input value
% Return parameters:
% z = the calculation result
function z = NNF_NN_Sigmoid(x1, x2)
    % get value of global variable W
    W = Get_Global_W();
    % perform forward propagation through NN for result
    z = NN_out_sigmoid([x1; x2], W, 1);
end