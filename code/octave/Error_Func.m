% Error function that is optimized with batch learning algorithms
% Assumption: each column of x contains an input pattern
% Input parameters:
% x = the input vector
% W = the weight matrix
% D = the expected output values
% N = the size if the output vector
% Return values:
% E = the total error for this combination of parameters
function E = Error_Func(x, W, D, N)
    % initalize total error
    E = 0;
    % get number of columns of x
    [rows, cols] = size(x);
    % for all input patterns
    for k=1:cols
        % perform forward propagation to get y
        [ret, y] = NN_out_sigmoid(x(:,k), W, N);
        % calculate single error for this input pattern
        e_k  = sum(Single_error(ret, D(:,k)));
        % add error to total Error
        E = E + e_k;
    end
