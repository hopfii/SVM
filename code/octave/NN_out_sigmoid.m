% function for neural net using the sigmoid function as activation function
% input parameters:
% x_in = input vector
% w = weight matrix
% N = size of output vector
function [ret, y] = NN_out_sigmoid(x_in, w, N)
    % get sizes of weight matrix, need columns later
    % get columns and rows
    [rows, cols] = size(x_in);
    [K,M] = size(w);
    % get number of rows of input vector
    input_dim = rows;
    % add bias row to x
    y = ones(M, 1);
    % copy values from x_in to x
    y (2: input_dim + 1) = x_in;
    % iterate through layers and consecutively calculate outputs
    for k=2+input_dim : M
        s = w(:, k)' * y;
        % activation function is sigmoid
        y(k) = Sigmoid(s);
    end
    % return the output vector as determined by N
    ret = y(end-N+1 : end);
end




