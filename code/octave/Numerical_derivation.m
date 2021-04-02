% function for the numerical derivation of gradients for networks with logistical sigmoid activation function
% input parameters:
% x_in = input values
% d = class value
% W = weight matrix
% h = parameter for (small compared to W values, but not too small!)
% N = size of output vector
% T = the topology matrix
% return values:
% gradients = matrix with the calculated gradients
function gradients = Numerical_derivation(x_in, d, W, h_factor, N, T)
    % get size of weight matrix
    [K,M] = size(W);
    % set h to h_factor of the maximum value of W
    h = max(max(W)) * h_factor;
    % do forward propagation to get y of W
    result_w = NN_out_sigmoid(x_in, W, N);
    % initialize gradient matrix with ones (or zeros, does not matter)
    gradients = ones(K,M);
    % loop over all rows
    for i=1:K
        % loop over all columns
        for j=1:M
            % create a matrix of zeros with same size as W
            H = zeros(K,M);
            % set value in matrix H at index ij to 1
            H(i,j) = 1;
            % do a forward propagation with W + h1ij as weight
            result_loop = NN_out_sigmoid(x_in, W + (h*H), N);
            % calculate gradient at index ij with formula from slides (e(W + h1ij) - e(W))/H)
            grad = (Single_error(result_loop, d) - Single_error(result_w, d))/h;
            % put gradient into matrix at index ij
            gradients(i,j) = grad;
        end
    end
    % multiply component-wise with toplogy matrix to get properly "shaped" weight matrix
    gradients = gradients .* T;
