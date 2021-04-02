% assumption: X with each training pattern in separate colum ([x_11,x_21,x_31;x_12,x_22,x_32]
% and D in one row ([1,1,0])
% function for calculating optimal weight matrix with stochastic gradient descent
% input parameters:
% X = input value vector/matrix
% D = class vector/matrix
% T = Topology matrix
% max_epochs = maximum number of epochs the algorithm should take
% n = learning parameter (for mutating of weight matrix)
% N = size of output vector of NN (information could also be gained with rows(D) most likely)
% min_e = acceptable total error, if an epoch manages a total error below this, the algorithm can finish early
% return values:
% W = the optimized weight matrix
% sqr_errors = vector with the squared errors over the epochs
% g = number of epochs until algorithm finished (in case of early termination)
function [W, sqr_errors, g] = NN_learning_SGD(X, D, T, max_epochs, n, N, min_e)
    % get measurements of topology matrix
    [M,K] = size(T);
    % randomly initialize W (with size of topology matrix)
    W = randn(M,K);
    % multiply W with T components-wise (will put zeros in all places where no connection between neurons is supposed to be)
    W = W .* T;
    % initialize epoch counter
    g = 1;
    % initialize vector to save error
    sqr_errors = [];
    % get rows and columns
    [rows, cols] = size(X);
    % while maximum number of epochs is not reached
    while g <= max_epochs
        % initialize total error for this epoch
        E = 0;
        % for each pattern (column, in our case) of input values
        for k=1:cols
            % perform forward propagation to get y
            [ret, y] = NN_out_sigmoid(X(:,k), W, N);
            % perform backpropagation to get gradients
            [gradients, deltas] = Back_propagation(y, D(:,k),N, W, T);
            % transform weight matrix with gradients and learning parameter
            W = W - (n * gradients);
            % multiply W with T components-wise (will put zeros in all places where no connection between neurons is supposed to be)
            W = W .* T;
            % perform forwad propagation with new weight matrix
            [ret, y] = NN_out_sigmoid(X(:,k), W, N);
            % calculate square single error
            e_k  = sum(Single_error(ret, D(:,k)));
            % add error to total Error for this epoch
            E = E + e_k;
        end
        % add total error for this epoch to error vector
        sqr_errors = [sqr_errors, E];
        % increase epoch counter
        g = g + 1;
        % change in weight matrix is smaller than min_e -> Done!
        if E < min_e
            disp("Done early!");
            break;
        end
    end