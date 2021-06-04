% assumption: X with each training pattern in separate colum ([x_11,x_21,x_31;x_12,x_22,x_32]
% and D in one row ([1,1,0])
% function for calculating optimal weight matrix with stochastic gradient descent
% Input parameters:
% X = input value vector/matrix
% D = class vector/matrix
% T = Topology matrix
% max_epochs = maximum number of epochs the algorithm should take
% eta = starting learning parameter (will be used as starting delta of golden section)
% N = size of output vector of NN (information could also be gained with rows(D) most likely)
% eps = acceptable total error, if an epoch manages a total error below this, the algorithm can finish early
% f = the function to optimize
% return values:
% W = the optimized weight matrix
% sqr_errors = vector with the squared errors over the epochs
% g = number of epochs until algorithm finished (in case of early termination)
function [W, sqr_errors, g] = Batch_Learning_Line_Search_SGD(X, D, T, max_epochs, eta, N, eps, f)
    % get measurements of topology matrix
    [M,K] = size(T);
    % randomly initialize W (with size of topology matrix)
    W = randn(M,K);
    % multiply W with T components-wise (will put zeros in all places where no connection between neurons is supposed to be)
    W = W .* T;
    % initialize epoch counter
    g = 0;
    % initialize vector to save error
    sqr_errors = [];
    % while maximum number of epochs is not reached
    while g <= max_epochs
        % initialize batch gradient G
        G = zeros(M,K);
        % initialize total error for this epoch
        E = 0;
        % get number of cols of input patterns for loop below
        [rows, cols] = size(X);
        % for each pattern (column, in our case) of input values
        for k=1:cols
            % perform forward propagation to get y
            [ret, y] = NN_out_sigmoid(X(:,k), W, N);
            % perform backpropagation to get gradients
            [gradients, deltas] = Back_propagation(y, D(:,k),N, W, T);
            % calculate single error of this input pattern and weight matrix
            e_k  = sum(Single_error(ret, D(:,k)));
            % add error to total Error for this epoch
            E = E + e_k;
            % add gradient batch gradient
            G = G + gradients;
        end
        % add total error for this epoch to error vector
        sqr_errors = [sqr_errors, E];
        % get eta_dach with golden section line search
        eta_dach = Golden_section_error_wrapper(X, -G, eta, f, D, W, N);
        % update W
        W = W - eta_dach * G;
        % just to be safe, multiply with topology
        W = W .* T;
        % increase epoch counter
        g = g + 1;
        % terminate if G smaller than eps
        if norm(G) < eps
            break;
        end
    end