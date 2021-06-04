% Function for batch learning with conjugate gradient descent
% Input parameters:
% X = the input values
% D = the expected outcome values
% T = the topology matrix
% max_epochs = the maximum number of epochs before the algorithm terminates
% eta = the starting eta value for optimal step size calculation (will determine delta in golden section)
% N = size of output vector
% eps = value for early termination checks
% f = function to opimize
% alpha = alpha value (used in conjuage gradient descent algorithm)
% W = the initial weight matrix
% Return values:
% W = the optimized weight matrix
% sqr_errors = vector containing the total errors over the epochs
% g = the number of iterations that algorithm ran before terminating
function [W, sqr_errors, g] = Batch_Learning_Line_Search_CGD_Vectorstack(X, D, T, max_epochs, eta, N, eps, f, alpha, W)
    % get measurements of topology matrix
    [M,K] = size(T);
    [rows, cols] = size(X);
    % initialize epoch counter
    g = 0;
    % initialize vector to save error
    sqr_errors = [];
    % initalize variable for remembering previous batch gradient G
    G_old = 0;
    % initalize d (step direction)
    d = 0;
    % while maximum number of epochs is not reached
    while g <= max_epochs
            % intiailize batch gradient G
            G = zeros(M,K);
            % for each pattern (column, in our case) of input values
            E = 0;
            % for each pattern (column, in our case) of input values
            for k=1:cols
                % perform forward propagation to get y
                [ret, y] = NN_out_sigmoid(X(:,k), W, N);
                % perform backpropagation to get gradients
                [gradients, deltas] = Back_propagation(y, D(:,k),N, W, T);
                % calculate Single error for input pattern k
                e_k  = sum(Single_error(ret, D(:,k)));
                % add error to total Error for this epoch
                E = E + e_k;
                % add gradient to batch gradient
                G = G + gradients;
            end
            % turn G into vector
            G = G(:);
            if G_old == 0
                G_old = G;
            end
            % turn d into vector
            d = d(:);
            % calculate d
            d = - G + ((norm(G)^2)/(norm(G_old)^2)) * d;
            % calculate angle
            angle = (G .* d)/((norm(G) * norm(d)));
            % if angle is bigger than - alpha, reset d to G
            if angle > - alpha
                d = - G;
            end
            % add total error of epoch to error vector
            sqr_errors = [sqr_errors, E];
            % transform d back into matrix for golden section search
            d = reshape(d,M,K);
            % get eta_dach with golden section line search
            eta_dach = Golden_section_error_wrapper(X, d, eta, f, D, W, N);
            % remember batch gradient for next epoch
            G_old = G;
            % update W
            W = W + eta_dach * d;
            % just to be safe, multiply with topology
            W = W .* T;
            % increase epoch counter
            g = g + 1;
            % terminate if norm(G) smaller than eps
            if (norm(G) < eps)
                break;
            end
        end