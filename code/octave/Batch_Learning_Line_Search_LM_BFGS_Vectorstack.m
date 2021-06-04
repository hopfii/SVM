% Function for batch learning with limited memory BFGS
% Input parameters:
% X = the input values
% D = the expected outcome values
% T = the topology matrix
% max_epochs = the maximum number of epochs before the algorithm terminates
% eta = the starting eta value for optimal step size calculation (will determine delta in golden section)
% N = size of output vector
% eps = value for early termination checks
% f = function to opimize
% W = initial weight matrix
% Return values:
% W = the optimized weight matrix
% sqr_errors = vector containing the total errors over the epochs
% g = the number of iterations that algorithm ran before terminating
function [W, sqr_errors, g] = Batch_Learning_Line_Search_LM_BFGS_Vectorstack(X, D, T, max_epochs, eta, N, eps, f, W)
    % get measurements of topology matrix
    [M,K] = size(T);
    % initialize epoch counter
    g = 0;
    % initialize vector to save error
    sqr_errors = [];
    % set m to 20
    m = 20;
    % Initialize t (iteration counter)
    t = 0;
    % get number of cols of x
    [rows, cols] = size(X);
    % create M x K x m matrix to hold s-vectors
    s_mat = zeros(M * K, m);
    % create M x K x matrix to hold q-vectors
    q_mat = zeros(M * K, m);
    % storing alphas here
    alphas = [];
    % while maximum number of epochs is not reached
    while g <= max_epochs
            % initialize batch gradient G
            G = zeros(M,K);
            % for each pattern (column, in our case) of input values
            E = 0;
            for k=1:cols
                % perform forward propagation to get y
                [ret, y] = NN_out_sigmoid(X(:,k), W, N);
                % perform backpropagation to get gradients
                [gradients, deltas] = Back_propagation(y, D(:,k),N, W, T);
                e_k  = sum(Single_error(ret, D(:,k)));
                % add error to total Error for this epoch
                E = E + e_k;
                % add gradient to batch gradient
                G = G + gradients;
            end
            % turn G into vector
            G = G(:);
            % d is negative batch Gradient
            d = (-1) * G;
            % for 1 to minimum between t and m
            for i=1:min(t,m)
                % calculate alpha_i
                alpha_i = (s_mat(:,i)'*d)/(s_mat(:,i)'*q_mat(:,i));
                % add alpha_i to alphas at index i
                alphas(:, i) = alpha_i;
                % calculate d
                d = d - alpha_i * q_mat(:,i);
            end
            % if t > 0
            if t > 0
                % calculate d
                d = ((s_mat(:,1)'*q_mat(:,1))/(q_mat(:,1)'*q_mat(:,1))) * d;
            end
            % for minimum between t and m counted down to 1
            for i = min(t, m):-1:1
                % calculate beta
                beta = (q_mat(:,i)'*d)/(q_mat(:,i)'*s_mat(:,i));
                % get alpha_i (calculated in loop above)
                alpha_i = alphas(:, i);
                % calculate d
                d = d + (alpha_i - beta)*s_mat(:,i);
            end
            % perform ring buffer shift (move all columns one place to the right) for q and s
            s_mat = circshift(s_mat,1,2);
            q_mat = circshift(q_mat,1,2);
            % add total error for this epoch to error vector
            sqr_errors = [sqr_errors, E];
            % turn d back into matrix for golden section
            d_sec = reshape(d,M,K);
            % get eta_dach with golden section line search
            eta_dach = Golden_section_error_wrapper(X, d_sec, eta, f, D, W, N);
            % calculate step (step size * direction) and save at first index of s_mat
            s_mat(:,1) = eta_dach * d;
            % initalize batch gradient with zeros
            G_q = zeros(M,K);
            % turn W into vector
            W = W(:);
            % update W
            W = W + s_mat(:,1);
            % turn W back into matrix
            W = reshape(W,M,K);
            % just to be safe, multiply with topology
            W = W .* T;
            % for all columns of the input vector
            for k=1:cols
                % perform forward propagation to get y
                [ret, y] = NN_out_sigmoid(X(:,k), W, N);
                % perform backpropagation to get gradients, add to batch gradient
                [gradients, deltas] = Back_propagation(y, D(:,k),N, W, T);
                % add gradient to new batch gradient for q-calculation
                G_q = G_q + gradients;
            end
            % turn G_q into vector
            G_q = G_q(:);
            % q_mat at the first index is batch gradient after step - batch gradient before step
            q_mat(:,1) = G_q - G;
            % increase epoch counter
            g = g + 1;
            % terminate if G smaller than eps
            if (norm(G) < eps)
                break;
            end
    end
