% function for error backpropagation (unfinished)
% input parameters:
% y = the matrix with the activation values of the neurons of the neural net
% d = the predicted outputs in a vector
% N = size of the output vector
% w = the weight matrix used to produce y
% t = the topology matrix (may be unneccessary here)
% return values:
% gradients = the error gradients in the form of a vector where row k = gradient of neuron k
% deltas = the deltas in a vector, where row k = delta of neuron k
function [gradients, deltas] = Back_propagation(y, d, N, w, T)
    % get row and col size for weight matrix (need rows)
    [K,M] = size(w);
    % vector for remembering deltas
    deltas = ones(K,1);
    % get output from y
    out = y(end-N+1 : end);
    % calc delta for output neurons
    delta = 2 .* out .* (1 - out) .* (out - d);
    % add delta output to error vector
    deltas(end-N+1:end) = delta;
    % iterate backwards through weights and neurons, calc the deltas and add them to the errors
    for k=K-N:-1:1
        % delta_j = F_j'(s_j) * sum (w_jk * delta_k)
        delta_k = (w(k,:) * deltas) * y(k) * (1 - y(k));
        % add delta to delta vector
        deltas(k) = delta_k;
    end
    % calc the gradients
    gradients = y .* deltas' .*T;