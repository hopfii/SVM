% function for calculating a weight matrix using a Madalnin neural network
% (using the pseudoinverse)
% input parameters:
% x = input values (as vector with k columns, with k = number of inputs)
% d = class values (vector with k columns)
% return values:
% weight = the calcualted weight matrix
function weight = Madalin_nn(x, d)
    % get the row and column size of x
    [M, K] = size(x);
    % phi is x
    phi = x;
    % add another row where every column is filled with ones for thresholds
    phi(M+1,:)=ones(1,K);
    % get the pseudoinverse of transposed phi and multiply it with transposed d
    % to get the weight matrix
    weight = pinv(phi') * d';