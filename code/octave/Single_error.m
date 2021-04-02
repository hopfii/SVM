% function for calculating the quadratic deviation ek
% input parameters:
% y = actual output vector of the neural network
% d = vector with desired outputs of neural network
% return values:
% single_error = the errors of the k input values in a vector
% Note: for the toal error, simply perform sum on the result of this function: sum(error)
function single_error = Single_error(y, d)
    % ek = ||y(xk) - dk||^2
    single_error = norm(y - d).^2;