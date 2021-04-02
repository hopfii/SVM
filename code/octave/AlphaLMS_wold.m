% the following function uses the Alpha LMS algorithm from slide 96 to calculate a weight matrix
% input parameters:
% x = the input values in the form of column vectors
% d = the corresponding class values (a row vector)
% alpha = value for changing the weight matrix over the iteration
% delta = value that determines when calculation is finished
% return values:
% w = the calculated weight matrix
% error_save = the vector with the total errors over the iterations
% iteration = the number of total iterations the algorithm needed to finish
function [w, error_save, iteration] = AlphaLMS_wold(x, d, alpha, delta)
    % get rows and columns of inputs x (e.g. 16, 2)
    [M, K] = size(x);
    % random weight vector with 1 row and M+1 (+1  = threshold/theta) columns
    w = rand(M+1,1);
    % add threshold value 1 to input vectors
    x(M+1,:) = 1;
    % initialize vector for saving the total error values over the iterations
    error_save = [];
    % set iteration variable to zero
    iteration = 0;
    % while not finished
    while true
        % remember weight matrix of iteration before this one
        w_old = w;
        % set total error to zero (since this is the start of a new iteration)
        total_error = 0;
        % loop over training set
        for i=1:K
            % get the vector column i
            x_i = x(:, i);
            % calculate the error
            err = (w' * x_i) - d(:,i);
            % calculate the new weight matrix
            w = w - (alpha * (x_i / (norm(x_i))^2) * err);
            % add the error of this loop over the training set to the total error
            total_error = total_error + err^2;
        end
        % display current total error and current iteration
        disp("Total error: "), disp(total_error), disp(" Iteration: "), disp(iteration);
        % check if the normalized difference between the old and the new weight matrix is smaller than delta
        % if yes, we are finished -> abort while true loop
        if(norm((w_old - w)) < delta);
            break;
        end
        % add total error of current iteration to error_save vector (as new row)
        error_save = [error_save, total_error];
        % increment iterations by one
        iteration = iteration + 1;
    end
    
