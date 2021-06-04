% Golden section line sarch for use with Error_Func as function to optimize
% Input parameters:
% x = the starting point for the search
% G = batch Gradient
% eta = starting step size (used as starting delta)
% f = function to optimize
% D = expected outcome
% W = Weight matrix
% N = size if output vector
% Return values:
% optimum = the optimal eta
function optimum = Golden_section_error_wrapper(x, G, eta, f, D, W, N)
    % initialize phi
    phi = 0.618;
    % initalize delta with eta
    delta = eta;
    % set starting point of interval
    a = 10^-7;
    % initialize end point of interval with a
    b = a;
    % variable for exponent
    i = 0;
    % remember the function value with the starting a to know when b is found
    limit = feval(f, x, W + a * G, D, N);
    % loop for calculating end point of search interval
    while 1
        % increase b exponentially by delta
        b = a + (2^i)  * delta;
        % increase exponent by 1
        i = i + 1;
        % if the function value with step size b is bigger than the function value
        % with step size b, we are finished calculating b
        if(feval(f, x, W + b * G, D, N) > limit)
            break;
        end
    end
    % termination criterum
    epsilon = 10^-2;
    % the entire rest of the code performs a golden section line search
    % The length of the interval bracketing the optimal eta is reduced by a factor
    % phi every step. This is achieved by moving either a or b, depending on
    % the function value of f using the respective step size.
    lambda = b - (b - a) * phi;
    f_a = feval(f, x, W + lambda * G, D, N);
    mu = a + (b - a) * phi;
    f_b = feval(f, x, W + mu * G, D, N);
    while b-a > epsilon
        if f_a > f_b
            a = lambda;
            lambda = mu;
            mu = a + (b - a) * phi;
            f_a = f_b;
            f_b = feval(f, x, W + mu * G, D, N);
        else
            b = mu;
            mu = lambda;
            lambda = b - (b - a) * phi;
            f_b = f_a;
            f_a = feval(f, x, W + lambda * G, D, N);
        end
    end
    optimum = 0.5 * (a + b);