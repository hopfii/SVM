% Note: to use this function, some requirements need to be fulfilled.
% Matlab: install the Optimization Toolbox (HOME -> Addons -> Get Addons -> serach for Optimization Toolbox)
% Octave: load the optim package (on the command line, enter "pkg load optim").
% If the package is not there, install it with: "pkg install -forge optim"
% Function for a hard margin support vector machine
% Input parameters:
% X = the training data, where each row is an input-pattern
% D = the class data belonging to the training data, where each row is a class
% Return values:
% weight = the calculated weight vector w (for classification with w*x + bias)
% bias = the bias (for classification with w*x + bias)
function [weight, bias] = Hard_Margin_SVM(X,Y)
    % get the number of rows for X
    [rows,_] = size(X);
    % calculate Q and c (from 1/2 * x' * Q * x + c * x)
    K = ones(rows,rows);
    for i=1:rows
        for j=1:rows
            K(i,j) = dot(X(i,:),X(j,:));
        end
    end
    outer = Y*Y';
    Q = outer * K;
    % make sure Q (Hessian) is symmetric
    Q=(Q+Q')/2;
    c = ones(rows,1) * -1;

    % Equality constraints
    Aeq = Y';
    beq = 0;

    % Inequality constraints
    A = diag(-1*ones(rows,1));
    b = zeros(rows,1);

    % solve optimization
    alpha = quadprog(Q,c,A,b,Aeq,beq);

    % determine support vectors
    positive = alpha > 10^-10;
    % pos_alphas = alpha(positive);
    support_vecs = X(positive,:);
    support_class_vecs = Y(positive);

    % calculate weight and bias
    weight = sum(alpha.*X.*Y);
    bias = (1/support_class_vecs(1,:)) - dot(weight,support_vecs(1,:)');