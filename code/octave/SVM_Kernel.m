% Note: to use this function, some requirements need to be fulfilled.
% Matlab: install the Optimization Toolbox (HOME -> Addons -> Get Addons -> serach for Optimization Toolbox)
% Octave: load the optim package (on the command line, enter "pkg load optim").
% If the package is not there, install it with: "pkg install -forge optim"
% Function for a soft or hard margin support vector machine using a kernel function
% Input parameters:
% X = the training data, where each row is an input-pattern
% Y = the class data belonging to the training data, where each row is a class
% C = parameter that determines "punishment" for errors -> large values: hard margin, small values: soft margin
% kernel_func = Kernel function to use for scalar products
% Return values:
% pos_alphas = lagrange factors of support vectors
% bias = the bias
% support_vecs = the support vectors
% support_class_vecs = the class labels belonging to the support vectors
function [pos_alphas, bias, support_vecs, support_class_vecs] = SVM_Kernel(X,Y,C, kernel_func)
    % get the number of rows for X
    [rows,cols] = size(X);
    % calculate Q and c (from 1/2 * x' * Q * x + c * x)
    K = ones(rows,rows);
    for i=1:rows
        for j=1:rows
            K(i,j) = feval(kernel_func, X(i,:), X(j,:));
        end
    end
    outer = Y*Y';
    Q = outer .* K;
    % make sure Q is symmetric
    Q=(Q+Q')/2;
    c = ones(rows,1) * -1;

    % Equality constraints
    Aeq = Y';
    beq = 0;

    % Inequality constraints
    A = diag(-1*ones(rows,1));
    b = zeros(rows,1);

    % soft margin lower and upper bounds
    lb = zeros(rows,1);
    ub = C * ones(rows,1);

    % solve optimization
    [alpha,fval,exitflag,output,lambda] = quadprog(Q,c,A,b,Aeq,beq,lb,ub);

    % determine support vectors
    positive = alpha > 10^-10;
    pos_alphas = alpha(positive);
    disp("pos alphas"), disp(pos_alphas);
    support_vecs = X(positive,:);
    support_class_vecs = Y(positive);

    % calculate bias
    polysum = 0;
    for p=1:max(size(pos_alphas))
        polysum = polysum + pos_alphas(p) * support_class_vecs(p) * feval(kernel_func, support_vecs(p,:),support_vecs(1,:));
    end
    bias = support_class_vecs(1) - polysum;
