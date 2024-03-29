% Note: to use this function, some requirements need to be fulfilled.
% Matlab: install the Optimization Toolbox (HOME -> Addons -> Get Addons -> serach for Optimization Toolbox)
% Octave: load the optim package (on the command line, enter "pkg load optim").
% If the package is not there, install it with: "pkg install -forge optim"
% Function for a hard or soft margin support vector machine
% Input parameters:
% X = the training data, where each row is an input-pattern
% Y = the class data belonging to the training data, where each row is a class
% C = parameter that determines "punishment" for errors -> large values: hard margin, small values: soft margin
% Return values:
% weight = the calculated weight vector w (for classification with w*x + bias)
% bias = the bias (for classification with w*x + bias), calculated from one support vector
% sup_weight = weight vector calculated from support vectors
% avg_bias = bias calculated as average over all suport vectors
function [weight, bias, sup_weight, avg_bias] = SVM(X,Y,C)
    % get the number of rows for X
    [rows,cols] = size(X);
    % calculate Q and c (from 1/2 * x' * Q * x + c * x)
    K = ones(rows,rows);
    for i=1:rows
        for j=1:rows
            K(i,j) = dot(X(i,:),X(j,:));
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
    disp("alpha"), disp(alpha);

    % determine support vectors
    positive = alpha > 10^-10;
    pos_alphas = alpha(positive);
    disp("pos alphas"), disp(pos_alphas);
    support_vecs = X(positive,:);
    support_class_vecs = Y(positive);

    % calculate weight
    weight = sum(alpha.*X.*Y);
    % calculate weight from support vectors
    sup_weight = sum(pos_alphas.*support_vecs.*support_class_vecs);
    % calculate bias from one support vector
    bias = (1/support_class_vecs(1,:)) - dot(weight,support_vecs(1,:)');
    % calculate bias from average of support vectors
    avg_bias = sum(support_class_vecs - support_vecs*weight.')/max(size(support_class_vecs));