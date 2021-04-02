function [weight, bias] = Hard_Margin_SVM(X,Y)
    [rows,cols] = size(X);
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