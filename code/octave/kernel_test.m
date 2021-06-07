% script for testing whether polynomial kernel produces same result as scalar product of transformed input vectors for Q matrix
% set kernel parameters
a = 1;
b = 1;
q = 2;
set_global_a_kernel(a);
set_global_b_kernel(b);
set_global_exponent_kernel(q);
% x vector for testing
X = [
    1,2;
    2,1;
    3,3;
    4,1;
    6,2;
    4,9;
    2,10;
    3,9;
    4,8;
    5,8;
    6,9;
    7,5;
];
% perform Q calculation from SVM with kernel
[rows,cols] = size(X);
% calculate Q and c (from 1/2 * x' * Q * x + c * x)
K = ones(rows,rows);
for i=1:rows
    for j=1:rows
        K(i,j) = polynomial_kernel(X(i,:),X(j,:));
    end
end
% transform dimensionality
X_t = Dimensionality_Transform(X);
% perform Q calculation from SVM with transformed x
[rows,cols] = size(X_t);
% calculate Q and c (from 1/2 * x' * Q * x + c * x)
K_t = ones(rows,rows);
for i=1:rows
    for j=1:rows
        K_t(i,j) = dot(X_t(i,:),X_t(j,:));
    end
end
% display kernel K
disp("K with kernel function"), disp(K);
% display K from transformed X
disp("K with transformed vector"), disp(K_t);
% display comparison
disp(K - K_t);