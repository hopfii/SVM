x = [
    1,2; 
    2,1; 
    3,3; 
    4,1; 
    4,9;
    2,10;
    3,9;
    4,8;
    5,8;
    6,9;
    7,5;
    6,2;
];
d = [1;1;1;1;1;-1;-1;-1;-1;-1;-1;-1];

[w, num_of_misclassifications] = PseudoInv(x, d);

% below: plotting and outputs
[N, dim] = size(x);
disp("Misclassifications over "), disp(N), disp(" input samples: "), disp(num_of_misclassifications);

xl=min(x(:,1));
xu = max(x(:,1));
yl=min(x(:,2));
yu = max(x(:,2));

% plot x inputs with decision boundary
plot(x(1:4, 1), x(1:4, 2), 'r+', x(5:11, 1), x(5:11, 2), 'g+', [xl, xu], [-(xl*w(1) + w(3))/w(2), -(xu*w(1)+w(3))/w(2)]);
% set axis limits
axis([xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl))]);
% label x axis
xlabel("x1");
% label y axis
ylabel("x2");

