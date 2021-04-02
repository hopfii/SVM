% input values
% note: x input is transferred internally into column vectors i.e. x1 = [1; 1; 2] (1 for bias)
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
% class values
d = [1;1;1;1;1;-1;-1;-1;-1;-1;-1;-1];
% calculate x1 min
xl=min(x(:,1));
% calculate x1 max
xu = max(x(:,1));
% calculate x2 min
yl=min(x(:,2));
% calculate x2 max
yu = max(x(:,2));
[weight, bias] = Hard_Margin_SVM(x, d);
[w, num_of_misclassifications] = PseudoInv(x, d);
set_global_weight(weight);
set_global_bias(bias);
% plot x inputs with decision boundary
plot(x(1:4, 1), x(1:4, 2), 'r+', x(5:11, 1), x(5:11, 2), 'g+');
% set axis limits
axis([xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl))]);
% label x axis
xlabel("x1");
% label y axis
ylabel("x2");
hold on;
% plot decision boundary
ContourPlot("NNF_SVM", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 100, 1);
% plot x inputs with decision boundary
plot([xl, xu], [-(xl*w(1) + w(3))/w(2), -(xu*w(1)+w(3))/w(2)]);
% create legend
h = legend('class 1', 'class -1', 'SVM', 'PseudoInv');
% set legend interpreter
set(h, "interpreter", "tex");
% set legend location
legend("location", "northeast");
% stop holding figure
hold off
% print image to file
print -dpng -r300 svmsimple.png;



