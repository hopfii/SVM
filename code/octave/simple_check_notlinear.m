% script for comparing Hard Margin SVM with Soft Margin SVM on dataset with some outliers that is otherwise linearly separable
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
% start with large C -> Hard margin svm
C = 2000;
% calculate x1 min
xl=min(x(:,1));
% calculate x1 max
xu = max(x(:,1));
% calculate x2 min
yl=min(x(:,2));
% calculate x2 max
yu = max(x(:,2));
[weight, bias] = SVM(x, d, C);
% rerun with small C -> Soft margin SVM
C = 0.1;
[weight_2, bias_2] = SVM(x, d, C);
% weight = [0, 0.00011063];
% bias = 0.9997787475585938;
set_global_weight(weight);
set_global_bias(bias);
% plot x inputs with decision boundary
plot(x(1:5, 1), x(1:5, 2), 'r+', x(6:12, 1), x(6:12, 2), 'g+');
% set axis limits
axis([xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl))]);
% label x axis
xlabel("x1");
% label y axis
ylabel("x2");
% hold figure
hold on;
% plot decision boundary
ContourPlot_w_color("NNF_SVM", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 500, 1, 'b');
% set global weight and bias
set_global_weight(weight_2);
set_global_bias(bias_2);
% plot decision boundary
ContourPlot_w_color("NNF_SVM", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 500, 1, 'k');
% create legend
h = legend('class 1', 'class -1', 'Hard margin SVM', 'Soft margin SVM');
% set legend interpreter
set(h, "interpreter", "tex");
% set legend location
legend("location", "northeast");
% stop holding figure
hold off
% print image to file
print -dpng -r300 svmsimplenotlinear.png;



