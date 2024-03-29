% script for using rbf kernel SVM with gamma=1 for not linearly seperable dataset
% input values
load("K40M2-NN-binary.mat");
X=X';
D = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];
D=D';
C = 100;

k_func = "rbf_kernel";
set_global_kernel_func(k_func);
set_global_gamma(1);

% use the algorithm to calculate best weight matrix and error vector
[pos_alpha, bias, s_vec, s_c_vec] = SVM_Kernel(X, D, C, k_func);
set_global_alpha(pos_alpha);
set_global_x(s_vec);
set_global_y(s_c_vec);
set_global_bias(bias);
% calculate lower x
xl=min(X(:,1));
% calculate upper x
xu = max(X(:,1));
% caclulate lower y
yl=min(X(:,2));
% calculate upper y
yu = max(X(:,2));
% open new figure window
figure();
% plot the data points as crosses of different colors depending on class
plot(X(1:20,1), X(1:20,2), 'b+', X(21:40,1), X(21:40,2), 'r+');
% set axis limits
axis([xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl))]);
% label x axis
xlabel("x1");
% label y axis
ylabel("x2");
% hold figure
hold on;
% plot decision boundary
ContourPlot("NNF_SVM_Kernel", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 200, 1);
% stop holding figure
hold off;
% print image to file
print -dpng -r300 sgdrbfkernel.png;
