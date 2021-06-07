% script for comparing results of RBF Kernel SVM with various values for gamma on not linearly seperable dataset
% input values
load("K40M2-NN-binary.mat");
X=X';
D = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];
D=D';
C = 100;

k_func = "rbf_kernel";
set_global_kernel_func(k_func);
set_global_gamma(0.05);
[pos_alpha_1, bias_1, s_vec_1, s_c_vec_1] = SVM_Kernel(X, D, C, k_func);
set_global_gamma(0.1);
[pos_alpha_2, bias_2, s_vec_2, s_c_vec_2] = SVM_Kernel(X, D, C, k_func);
set_global_gamma(0.25);
[pos_alpha_3, bias_3, s_vec_3, s_c_vec_3] = SVM_Kernel(X, D, C, k_func);
set_global_gamma(0.5);
[pos_alpha_4, bias_4, s_vec_4, s_c_vec_4] = SVM_Kernel(X, D, C, k_func);
set_global_gamma(1);
[pos_alpha_5, bias_5, s_vec_5, s_c_vec_5] = SVM_Kernel(X, D, C, k_func);
set_global_gamma(1.5);
[pos_alpha_6, bias_6, s_vec_6, s_c_vec_6] = SVM_Kernel(X, D, C, k_func);


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
set_global_alpha(pos_alpha_1);
set_global_x(s_vec_1);
set_global_y(s_c_vec_1);
set_global_bias(bias_1);
% plot decision boundary
ContourPlot_w_color("NNF_SVM_Kernel", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 200, 1, 'r');

set_global_alpha(pos_alpha_2);
set_global_x(s_vec_2);
set_global_y(s_c_vec_2);
set_global_bias(bias_2);
% plot decision boundary
ContourPlot_w_color("NNF_SVM_Kernel", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 200, 1, 'b');

set_global_alpha(pos_alpha_3);
set_global_x(s_vec_3);
set_global_y(s_c_vec_3);
set_global_bias(bias_3);
% plot decision boundary
ContourPlot_w_color("NNF_SVM_Kernel", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 200, 1, 'g');

set_global_alpha(pos_alpha_4);
set_global_x(s_vec_4);
set_global_y(s_c_vec_4);
set_global_bias(bias_4);
% plot decision boundary
ContourPlot_w_color("NNF_SVM_Kernel", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 200, 1, 'm');

set_global_alpha(pos_alpha_5);
set_global_x(s_vec_5);
set_global_y(s_c_vec_5);
set_global_bias(bias_5);
% plot decision boundary
ContourPlot_w_color("NNF_SVM_Kernel", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 200, 1, 'k');

set_global_alpha(pos_alpha_6);
set_global_x(s_vec_6);
set_global_y(s_c_vec_6);
set_global_bias(bias_6);
% plot decision boundary
ContourPlot_w_color("NNF_SVM_Kernel", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 200, 1, 'y');
% create legend
h = legend('class 1', 'class -1', '\gamma=0.05', '\gamma=0.1', '\gamma=0.25', '\gamma=0.5', '\gamma=1', '\gamma=1.5');
% set legend interpreter
set(h, "interpreter", "tex");
% set legend location
legend("location", "northeast");
% stop holding figure
hold off;
% print image to file
print -dpng -r300 sgdrbfkernelcomp.png;
