% script for comparing various algorithms on not linearly separable dataset
% input values
X = load("K40M2-NN.dat")';
% class values
D = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
% topology matrix
T = [
0,0,0,1,1,1,1,1,1,1,1,1,1,1;
0,0,0,1,1,1,1,1,0,0,0,0,0,0;
0,0,0,1,1,1,1,1,0,0,0,0,0,0;
0,0,0,0,0,0,0,0,1,1,1,1,1,0;
0,0,0,0,0,0,0,0,1,1,1,1,1,0;
0,0,0,0,0,0,0,0,1,1,1,1,1,0;
0,0,0,0,0,0,0,0,1,1,1,1,1,0;
0,0,0,0,0,0,0,0,1,1,1,1,1,0;
0,0,0,0,0,0,0,0,0,0,0,0,0,1;
0,0,0,0,0,0,0,0,0,0,0,0,0,1;
0,0,0,0,0,0,0,0,0,0,0,0,0,1;
0,0,0,0,0,0,0,0,0,0,0,0,0,1;
0,0,0,0,0,0,0,0,0,0,0,0,0,1;
0,0,0,0,0,0,0,0,0,0,0,0,0,0;
];
% maximum number of epochs
max_epochs = 2000;
% learning parameter
eta = 10^-7;
% learning parameter for NN SGD
n = 0.05;
% size of output vector
N = 1;
% acceptable total error for epoch
min_e = 10^-7;
% alpha for cgd
alpha = 0.1;
% initialize W
% get measurements of topology matrix
[M,K] = size(T);
% randomly initialize W (with size of topology matrix)
W = randn(M,K);
% start clock for time measuring
t0 = clock ();
% use the algorithm to calculate best weight matrix and error vector
% X, D, T, max_epochs, eta, N, eps, f, W
[W_bfgs, errors_bfgs, g_bfgs] = Batch_Learning_Line_Search_LM_BFGS_Vectorstack(X, D, T, max_epochs, eta, N, min_e, "Error_Func", W);
% get duration from last clock (re)start
duration_bfgs = etime (clock (), t0);
% restart clock for time measuring
t0 = clock ();
% use the algorithm to calculate best weight matrix and error vector
[W_cgd, errors_cgd, g_cgd] = Batch_Learning_Line_Search_CGD_Vectorstack(X, D, T, max_epochs, eta, N, min_e, "Error_Func", alpha, W);
% get duration from last clock (re)start
duration_cgd = etime (clock (), t0);
% restart clock for time measuring
t0 = clock ();
% use the algorithm to calculate best weight matrix and error vector
[W_sgd, errors_sgd, g_sgd] = Batch_Learning_Line_Search_SGD(X, D, T, max_epochs, eta, N, min_e, "Error_Func");
% get duration from last clock (re)start
duration_sgd = etime (clock (), t0);
% start clock for time measuring
t0 = clock ();
% use the algorithm to calculate best weight matrix and error vector
[W_nn, errors_nn, g_nn] = NN_learning_SGD(X, D, T, max_epochs, n, N, min_e);
% get duration from last clock (re)start
duration_nnsgd= etime (clock (), t0);

load("K40M2-NN-binary.mat");
X=X';
D = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];
D=D';

C = 0.1;
k_func = "polynomial_kernel";
set_global_b_kernel(1);
set_global_a_kernel(1);
set_global_exponent_kernel(4);
set_global_kernel_func(k_func);
% use the algorithm to calculate needed values
t0 = clock ();
[pos_alpha_poly, bias_poly, s_vec_poly, s_c_vec_poly] = SVM_Kernel(X, D, C, k_func);
duration_polykern= etime (clock (), t0);

C = 100;
k_func = "rbf_kernel";
set_global_kernel_func(k_func);
set_global_gamma(1);
t0 = clock ();
% use the algorithm to calculate best weight matrix and error vector
[pos_alpha_rbf, bias_rbf, s_vec_rbf, s_c_vec_rbf] = SVM_Kernel(X, D, C, k_func);
duration_rbfkern= etime (clock (), t0);

Set_Global_W(W_sgd);
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
hold on
% plot decision boundary SGD
ContourPlot_w_color("NNF_NN_Sigmoid", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 200, 1, 'r');
% set global W to CGD
Set_Global_W(W_cgd);
% plot decision boundary CGD
ContourPlot_w_color("NNF_NN_Sigmoid", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 200, 1, 'b');
% set global W to BFGS
Set_Global_W(W_bfgs);
% plot decision boundary BFGS
ContourPlot_w_color("NNF_NN_Sigmoid", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 200, 1, 'm');
% set global W to NN SGD
Set_Global_W(W_nn);
% plot decision boundary NN SGD
ContourPlot_w_color("NNF_NN_Sigmoid", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 200, 1, 'g');
% plot decision boundary POLY SVM
set_global_alpha(pos_alpha_poly);
set_global_x(s_vec_poly);
set_global_y(s_c_vec_poly);
set_global_bias(bias_poly);
set_global_kernel_func("polynomial_kernel");
ContourPlot_w_color("NNF_SVM_Kernel", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 200, 1, 'y');

set_global_alpha(pos_alpha_rbf);
set_global_x(s_vec_rbf);
set_global_y(s_c_vec_rbf);
set_global_bias(bias_rbf);
set_global_kernel_func("rbf_kernel");
% plot decision boundary RBF SVM
ContourPlot_w_color("NNF_SVM_Kernel", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 200, 1, 'k');

% create legend
h = legend('class -1', 'class 1', 'SGD', 'CGD', 'BFGS', 'NN SGD', 'POLY SVM', 'RBF SVM');
% set legend interpreter
set(h, "interpreter", "tex");
% set legend location
legend("location", "northeast");
% stop holding the figure
hold off
% print image to file
print -dpng -r300 compbatchdecisionboundary.png;
% display time algorithms needed to finish
disp("SGD, time taken: "), disp(duration_sgd);
disp("CGD, time taken: "), disp(duration_cgd);
disp("BFGS, time taken: "), disp(duration_bfgs);
disp("NN SGD, time taken: "), disp(duration_nnsgd);
disp("POLY SVM, time taken: "), disp(duration_polykern);
disp("RBF SVM, time taken: "), disp(duration_rbfkern);