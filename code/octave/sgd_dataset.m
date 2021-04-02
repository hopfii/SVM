% input values
load("K40M2-NN-binary.mat");
% maximum number of epochs
max_epochs = 2000;
% learning parameter
n = 0.05;
% size of output vector
N = 1;
% acceptable total error for epoch
min_e = 1.0e-16;
% use the algorithm to calculate best weight matrix and error vector
[W, errors, g] = NN_learning_SGD(X, D, T, max_epochs, n, N, min_e);
% plot error over epochs (g-1 because g starts at 1 while errors start empty)
semilogy(1:g-1,errors);
% label x axis
xlabel("epochs");
% label y axis
ylabel("quadratic errors");
% print image to file
print -dpng -r300 sgsemilogy.png;
% helper function for ContourPlot(), copyright Hans-Georg Beyer
Set_Global_W(W);
% calculate lower x
xl=min(X(1,:));
% calculate upper x
xu = max(X(1,:));
% caclulate lower y
yl=min(X(2,:));
% calculate upper y
yu = max(X(2,:));
% open new figure window
figure();
% plot the data points as crosses of different colors depending on class
plot(X(1, 1:20), X(2, 1:20), 'b+', X(1, 21:40), X(2,21:40), 'r+');
% set axis limits
axis([xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl))]);
% label x axis
xlabel("x1");
% label y axis
ylabel("x2");
% print image to file
print -dpng -r300 datapointssgd.png;
% open new figure window
figure();
% plot the data points as crosses of different colors depending on class
plot(X(1, 1:20), X(2, 1:20), 'b+', X(1, 21:40), X(2,21:40), 'r+');
% set axis limits
axis([xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl))]);
% label x axis
xlabel("x1");
% label y axis
ylabel("x2");
% hold figure
hold on
% plot decision boundary
ContourPlot("NNF_NN_Sigmoid", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 100, 1);
% stop holding figure
hold off
% print image to file
print -dpng -r300 sgddecisionboundary.png;