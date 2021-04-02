% input values
load("K40M2-NN-binary.mat");
X=X'
D = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];
D=D'
% use the algorithm to calculate best weight matrix and error vector
[weight, bias] = Hard_Margin_SVM(Dimensionality_Transform(X), D);
set_global_weight(weight);
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
hold on
% plot decision boundary
ContourPlot("NNF_SVM_Transform", xl-abs(0.1*(xu-xl)), xu+abs(0.1*(xu-xl)), yl-abs(0.1*(yu-yl)), yu+abs(0.1*(yu-yl)), 100, 1);
% stop holding figure
hold off
% print image to file
print -dpng -r300 sgddecisionboundarysvm.png;
