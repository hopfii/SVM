function [w, num_of_misclassifications] = PseudoInv(x, d)
  
  % basic data preparation
  [N, dim] = size(x);
  x(:,dim+1)=ones(N,1);
  x = x';

  % calculating weight with pseudoinverse
  w = inv(x * x') * x * d;

  % check mis_classifications
 num_of_misclassifications = 0;

  for i=1:N
    if sign(w' * x(:,i)) ~= d(i)
      num_of_misclassifications = num_of_misclassifications + 1;
    end
  end