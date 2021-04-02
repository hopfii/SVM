% logistic sigmoid function (doesn't seem to be built into Octave)
% input parameters:
% s = the value (scalar, matrix) to which the sigmoid function is to be applied
% return values:
% result = the result of applying the sigmoid activation function
function result = Sigmoid(s)
    % logistic sigmoid: 1/(1 + e^-lambda*s)
    % lambda plays no important role for this computation, so it is set to 1
    % Note: the ./ and the exp function make sure this calculation is applied for every value separately
    result = 1./(1 + exp(-s));