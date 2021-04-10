function X_trans = Dimensionality_Transform(X)
    % Note: still need to find the right transformation!
    X_trans = [X(:,1) * 0 + 1, X(:,1).^2, X(:,2).^2, sqrt(2)*X(:,1), sqrt(2) * X(:,2), sqrt(2) * X(:,1) .* X(:,2)];