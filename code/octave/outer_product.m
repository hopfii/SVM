function outer = outer_product(a, b)
    [a_row, a_col] = size(a);
    [b_row, b_col] = size(b);
    outer = ones(a_row,b_row);
    for i=1:a_row
        for j=1:b_row
            outer(i,j) = a(i,:) * b(j,:);
        end
    end