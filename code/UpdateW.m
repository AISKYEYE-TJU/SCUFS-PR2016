function W = UpdateW(X,F,lambda1,lambda3)

% X is the matrix d*n
% F is the matrix n*c
% W is the matrix d*c 

r = lambda3/lambda1;

[W,~] = L21R21_inv(X',F,r);
end
