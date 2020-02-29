function Z  = UpdateZ(Z,X,F,lambda2)
    [d,n] = size(X);
    lapha = 1e-12;
    iter_num = 20;
    P = zeros(n,n);

    X = [X',lapha*ones(n,n)]';
    
    %compute the P
    for i = 1:n        
        tmp = repmat(F(i,:),n,1)-F;
        P(:,i) = sum(tmp.*tmp,2);
    end
    
%     for iter = 1:iter_num
        %update the j-th row of Z
        for j = 1:n
            z = Z(j,:);
            p = P(:,j);
            x = X(:,j);
            
            X_1 = X-(X*Z-x*z);
            v = (X_1'*x)/(x'*x);
            
            tmp = abs(v)-lambda2*p/4;
            Z(j,:) = sign(v).*tmp.*(tmp>0);
            Z(j,j)=0;
        end    
%     end
end
