function indx = feature_selection(X,nClass,para)
    [d,n] = size(X);
    
    %parameter setting
    lambda1 = para.lambda1;
    lambda2 = para.lambda2;
    lambda3 = para.lambda3;
    iter_num    = para.iter;

    %initialization
    %[~, A] = vl_kmeans(X,nClass,'verbose','distance','l2','algorithm','elkan');
    F = rand(n,nClass);
    F = F*diag(sqrt(1./(diag(F'*F)+eps))); 
    W = pinv(X')*F;
    Z = ones(n,n)/n;
    
    %F,Z and W 
    Loss  = [];
    iter  = 0 ;
    bStop = 0 ;

   while ~bStop

	iter = iter + 1;

        fprintf('Update Z\n');
        Z = UpdateZ(Z,X,F,lambda2);

        L = Laplacian(Z);
        fprintf('Update F\n'); 
        F = UpdateF(X,W,L,F,lambda1,lambda2,nClass);

        fprintf('Update W\n');
        W = UpdateW(X,F,lambda1,lambda3);
	

        Loss(iter) = lambda2*trace(F'*L*F);
        Loss(iter) = Loss(iter) + norm(X-X*Z,'fro')^2;
        Loss(iter) = Loss(iter) + lambda1*norm(X'*W-F,'fro')^2;
        Loss(iter) = Loss(iter) + lambda3*sum(sqrt(sum(W.*W,2)));

    	if iter > 1
            if iter > iter_num
                bStop = 1;
            else
                if abs(Loss(iter)-Loss(iter-1))/Loss(iter)<=1e-6
                    bStop = 1;
                end
            end
        end
    end
    
    for i=1:d
        w(i)=norm(W(i,:),2);
    end 
    
    [~, indx] = sort(w, 'descend');

end
