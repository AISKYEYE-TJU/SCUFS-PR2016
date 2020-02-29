function F = UpdateF(X,W,L,F,lambda1,lambda2,nClass)
    [d,n] = size(X);
    
    alpha = lambda1/lambda2;
    gamma = 1e-6;
    iter = 0;
    bStop = 0;
    objValue = [];
    
    while ~bStop
        iter = iter + 1;
        
        FF = F'*F;
        F = F.*(alpha*X'*W+2*gamma*F)./max(L*F+2*gamma*F*FF+alpha*F, 1e-10);
        k = find(F == Inf);
        if ~isempty(k)
             F(k) = 0;
        end
        
        FF = F'*F;
        F = F*diag(sqrt(1./(diag(F'*F)+eps))); %normalize  
        
        objValue(iter) = calMainObjFuncValue(X, L, F, W, alpha, gamma);
        if iter > 1
            %maxIterNum
            if iter > 30
                bStop = 1;
            else
                %minDiffValue
                if abs(objValue(iter-1) - objValue(iter))/objValue(iter) <= 1e-3
                        bStop = 1;
                end
            end
        end
        
    end
end

function objValue = calMainObjFuncValue(X, L, F, W, alpha, gamma)
% Calculate the main subjective function value.
    [d,c] = size(W);
    objValue = trace(F'*L*F);
    objValue = objValue + alpha*norm(X'*W-F,'fro')^2;
    objValue = objValue + gamma*norm(F'*F-eye(c))^2;
end

