function indx=evalute(X,L,lam,lambda,tau)
% function [indx,para_sv]=evalute(p,X,L,method,lam,lambda,tau,DictSize,vie_dim)
% function [idx,obj] = evalute(p,X,L,method,feature_num,lambda,lambda2) 

lab_val=unique(L);
mm=length(lab_val);
dat=[];

for i=1:mm
      dat=[dat;X(L==lab_val(i),:)];
end
X=dat;

nClass  = 10;
para.lambda1 = lam;
para.lambda2 = lambda;
para.lambda3 = tau;
para.iter    = 100;
indx  = feature_selection(X',nClass,para); 
   


