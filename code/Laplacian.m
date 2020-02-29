function  L = Laplacian(Z)
   W = (abs(Z)+abs(Z'))/2;
   D=diag(sum(W,2));
   L=D-W;
end