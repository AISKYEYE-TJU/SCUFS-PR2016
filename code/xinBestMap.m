%% Written by Xin Zhao (x.zhao@uq.edu.cn). May 2011.
% best mapping that permues clustering result labels to match the ground
% truth labels
%
% Y: ground truth labels
% Y_star: clustering result labels
% 
% mappedYstar
% mappedConfusionMatrix
% acc: clustering accuracy
% nmi: normalized mutual information
%
function [acc, nmi] = xinBestMap(Y,Y_star)

k = max(Y);
n = length(Y);

nY = zeros(k,1);
nYstar = zeros(k,1);
ConfusionMatrix = zeros(k,k);
Perf = zeros(k,k);
for i = 1:k
    nY(i) = sum(Y==i);
    nYstar(i) = sum(Y_star==i);
    for j = 1:k
        ConfusionMatrix(i,j) = sum((Y==i)&(Y_star==j));
        Perf(i,j) = sum(Y==i) - ConfusionMatrix(i,j);        
    end
end

temp = n*ConfusionMatrix./(nY*nYstar');
temp(temp==0) = 1;
temp(isnan(temp)) = 1;
term1 = sum(sum((ConfusionMatrix.*log(temp))));
temp = nY/n;
temp(temp==0) = 1;
temp(isnan(temp)) = 1;
term2 = sum(nY.*log(temp));
temp = nYstar/n;
temp(temp==0) = 1;
temp(isnan(temp)) = 1;
term3 = sum(nYstar.*log(temp));
nmi = term1/((term2*term3)^0.5);

[Matching,cost] = hungarianold(Perf);
% 1-cost/n    % = acc

[YIndex YstarIndex] = find(Matching==1);
mappedYstar = zeros(size(Y_star));
mappedConfusionMatrix = zeros(size(ConfusionMatrix));

for i = 1:k
    mappedYstar(Y_star == YstarIndex(i)) = YIndex(i);
    mappedConfusionMatrix(:,YIndex(i)) = ConfusionMatrix(:,YstarIndex(i));
end

acc = sum(Y == mappedYstar)/n;
