function [ RD ] = evalFSRedncy( X, fList, numF, maxSel )
RD = zeros(length(numF),1);
if nargin < 4
    maxSel = max(numF) + 1;
end

orderIdx = fList;
for i = 1:length(numF)
    if numF(i) > size(X,2) || numF(i) > maxSel %|| size(orderIdx, 1) < numF(i)
        RD(i) = RD(i-1);
        continue;
    end
    new_X = X(:,(1:numF(i)));
    R = abs(corrcoef(new_X));
    R = R - diag(diag(R));
    RD(i) = fsMean(R,false); %Ignore NaNs
end
