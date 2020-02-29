function [ MIres, ACCres ] = evalFSClustKM( X, Y, orderIdx, numF )
% function [ MI, ACC ] = evalFSClustKM( X, Y, wFeat, orderIndicator, numF )
%   X - the data, each row is an instance
%   Y - the class label in form 1,2,3,...
%   wFeat - the weight of features
%   orderIndicator - 1, the small the better, -1 the bigger the better
%   numF - the arrary of the number of features

MIres = zeros(length(numF),1);
ACCres = zeros(length(numF),1);
nC = length(unique(Y));


for i = 1:length(numF)
    if numF(i) > size(X,2)
        break;
    end
    new_X = X(:,orderIdx(1:numF(i)));
    
    % do cluster using kmeans
    numClust = nC;
    options = statset('MaxIter',300);
    options.Display = 'off';
    cidx = kmeans(new_X, numClust, 'dist', 'sqEuclidean', 'emptyaction', 'singleton', 'rep', 10, 'options', options);

    % test performance
    ACCres(i) = calAcc(cidx, Y, nC);
    MIres(i) = MI(cidx, Y, nC);
    
    fprintf('num of feauture: %5i, kmeans acc: %5.3f, mi: %5.3f\n', numF(i), ACCres(i), MIres(i));
end

disp(' ');
end