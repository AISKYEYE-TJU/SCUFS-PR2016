function average = fsMean(input, nanIsZero)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  This function will compute the mean, with support for NaN values.
%  Input:
%   input: the 1xN array of numbers whose mean you wish to have computed.
%   nanIsZero: boolean value telling whether you wish to have NaN's treated
%   as zeros. If not, then they will not be counted when computing the
%   mean.

    count = 0;
    sum = 0;
    
    %if it's a vector, do each col separately
    meanVec = input;
    if size(input, 1) > 1
        meanVec = zeros(1, size(input, 2));
        for i = 1:length(meanVec)
            meanVec(i) = fsMean(input(:,i)', nanIsZero);
        end
    end
    
    for i = 1:length(meanVec)
        if ~isnan(meanVec(i))
            count = count + 1;
            sum = sum + meanVec(i);
        else
            count = count + nanIsZero;
        end
    end
    
    average = sum/count;
end