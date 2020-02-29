function [acc_mean,acc_std]=crossvalidate(data,fold,method,varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%data is all data with decision
%fold is the number of folds in cross validation
%method is the flag to specify the classification algorithm. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%addpath 'D:\Program Files\MATLAB\R2008a\work\osu_svm3.00'
N=length(varargin);
if N>0
    switch N
        case {1}
            k=cell2mat(varargin(1));
            Gamma=cell2mat(varargin(1));
            q=cell2mat(varargin(1));
            kappa=cell2mat(varargin(1,1));
        case {2}
            delta=cell2mat(varargin(1,1));
            pct=cell2mat(varargin(1,2));
            Gamma=cell2mat(varargin(1,1));
            dell=cell2mat(varargin(1,2));
            kappa=cell2mat(varargin(1,1));
            thre=cell2mat(varargin(1,2));
        case {3}
            k=cell2mat(varargin(1));
            delta=cell2mat(varargin(1,2));
            pct=cell2mat(varargin(1,3));
            kappa=cell2mat(varargin(1));
            lambda1=cell2mat(varargin(1,2));
            lambda2=cell2mat(varargin(1,3));
    end
end
[row column]=size(data);
for i=1:column-1
    data(:,i)=(data(:,i)-min(data(:,i)))/(max(data(:,i))-min(data(:,i)));
end

label=data(:,column);
classnum=max(label);
start1=1;
for i=1:classnum
    [a,b]=find(label==i);
    datai=data(a,:);      %select the i class data 
    [rr1,cc1]=size(datai);
    start1=1;
    %%%%%%%%%part the i class in (fold)%%%%%%%%%%%%%%%%%%%%%
    for j=1:fold-1
        a1=round(length(a)/fold);
        a2=a1-1;
        %fun1=strcat('x*',num2str(a1),'+y*',num2str(a2),'=',num2str(rr1)); 
        %fun2=strcat('x+y=',num2str(fold)); 
        %[x,y]=solve(fun1,fun2) 
        %[x,y] = solve('x*a1+a2*y=rr1','x+y=fold')
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        A=[a1 a2;1 1];
        b=[rr1 fold]';
        x=A\b;
        if (j<x(1)+1)
            everynum=a1;
        else
            everynum=a2;
        end
        start2=start1+everynum-1;       
        eval(['data' num2str(i) num2str(j) '=datai([start1:start2],:);']);
        start1=start2+1;
    end
    eval(['data' num2str(i) num2str(fold) '=datai([start1:length(a)],:);']);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=1:fold
    eval(['part' num2str(j) '=[];']);
    for i=1:classnum
      eval(['part' num2str(j) '=[part' num2str(j) ';data' num2str(i) num2str(j) '];']);
    end   
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

accu_m=[];
label1=[];
testL1=[];
for j=1:fold
    Samples=[];
     Labels=[];
     testS=[];
     testL=[];
    for i=1:fold
        
        if (i~=j)
            eval(['Samples=[Samples;part' num2str(i) '(:,1:column-1)];'])
            eval(['Labels=[Labels;part' num2str(i) '(:,column)];'])
        end
    end
    eval(['testS=part' num2str(j) '(:,1:column-1);'])
    eval(['testL=part' num2str(j) '(:,column);'])
    switch method
        case 'CART' %CART classifier       
            t = treefit(Samples,Labels,'method','classification');  % create decision tree
            sfit = treeval(t,testS);      % find assigned class numbers      
            ClassRate=length(find((sfit-testL)==0))/length(testL);
            label1=[label1, sfit' ];
            testL1=[testL1;testL];
            len(j)=length(testL);
        case 'LSVM'  %Linear-SVM classifier
            [AlphaY, SVs, Bias, Parameters, nSV, nLabel] =LinearSVC(Samples', Labels');
            AlSv=[];
            wtemp=[];
            for m=1:length(AlphaY)
                AlSv(:,m)=AlphaY(m).*SVs(:,m);
            end
            %wtemp=sum(AlSv')
            [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(testS', testL', AlphaY, SVs, Bias,Parameters, nSV, nLabel);      
     
        case 'Margin_Tree' % C4.5 classifier
            test_targets = Margin_Tree(Samples', Labels', testS',k,100);    
            ClassRate=length(find((test_targets'-testL)==0))/length(testL); 
        case 'C45' % C4.5 classifier
            training=[Samples,Labels];
            row1=size(training,1);
            testing=[testS,testL];
            data1=[training;testing];
            run1=Arsenal('classify  -- Train_Test_Validate -- WekaClassify -MultiClassWrapper 0 -- trees.J48',data1,row1);
            test_targets=run1.Y_pred(:,3);
            ClassRate=length(find((test_targets-testL)==0))/length(testL);      
        case 'adaboost' % C4.5 classifier
            training=[Samples,Labels];
            row1=size(training,1);
            testing=[testS,testL];
            data1=[training;testing];
            run1=Arsenal('classify  -- Train_Test_Validate -- MCAdaBoostM1 -Iter 10 -- WekaClassify -MultiClassWrapper 0 -- trees.SimpleCart',data1,row1);
            test_targets=run1.Y_pred(:,3);
            ClassRate=length(find((test_targets-testL)==0))/length(testL); 
            
        case 'NEC' % NEC classifier
            training=[Samples,Labels];
            label=NEC(training,testS,2,k);
            ClassRate=length(find((label'-testL)==0))/length(testL);
        case 'KNN' %KNN classifier
           ind=knnsearch(Samples,testS);
           label=Labels(ind);
           ClassRate=length(find((label-testL)==0))/length(testL);            
        case 'CART_SR' %CART classifier    
            training=[Samples,Labels];
            feature_slct=NRS_random_FW_FS(training,k,0.001,1);  %find the feature subspace 
            t = treefit(Samples(:,feature_slct),Labels,'method','classification');  % create decision tree
            lab1= treeval(t,testS(:,feature_slct));     %get the classification result of the test set   
            ClassRate=length(find((lab1-testL)==0))/length(testL);
        case 'C45_SR' % C4.5 classifier
            training=[Samples,Labels];
            feature_slct=NRS_random_FW_FS(training,0.1,0.001,1);  %find the feature subspace 
            row1=size(training,1);
            testing=[testS,testL];
            data1=[training;testing];
            run1=Arsenal('classify  -- Train_Test_Validate -- WekaClassify -MultiClassWrapper 0 -- LibSVM -Kernel 0 -CostFactor 100',data1(:,[feature_slct,column]),row1);
            test_targets=run1.Y_pred(:,3);
            ClassRate=length(find((test_targets-testL)==0))/length(testL); 
        case 'rbfsvm'
            model = svmtrain(Labels,Samples,'-t 2 -c 1 -g 0.2');
            [predict_label, accuracy, dec_values] = svmpredict(testL, testS, model); % 
            ClassRate=length(find((predict_label-testL)==0))/length(testL);
       case 'lsvm'
            model = svmtrain(Labels,Samples,'-t 0 -c 1');
            [predict_label, accuracy, dec_values] = svmpredict(testL, testS, model); % 
            ClassRate=length(find((predict_label-testL)==0))/length(testL);
        case 'CRC'
             tr_dat=Samples';
             tt_dat=testS';
             trls=Labels;
             ttls=testL;
             [ID,ClassRate]=CRC(tr_dat,tt_dat,trls,ttls',kappa);
         case 'NSC'
             tr_dat=Samples';
             tt_dat=testS';
             trls=Labels;
             ttls=testL;
             [ClassRate]=NNSC(tr_dat,tt_dat,trls,ttls',kappa,0); 
           case 'RSRC'
             tr_dat=Samples';
             tt_dat=testS';
             trls=Labels;
             ttls=testL;           
             ClassRate=RSNSC(tr_dat,tt_dat,trls,ttls',0.005,kappa,0);
          case 'SRC'
             tr_dat=Samples';
             tt_dat=testS';
             trls=Labels;
             ttls=testL;
             [ID,ClassRate]=SRC(tr_dat,tt_dat,trls,ttls',kappa);
        case 'CRSVM'
             tr_dat=Samples';
             tt_dat=testS';
             trls=Labels;
             ttls=testL;
             [ClassRate]=SVMDIC(tr_dat,tt_dat,trls,ttls,kappa,thre);
        case 'LMNS'
             tr_dat=Samples';
             tt_dat=testS';
             trls=Labels';
             ttls=testL';             
             [ClassRateU,ClassRate]=digitrc(tr_dat,tt_dat,trls,ttls,kappa,0,0,kappa);
         case 'nnfast'  
              ClassRate = knn_fast(Samples,testS,Labels,testL);
    end
    accu_m(j)=ClassRate;  
end
acc_mean=mean(accu_m);
acc_std=std(accu_m);

