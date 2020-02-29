    clear;
    clc;
    %the path of data
    s=pwd;
    addpath(s);
    
    datalist{1}='warpAR10P';
    datalist{2}='warpPIE10P';    
    datalist{3}='TOX-171'; 
    datalist{4}='Prostate-GE';    
    datalist{5}='ALLAML'; 
    datalist{6}='GLI-85';
    
    num=[10:10:150];
    lam_ind=-6:6;
    lam=10.^lam_ind;
    
    data_select =1;
  %  data_select =[1:6];

    maxRed = 1000;
    minVal = 0;
 
    s=0;

for pi = 1:1
   for pj = 1:1
      for pk = 1:length(lam)
        for pm=1:length(lam)
            lambda0=1;
            lambda1=1;
            lambda2=lam(pk);
            lambda3=lam(pm);
           
            for i=1:length(data_select)
                fprintf('(lambda0,lambda1,lambda2,lambda3,data)==========(%d,%d,%d)\n',pi,pm,i);
                kk=data_select(i);
                eval(['load ' datalist{kk}]);
                
                fprintf([datalist{kk} '\n']);
                X = NormalizeFea(X);
                sam_num=size(X,1);
                feature_num=size(X,2);

                indx =evalute(X,Y,lambda1,lambda2,lambda3);

                for j=1:length(num)
                    if num(j)<length(indx)
                        feature_num=num(j);             
                        [rec_acc_fs(i,j),rec_clu(i,j),rebunduncy(i,j),rec_acc_clu(i,j)]=evalute_num(X,Y,feature_num,indx);
                    end
                end  
            end
            
            s=s+1;
            recacc{s}=rec_acc_fs';
            recclu{s}=rec_clu';
            rebund{s}=rebunduncy';
    	    recacclu{s}=rec_acc_clu';
	 end
     end
  end
end


for k=1:s
         precacc(k,:)=mean(recacc{k});
         precclu(k,:)=mean(recclu{k});
         prebund(k,:)=mean(rebund{k});
         precacclu(k,:)=mean(recacclu{k});
end

result=[max(precacc);max(precclu);min(prebund);max(precacclu)];

save([datalist{data_select} '_' 'data.mat']);

