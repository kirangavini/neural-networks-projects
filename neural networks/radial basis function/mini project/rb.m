load('P.mat');
load('T.mat');
[trainP,valP,testP,trainInd,valInd,testInd]=dividerand(P,0.6,0.2,0.2);
[trainT,valT,testT]=divideind(T,trainInd,valInd,testInd);
GOAL=0.5;

for SPREAD1=10:3:150;
net1=newrbe(trainP,trainT,SPREAD1);	
y1T=sim(net1,testP);
mse1T=mse(y1T-testT);
if (mse1T <= 0.5)
 display(SPREAD1)
 display(mse1T)
  break
   
end


% y1v=sim(net1,valP);
% mse1v=mse(y1v-valT);
% if (mse1v <= 0.5)
%  display(SPREAD1)
%  display(mse1v)
%   break
%    
% end
end

for SPREAD2=10:3:150;
net2=newrb(trainP,trainT,GOAL,SPREAD2);	
y2T=sim(net2,testP);
mse2T=mse(y2T-testT);
if (mse2T <= 0.5)
 display(SPREAD2)
 display(mse2T)
  break
end

% y2v=sim(net2,valP);
% mse2v=mse(y2v-valT);
% if (mse2v <= 0.5)
%  display(SPREAD2)
%  display(mse2v)
%   break
%    
% end
end


 y1v= sim(net1,valP);
 y1Tr= sim(net1,trainP);
 y1T=sim(net1,testP);
  c=confmatrix(y1v,valT);
  c1=confmatrix(y1Tr,trainT);
  c2=confmatrix(y1T,testT);
 y2v=sim(net2,valP);
 y2Tr=sim(net2,trainP);
 y2T=sim(net2,testP);
  c3=confmatrix(y2v,valT);
 c4=confmatrix(y2Tr,trainT);
  c5=confmatrix(y2T,testT);

 roc= ezroc3((y1v+1)/2,(valT+1)/2,2,'ROC',1);
 roc1= ezroc3((y1T+1)/2,(testT+1)/2,2,'ROC',1);
 roc2= ezroc3((y1Tr+1)/2,(trainT+1)/2,2,'ROC',1);
  roc3= ezroc3((y2v+1)/2,(valT+1)/2,2,'ROC',1);
 roc4= ezroc3((y2T+1)/2,(testT+1)/2,2,'ROC',1);
 roc5= ezroc3((y2Tr+1)/2,(trainT+1)/2,2,'ROC',1);

 
 
 

