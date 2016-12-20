function [p]=confmatrix(x,y)
for i=1:length(x)
    if (x(i)>=0);
        x(i)=1;
    else
        x(i)=-1;
    end;
end;

TP=0;
FN=0;
FP=0;
TN=0;
for i=1:length(x)
    if(y(i)==1 && x(i)==1)
        TP=TP+1;
    elseif(y(i)==1&& x(i)==-1)
            FP=FP+1;
    elseif(y(i)==-1&& x(i)==1)
                FN=FN+1;
    elseif(y(i)==-1&& x(i)==-1)
             TN=TN+1;
               
    end;
end;
            
FAR= FP/(TN+FP);
GAR= TP/(TP+FN);
FRR= FN/(TP+FN);
GRR= TN/(TN+FP);
p=[GAR,FAR;FRR,GRR];

