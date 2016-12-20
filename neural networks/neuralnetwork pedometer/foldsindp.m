function H1=foldsindptdnn(data,datatest)
testc=num2cell(repmat([ones(1,900),-1*ones(1,2700)],[1,5]));
testt=num2cell(repmat([-1*ones(1,900),ones(1,900),-1*ones(1,1800)],[1,5]));
testr=num2cell(repmat([-1*ones(1,1800),ones(1,900),-1*ones(1,900)],[1,5]));
testd=num2cell(repmat([-1*ones(1,2700),ones(1,900)],[1,5]));
n=10;
h=10;
net1 = timedelaynet(1:n,h);
net1.trainFcn='trainbr';
net1.trainParam.epochs=100;
net1.layers{2}.transferFcn='tansig';

[Xsc,Xic,Aic,Tsc] = preparets(net1,data,testc);
net1 = train(net1,Xsc,Tsc,Xic,Aic);
Yc=sim(net1,datatest);
yc=seq2con(Yc);
yc=yc{1};
h11=[mean((yc(1:900-n))),mean((yc(901-n:1800-n))),mean((yc(1801-n:2700-n))),mean((yc(901-n:1200-n)))];
net2 = timedelaynet(1:n,h);
net2.trainFcn='trainbr';
net2.trainParam.epochs=100;
net2.layers{2}.transferFcn='tansig';

[Xsc,Xic,Aic,Tsc] = preparets(net2,data,testt);
net2 = train(net2,Xsc,Tsc,Xic,Aic);
Yt=sim(net2,datatest);
yt=seq2con(Yt);
yt=yt{1};
h12=[mean((yt(1:300-n))),mean((yt(301-n:600-n))),mean((yt(601-n:900-n))),mean((yt(901-n:1200-n)))];
net3 = timedelaynet(1:n,h);
net3.trainFcn='trainbr';
net3.trainParam.epochs=100;
net3.layers{2}.transferFcn='tansig';

[Xsc,Xic,Aic,Tsc] = preparets(net3,data,testr);
net3 = train(net3,Xsc,Tsc,Xic,Aic);
Yr=sim(net3,datatest);
yr=seq2con(Yr);
yr=yr{1};
h13=[mean((yr(1:300-n))),mean((yr(301-n:600-n))),mean((yr(601-n:900-n))),mean((yr(901-n:1200-n)))];
net4 = timedelaynet(1:n,h);
net4.trainFcn='trainbr';
net4.trainParam.epochs=100;
net4.layers{2}.transferFcn='tansig';

[Xsc,Xic,Aic,Tsc] = preparets(net4,data,testd);
net4 = train(net4,Xsc,Tsc,Xic,Aic);
Yd=sim(net4,datatest);
yd=seq2con(Yd);
yd=yd{1};
h14=[mean((yd(1:300-n))),mean((yd(301-n:600-n))),mean((yd(601-n:900-n))),mean((yd(901-n:1200-n)))];

H1=[h11;h12;h13;h14];
end