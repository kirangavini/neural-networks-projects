 Data = imageSet('orl_faces','recursive');
trainingdata=cell(1,240);
 a = 1;

 for j=1:40
     for i=1:6;
         K= read(Data(j),i);
         K=double(K)/256;
         trainingdata{a} = K;
         a = a + 1;
     end;
 end;
rng('default');
hiddenSize1 =[200,300,400];
hiddenSize2 = [100,100,200];
        for q=1:3
            hid1 =hiddenSize1(q);
        autoenc1 = trainAutoencoder(trainingdata,hid1, ...
            'MaxEpochs',100, ...
            'L2WeightRegularization',0.004, ...
            'SparsityRegularization',1, ...
            'SparsityProportion',0.5, ...
            'ScaleData', false);
       
        feat1 = encode(autoenc1,trainingdata);
        hid2=hiddenSize2(q);
        autoenc2 = trainAutoencoder(feat1,hid2, ...
            'MaxEpochs',100, ...
            'L2WeightRegularization',0.004, ...
            'SparsityRegularization',1, ...
            'SparsityProportion',0.5, ...
            'ScaleData', false);
        feat2 = encode(autoenc2,feat1);
        %plotWeights(autoenc2);
        
 j=0;
 for i=1:1:40
     tTrain(i,:)=[ones(1,6),zeros(1,234)];   
     tTrain(i,:)=circshift(tTrain(i,:),[0 j]);
     j=j+6;
 end
 j=0;
 for i=1:1:40
     test(i,:)=[ones(1,4),zeros(1,156)];   
     test(i,:)=circshift(test(i,:),[0 j]);
     j=j+4;
 end
 softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);
 deepnet = stack(autoenc1,autoenc2,softnet);
 testdata=cell(1,160);
  a = 1;

 for j=1:40
     for i=7:10;
         K= read(Data(j),i);
         K=double(K)/256;
         testdata{a} = K;
         a = a + 1;
     end;
 end;
 imageWidth = 92;
imageHeight = 112;
inputSize = imageWidth*imageHeight;
xTest = zeros(inputSize,numel(testdata));
for i = 1:numel(testdata)
    xTest(:,i) = testdata{i}(:);
end
y=zeros(40,160,3);
y1=zeros(40,160,3);
y(:,:,q) = deepnet(xTest);
H2=ezroc3(y(:,:,q),test,2,' ',1);
xTrain = zeros(inputSize,numel(trainingdata));
for i = 1:numel(trainingdata)
    xTrain(:,i) = trainingdata{i}(:);
end
deepnet = train(deepnet,xTrain,tTrain);
y1(:,:,q) = deepnet(xTest);
H3=ezroc3(y1(:,:,q),test,2,' ',1);
end;
Y=y(:,:,1)+y(:,:,2)+y(:,:,3);
H=ezroc3(Y/3,test,2,' ',1);
