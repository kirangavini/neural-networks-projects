 Data = imageSet('orl_faces','recursive');
trainingdata=cell(1,240); % Create a cell for training data 
testdata=cell(1,160);     % Create a cell for testing data 
 a = 1;

 for j=1:40
     for i=1:6;
         K= read(Data(j),i);  % Storing training data
         K=double(K)/256;
         trainingdata{a} = K;
         a = a + 1;
     end;
 end;
 a=1;
  for j=1:40
     for i=7:10;
         K= read(Data(j),i);   % Storing testing data 
         K=double(K)/256;
         testdata{a} = K;
         a = a + 1;
     end;
 end;
rng('default');
% Creating first auto encoder
        hiddenSize1 =700;
        autoenc1 = trainAutoencoder(trainingdata,hiddenSize1, ...
            'MaxEpochs',100, ...
            'L2WeightRegularization',0.004, ...
            'SparsityRegularization',1, ...
            'SparsityProportion',0.5, ...
            'ScaleData', false);
       
        feat1 = encode(autoenc1,trainingdata);
 % Creating Second Auto Encoder
        hiddenSize2 = 500;
        autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
            'MaxEpochs',100, ...
            'L2WeightRegularization',0.004, ...
            'SparsityRegularization',1, ...
            'SparsityProportion',0.5, ...
            'ScaleData', false);
        feat2 = encode(autoenc2,feat1);
        
        % Checking AUC for 3 rd auto encoder
        hiddenSize3 = 100;
        autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
            'MaxEpochs',100, ...
            'L2WeightRegularization',0.004, ...
            'SparsityRegularization',1, ...
            'SparsityProportion',0.5, ...
            'ScaleData', false);
        feat3 = encode(autoenc3,feat2);
        
        j=0;
        ttrain=zeros(40,240);
        test=zeros(40,160);
        y=zeros(40,160);
        % Create binary classification for outputs for testing and training 
 for i=1:1:40
     ttrain(i,:)=[ones(1,6),zeros(1,234)];   
     ttrain(i,:)=circshift(ttrain(i,:),[0 j]);
     j=j+6;
 end
 j=0;
 for i=1:1:40
     test(i,:)=[ones(1,4),zeros(1,156)];   
     test(i,:)=circshift(test(i,:),[0 j]);
     j=j+4;
 end
 
 
 imagewidth = 92;
imageheight = 112;
inputSize = imagewidth*imageheight;
xTest = zeros(inputSize,numel(testdata));
for i = 1:numel(testdata)
    xTest(:,i) = testdata{i}(:);
end
xTrain = zeros(inputSize,numel(trainingdata));

for i = 1:numel(trainingdata)
    xTrain(:,i) = trainingdata{i}(:);
end
for i=1:1:40
    softnet = trainSoftmaxLayer(feat3,ttrain(i,:),'MaxEpochs',400);
 deepnet = stack(autoenc1,autoenc2,autoenc3,softnet); 
y(i,:) = deepnet(xTest);
t(i,:)= deepnet(xTrain);
end;
figure(1);
H=ezroc3(y,test,2,' ',1);

figure(2);
H1=ezroc3(t,ttrain,2,' ',1);
for i=1:1:40     % fine tuning
    softnet = trainSoftmaxLayer(feat3,ttrain(i,:),'MaxEpochs',400);
 deepnet = stack(autoenc1,autoenc2,autoenc3,softnet); 
 deepnet = train(deepnet,xTrain,ttrain(i,:));
y(i,:) = deepnet(xTest);
t(i,:) = deepnet(xTrain);
end;
figure(3);
H3=ezroc3(y,tTest,2,' ',1);
figure(4);
H4=ezroc3(t,ttrain,2,' ',1);
